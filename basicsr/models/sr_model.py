import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import os
import numpy as np
import matplotlib.cm as cm
import imageio


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """단일 이미지 초해상도(SR, Super-Resolution)을 위한 기본 모델"""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 네트워크 정의
        self.net_g = build_network(opt['network_g']) # 생성기 네트워크 생성
        self.net_g = self.model_to_device(self.net_g) # 네트워크를 디바이스(GPU/CPU)로 이동
        self.print_network(self.net_g) # 네트워크 구조 출력

        # 사전 학습된 모델 로드
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        """훈련을 위한 설정 초기화"""
        self.net_g.train() # 네트워크를 훈련 모드로 설정
        train_opt = self.opt['train'] # 훈련 관련 옵션 불러오기

        # EMA(Exponential Moving Average) 적용 여부 확인
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device) # EMA 네트워크 생성
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # 네트워크 가중치 복사
            self.net_g_ema.eval() # EMA 네트워크를 평가 모드로 설정

        # 손실 함수 정의
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # 옵티마이저 및 학습 스케줄러 설정
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """옵티마이저 설정"""
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters(): # 학습 가능한 매개변수 수집
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """입력 데이터를 장치로 이동"""
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        """훈련 과정에서 역전파 및 가중치 갱신 수행"""
        self.optimizer_g.zero_grad() # 기존의 기울기 초기화
        self.output = self.net_g(self.lq) # 네트워크 예측 수행

        l_total = 0
        loss_dict = OrderedDict()
        
        # 픽셀 손실 계산
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # 지각 손실 계산
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # 역전파 수행
        l_total.backward()
        self.optimizer_g.step() # 옵티마이저 스텝 실행하여 가중치 업데이트

        self.log_dict = self.reduce_loss_dict(loss_dict) # 손실 값 기록

        # EMA(Exponential Moving Average) 업데이트
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """모델 테스트 수행 (훈련되지 않은 상태에서 수행)"""
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        """테스트 시 여러 변환(augmentation)을 적용하여 결과를 평균화하는 self-ensemble 기법 사용"""
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            """입력 텐서를 다양하게 변환(좌우/상하 반전, 전치) 수행"""
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy() # 상하 반전
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy() # 좌우 반전
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy() # 전치

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # 변환된 입력 텐서 생성
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # 모델 추론 수행
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # 변환된 출력을 원래 형태로 복구
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 데이터셋 이름 가져오기
        dataset_name = dataloader.dataset.opt['name']

        # HR 데이터 범위 가져오기
        hr_min = dataloader.dataset.hr_min
        hr_max = dataloader.dataset.hr_max

        # 검증에서 사용할 메트릭(metric)이 정의되어 있는지 확인
        with_metrics = self.opt['val'].get('metrics') is not None

        # 진행 상태 표시 여부 확인 (progress bar)
        use_pbar = self.opt['val'].get('pbar', False)

        # 메트릭 결과 초기화
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # 최초 실행시
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        # Progress bar 설정
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # 데이터 로더 반복문
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            # 데이터를 모델에 입력
            self.feed_data(val_data)
            self.test()  # 테스트 실행

            # 모델 출력 가져오기
            visuals = self.get_current_visuals()

            # GT 이미지 확인
            if 'gt' not in visuals:
                raise ValueError(f"GT image is missing for {img_name}. Validation cannot proceed.")

            # 1. 모델 출력 및 GT 변환 (Tensor → NumPy)
            sr_tensor = visuals['result'].cpu().detach().numpy()
            gt_tensor = visuals['gt'].cpu().detach().numpy()

            # 2. 모든 불필요한 차원 제거 (강제 squeeze 적용)
            sr_tensor = visuals['result'].cpu().detach().numpy()
            gt_tensor - visuals['gt'].cpu().detach().numpy()

            # 3. 차원 체크 (H, W 형태가 아닐 경우 강제 변환)
            if sr_tensor.ndim != 2:
                print(f"⚠ Warning: sr_tensor has incorrect shape {sr_tensor.shape}, forcing reshape.")
                sr_tensor = sr_tensor.reshape((sr_tensor.shape[-2], sr_tensor.shape[-1]))  # (H, W) 변환
            if gt_tensor.ndim != 2:
                print(f"⚠ Warning: gt_tensor has incorrect shape {gt_tensor.shape}, forcing reshape.")
                gt_tensor = gt_tensor.reshape((gt_tensor.shape[-2], gt_tensor.shape[-1]))  # (H, W) 변환

            #print(f"✅ After fixing: sr_tensor.shape={sr_tensor.shape}, gt_tensor.shape={gt_tensor.shape}")

            # 4. 역정규화 (원래 풍속값 범위로 변환)
            sr_img_rescaled = sr_tensor * (hr_max - hr_min) + hr_min
            gt_img_rescaled = gt_tensor * (hr_max - hr_min) + hr_min

            # RMSE 계산 전에 NaN 및 Inf 체크
            if np.isnan(sr_img_rescaled).any() or np.isnan(gt_img_rescaled).any():
                raise ValueError(f"NaN detected in SR or GT image for {img_name}. Check normalization or model output.")
            if np.isinf(sr_img_rescaled).any() or np.isinf(gt_img_rescaled).any():
                raise ValueError(f"Infinite values detected in SR or GT image for {img_name}. Check model stability.")
            
            # RMSE 계산 전에 크기 확인
            if sr_img_rescaled.shape != gt_img_rescaled.shape:
                raise ValueError(f"Shape mismatch: SR {sr_img_rescaled.shape} vs GT {gt_img_rescaled.shape} for {img_name}")

            # 4. RMSE 계산을 위해 [0,255] 정규화 적용
            metric_data = {
                'img': ((sr_img_rescaled - hr_min) / (hr_max - hr_min) * 255).astype(np.float32),
                'img2': ((gt_img_rescaled - hr_min) / (hr_max - hr_min) * 255).astype(np.float32)
            }

            # 5. SR 및 GT 이미지 범위 확인
            #print(f"SR Image Range: min={sr_img_rescaled.min()}, max={sr_img_rescaled.max()}")
            #print(f"GT Image Range: min={gt_img_rescaled.min()}, max={gt_img_rescaled.max()}")

            # 6. GPU 메모리 관리
            del self.lq, self.output
            torch.cuda.empty_cache()

            # 5. 결과 이미지 저장
            if save_img:
                if sr_img_rescaled.ndim == 2:
                    epsilon = 1e-8
                    sr_img_normalized = (sr_img_rescaled - sr_img_rescaled.min()) / (
                        max(sr_img_rescaled.max() - sr_img_rescaled.min(), epsilon)
                    )
                    sr_img_colormap = cm.viridis(sr_img_normalized)[:, :, :3]  # Viridis 컬러맵 적용
                    sr_img_colormap = (sr_img_colormap * 255).astype(np.uint8)

                    # 저장 경로 설정
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{current_iter}.png')
                    save_dir = osp.dirname(save_img_path)
                    os.makedirs(save_dir, exist_ok=True)  # 폴더 생성

                    # 이미지 저장
                    try:
                        imageio.imwrite(save_img_path, sr_img_colormap)
                        print(f"✅ Image successfully saved at {save_img_path}")
                    except Exception as e:
                        print(f"❌ Failed to save image at {save_img_path}. Error: {e}")

            # 8. 메트릭 계산
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            # 9. Progress bar 업데이트
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        # 10. Progress bar 닫기
        if use_pbar:
            pbar.close()

        # 11. 최종 메트릭 계산 및 로그 출력
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)