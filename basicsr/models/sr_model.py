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


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
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

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
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
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
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
        hr_min = dataloader.dataset.hr_min  # 데이터셋에서 hr_min 값 가져오기
        hr_max = dataloader.dataset.hr_max  # 데이터셋에서 hr_max 값 가져오기

        # 검증에서 사용할 메트릭(metric)이 정의되어 있는지 확인
        with_metrics = self.opt['val'].get('metrics') is not None

        # 진행 상태 표시 여부 확인 (progress bar)
        use_pbar = self.opt['val'].get('pbar', False)

        # 메트릭 결과 초기화
        if with_metrics:
            if not hasattr(self, 'metric_results'): # 최초 실행시
                # 메트릭 이름을 키로 하는 딕셔너리 생성
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # 여러 데이터셋에 대해 최고 메트릭 결과 초기화
            self._initialize_best_metric_results(dataset_name)

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
        
        # 메트릭 계산에 사용할 데이터를 저장하는 딕셔너리
        metric_data = dict()

        # Progress bar 설정 (옵션에서 활성화한 경우)
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # 데이터 로더 반복문 (이미지 처리)
        for idx, val_data in enumerate(dataloader):
            # 현재 이미지 이름 가져오기
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            # 데이터를 모델에 입력
            self.feed_data(val_data)
            self.test() # 테스트 실행

            # 모델 출력 가져오기
            visuals = self.get_current_visuals()

            # **모델 출력 범위 확인**
            sr_tensor = visuals['result']  # 모델 출력 텐서
            print(f"Raw SR Tensor Range: min={sr_tensor.min().item()}, max={sr_tensor.max().item()}")

            # SR 이미지를 numpy 배열로 변환
            sr_img = tensor2img([visuals['result']]) # 모델 결과 반환
            metric_data['img'] = sr_img # 메트릭 계산용 데이터에 추가
            
            # GT 이미지 처리
            if 'gt' in visuals: # Ground Truth가 존재하면
                gt_img = tensor2img([visuals['gt']]) # GT 반환
                metric_data['img2'] = gt_img # 메트릭 계산용 데이터에 추가
                del self.gt # 메모리 절약을 위해 제거

            # **GT와 SR 데이터의 범위 확인**
            if 'gt' in visuals:
                print(f"GT Image Range: min={gt_img.min()}, max={gt_img.max()}")

            # GPU 메모리 관리
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # 결과 이미지 저장
            if save_img:
                # 결과 이미지 저장
                if sr_img.ndim == 2:  # 예: (H, W)
                    # 1. SR 이미지 값의 범위 출력 (디버깅용)
                    print(f"SR Image Range: min={sr_img.min()}, max={sr_img.max()}")

                    # 2. GT 값 (HR 데이터의 범위)을 기준으로 역정규화
                    sr_img_rescaled = sr_img * (hr_max - hr_min) + hr_min  # Rescale to original GT range
                    print(f"Rescaled SR Image Range: min={sr_img_rescaled.min()}, max={sr_img_rescaled.max()}")

                    # 3. 이미지 시각화를 위해 정규화 (0-1 범위로 변환)
                    epsilon = 1e-8  # 작은 값 추가로 안정성 보장
                    sr_img_normalized = (sr_img_rescaled - sr_img_rescaled.min()) / (
                        max(sr_img_rescaled.max() - sr_img_rescaled.min(), epsilon)
                    )
                    print(f"Normalized SR Image Range: min={sr_img_normalized.min()}, max={sr_img_normalized.max()}")

                    # 4. viridis 컬러맵 적용
                    sr_img_colormap = cm.viridis(sr_img_normalized)  # Apply viridis colormap
                    sr_img_colormap = (sr_img_colormap[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255]
                else:
                    raise ValueError("Expected a single-channel image for colormap, but got shape: {}".format(sr_img.shape))

                # 학습 중일 경우 저장 경로
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    # 검증 중일 경우 저장 경로
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], 
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png'
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], 
                            dataset_name,
                            f'{img_name}_{self.opt["name"]}.png'
                        )

                # 컬러맵 적용한 이미지를 저장
                imwrite(sr_img_colormap, save_img_path)

                # 저장된 이미지를 확인
            print(f"Saved Image: {save_img_path}, Shape: {sr_img_colormap.shape}")

            # 메트릭 계산
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            # progress bar 업데이트
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        
        # progress bar 닫기
        if use_pbar:
            pbar.close()

        # 최종 메트릭 계산 및 로그 출력
        if with_metrics:
            for metric in self.metric_results.keys():
                # 메트릭 평균 계산
                self.metric_results[metric] /= (idx + 1)
                # 최고 메트릭 결과 업데이트
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            # 검증 메트릭 결과를 로그에 기록
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