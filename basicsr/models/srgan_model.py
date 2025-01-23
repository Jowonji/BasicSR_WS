import torch
from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel

@MODEL_REGISTRY.register()
class SRGANModel(SRModel):
    """SRGAN 모델: 단일 이미지 초해상도(Single Image Super-Resolution)를 위한 모델."""

    def init_training_settings(self):
        """학습 설정 초기화 메서드."""
        train_opt = self.opt['train']

        # EMA(Exponential Moving Average) 설정
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # EMA를 사용하는 생성기 네트워크(net_g_ema) 정의
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # 사전 학습된 모델 로드
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # net_g의 가중치를 복사
            self.net_g_ema.eval()

        # 판별기 네트워크(net_d) 정의
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)  # 판별기 네트워크 구조 출력

        # 사전 학습된 모델 로드
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        # 네트워크 학습 모드 설정
        self.net_g.train()
        self.net_d.train()

        # 손실 함수 정의
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # 판별기 학습 설정
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # 옵티마이저와 스케줄러 설정
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """옵티마이저 설정."""
        train_opt = self.opt['train']
        # 생성기 옵티마이저 설정
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # 판별기 옵티마이저 설정
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        """생성기와 판별기의 매개변수를 최적화."""
        # -------------------------------
        # 생성기 최적화 단계
        # -------------------------------
        for p in self.net_d.parameters():
            p.requires_grad = False  # 판별기의 매개변수 고정

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)  # 생성기 출력 계산

        l_g_total = 0  # 생성기 손실 총합 초기화
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # 픽셀 손실 계산
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # Perceptual 손실 계산
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # GAN 손실 계산
            fake_g_pred = self.net_d(self.output)  # 생성된 이미지에 대한 판별기 출력
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # 역전파 및 생성기 최적화
            l_g_total.backward()
            self.optimizer_g.step()

        # -------------------------------
        # 판별기 최적화 단계
        # -------------------------------
        for p in self.net_d.parameters():
            p.requires_grad = True  # 판별기의 매개변수 업데이트 활성화

        self.optimizer_d.zero_grad()
        # 실제(real) 이미지 손실 계산
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # 생성(fake) 이미지 손실 계산
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        # 손실 값 로그 기록
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA 업데이트
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        """모델과 학습 상태 저장."""
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
