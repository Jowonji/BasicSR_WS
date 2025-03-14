import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt  # Wavelet Transform 라이브러리
import numpy as np
import skimage.feature

# Self-Similarity Loss (자기 유사성 유지)
def patch_similarity_loss(pred, target, patch_size=8):
    B, C, H, W = pred.shape
    loss = 0.0
    count = 0

    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            pred_patch = pred[:, :, i:i+patch_size, j:j+patch_size]
            target_patch = target[:, :, i:i+patch_size, j:j+patch_size]

            if pred_patch.shape[2:] == (patch_size, patch_size):
                # **채널(C) 차원까지 포함하여 유사도를 비교**
                sim = F.cosine_similarity(pred_patch.permute(0, 2, 3, 1).flatten(1),
                                          target_patch.permute(0, 2, 3, 1).flatten(1), dim=1)
                l1_diff = F.l1_loss(pred_patch, target_patch)

                loss += torch.mean(1 - sim) + 0.1 * l1_diff  # 유사성 + 픽셀 차이 고려
                count += 1

    return loss / count if count > 0 else loss

# WaveletHighFrequencyLoss 클래스 - pywt import 오류 수정
class WaveletHighFrequencyLoss(nn.Module):
    def __init__(self):
        super(WaveletHighFrequencyLoss, self).__init__()

        # Haar Wavelet 필터 (Low-Pass, High-Pass)
        self.haar_filter = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32).view(1, 1, 2, 2)
        self.haar_high_filter = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2)

    def wavelet_transform(self, img):
        """
        Haar Wavelet Transform을 사용하여 1채널 이미지의 고주파 성분을 분리.
        """
        img = F.pad(img, (1, 1, 1, 1), mode='reflect')  # 가장자리 반사 패딩 추가
        low_freq = F.conv2d(img, self.haar_filter.to(img.device), stride=2)
        high_freq = F.conv2d(img, self.haar_high_filter.to(img.device), stride=2)

        return torch.abs(high_freq)  # 고주파 성분 반환

    def sobel_filter(self, img):
        """
        Sobel Filter를 사용하여 1채널 이미지의 경계선 정보를 추출.
        """
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img.device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img.device)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)

        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)  # Sobel Edge Magnitude

    def forward(self, pred, target):
        """
        예측된 이미지와 실제 이미지의 고주파 성분을 비교하는 Loss 계산.
        """
        pred_wavelet = self.wavelet_transform(pred)
        target_wavelet = self.wavelet_transform(target)

        pred_sobel = self.sobel_filter(pred)
        target_sobel = self.sobel_filter(target)

        # L1 Loss로 고주파 성분 차이를 비교
        loss_wavelet = F.l1_loss(pred_wavelet, target_wavelet)
        loss_sobel = F.l1_loss(pred_sobel, target_sobel) * 0.5  # Sobel 가중치 조절

        return loss_wavelet + loss_sobel  # 두 가지 고주파 손실을 합산

# CharbonnierLoss (RMSE 최적화)
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):  # 더 안정적인 epsilon 값으로 조정
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


from basicsr.utils import get_root_logger

@MODEL_REGISTRY.register()
class ESRGANModel_V2(SRGANModel):
    """풍속 데이터 초해상화 (SR) 최적화 ESRGAN Model"""

    def __init__(self, opt):
        super(ESRGANModel_V2, self).__init__(opt)

        # ✅ 기존 Charbonnier Loss (픽셀 기반 손실)
        self.cri_pix = CharbonnierLoss().to(self.device) if self.opt['train'].get('pixel_opt') else None

        # ✅ SSL 기반 자기 유사성 손실 추가
        self.cri_ssl = patch_similarity_loss if self.opt['train'].get('ssl_opt') else None
        self.ssl_weight = self.opt['train'].get('ssl_weight', 0.05)

        # ✅ Wavelet 기반 고주파 손실 추가
        self.cri_wavelet = WaveletHighFrequencyLoss().to(self.device) if self.opt['train'].get('wavelet_opt') else None
        self.wavelet_weight = self.opt['train'].get('wavelet_weight', 0.05)

        # ✅ Logger 추가
        self.logger = get_root_logger()
        self.logger.info(f'Loss functions initialized - SSL: {self.ssl_weight}, Wavelet: {self.wavelet_weight}')

    def optimize_parameters(self, current_iter):
        # -------------------------------
        # ✅ 생성기 네트워크(net_g) 최적화
        # -------------------------------
        for p in self.net_d.parameters():
            p.requires_grad = False  # 판별기 고정

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # ✅ 생성기의 총 손실 초기화
        l_g_total = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        loss_dict = OrderedDict()

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # ✅ 1. Charbonnier Loss (픽셀 기반 손실)
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix.detach()

            # ✅ 2. SSL(Self-Similarity Loss) 적용
            if self.cri_ssl:
                l_g_ssl = self.cri_ssl(self.output, self.gt)
                weighted_ssl = self.ssl_weight * l_g_ssl
                l_g_total += weighted_ssl
                loss_dict['l_g_ssl'] = weighted_ssl.detach()

            # ✅ 3. Wavelet High-Frequency Loss 적용
            if self.cri_wavelet:
                l_g_wavelet = self.cri_wavelet(self.output, self.gt)
                weighted_wavelet = self.wavelet_weight * l_g_wavelet
                l_g_total += weighted_wavelet
                loss_dict['l_g_wavelet'] = weighted_wavelet.detach()

            # ✅ 4. GAN Loss (Relativistic GAN)
            real_d_pred = self.net_d(self.gt).detach()
            fake_g_pred = self.net_d(self.output)

            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan.detach()

            # ✅ 생성기 손실 역전파 및 업데이트
            l_g_total.backward()
            self.optimizer_g.step()

        # -------------------------------
        # ✅ 판별기 네트워크(net_d) 최적화
        # -------------------------------
        for p in self.net_d.parameters():
            p.requires_grad = True  # 판별기 학습 가능하게 변경

        self.optimizer_d.zero_grad()

        fake_d_pred = self.net_d(self.output.detach())
        real_d_pred = self.net_d(self.gt)

        # ✅ 판별기 학습 안정성 유지
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred.detach()), True, is_disc=True) * 0.5
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5

        l_d_total = l_d_real + l_d_fake
        l_d_total.backward()
        self.optimizer_d.step()

        # ✅ Loss dict 업데이트
        loss_dict['l_d_real'] = torch.clamp(l_d_real, min=1e-8).detach()
        loss_dict['l_d_fake'] = torch.clamp(l_d_fake, min=1e-8).detach()
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        # ✅ `loss_dict`을 float이 없도록 보장 후 변환
        for key, value in loss_dict.items():
            if isinstance(value, float):
                loss_dict[key] = torch.tensor(value, dtype=torch.float32, device=self.device).detach()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # -------------------------------
        # ✅ EMA 업데이트 (Exponential Moving Average)
        # -------------------------------
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)  # 🔄 EMA 적용


@MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    """ESRGAN 모델: 단일 이미지 초해상도(Single Image Super-Resolution)를 위한 모델."""

    def optimize_parameters(self, current_iter):
        # -------------------------------
        # 생성기 네트워크(net_g) 최적화
        # -------------------------------
        # 판별기 네트워크(net_d)의 모든 매개변수를 고정 (requires_grad=False)
        for p in self.net_d.parameters():
            p.requires_grad = False

        # 생성기 네트워크의 기울기 초기화
        self.optimizer_g.zero_grad()

        # 입력 저해상도 이미지를 사용해 생성기 출력 계산
        self.output = self.net_g(self.lq)

        # 생성기의 총 손실 초기화
        l_g_total = 0
        loss_dict = OrderedDict()

        # 생성기 학습 조건: 판별기의 초기화 단계가 끝났고, 지정된 반복 주기에 해당할 경우
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # 1. 픽셀 손실 계산
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)  # 생성된 이미지와 실제 고해상도 이미지 비교
                l_g_total += l_g_pix  # 총 손실에 추가
                loss_dict['l_g_pix'] = l_g_pix  # 손실 정보 저장

            # 2. Perceptual 손실 및 스타일 손실 계산
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)  # 생성된 이미지와 실제 이미지 비교
                if l_g_percep is not None:
                    l_g_total += l_g_percep  # Perceptual 손실 추가
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style  # 스타일 손실 추가
                    loss_dict['l_g_style'] = l_g_style

            # 3. GAN 손실 계산 (Relativistic GAN)
            real_d_pred = self.net_d(self.gt).detach()  # 실제 이미지에 대한 판별기의 출력
            fake_g_pred = self.net_d(self.output)  # 생성된 이미지에 대한 판별기의 출력

            # Relativistic GAN 손실 계산
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total += l_g_gan  # 총 손실에 GAN 손실 추가
            loss_dict['l_g_gan'] = l_g_gan

            # 역전파 및 생성기 최적화
            l_g_total.backward()
            self.optimizer_g.step()

        # -------------------------------
        # 판별기 네트워크(net_d) 최적화
        # -------------------------------
        # 판별기의 매개변수 업데이트를 허용 (requires_grad=True)
        for p in self.net_d.parameters():
            p.requires_grad = True

        # 판별기의 기울기 초기화
        self.optimizer_d.zero_grad()

        # Relativistic GAN 손실 계산
        # - 분산 학습 환경에서 발생할 수 있는 오류를 방지하기 위해
        #   실제(real)와 가짜(fake)의 역전파를 분리하여 실행

        # 1. 실제 이미지 손실 계산
        fake_d_pred = self.net_d(self.output).detach()  # 생성된 이미지에 대한 판별기 출력 (역전파 제외)
        real_d_pred = self.net_d(self.gt)  # 실제 이미지에 대한 판별기 출력
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()  # 역전파 실행

        # 2. 생성된 이미지 손실 계산
        fake_d_pred = self.net_d(self.output.detach())  # 생성된 이미지 출력 (역전파 제외)
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()  # 역전파 실행

        # 판별기 최적화
        self.optimizer_d.step()

        # 손실 값 저장
        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())  # 실제 이미지에 대한 평균 출력 저장
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())  # 생성된 이미지에 대한 평균 출력 저장

        # 손실 값을 로그에 기록
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # -------------------------------
        # EMA(Exponential Moving Average) 업데이트
        # -------------------------------
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
