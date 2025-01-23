import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel

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
