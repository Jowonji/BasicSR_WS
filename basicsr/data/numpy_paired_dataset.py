import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NumpyPairedDataset(Dataset):
    """Numpy 형식의 저해상도(LR) 및 고해상도(HR) 데이터를 로드하는 데이터셋 클래스"""

    def __init__(self, opt):
        super(NumpyPairedDataset, self).__init__()
        self.opt = opt
        
        # Numpy 파일에서 HR 및 LR 데이터 로드
        self.hr_data = np.load(opt['dataroot_gt'])  # HR 데이터 로드
        self.lr_data = np.load(opt['dataroot_lq'])  # LR 데이터 로드

        # HR과 LR 데이터의 샘플 수가 일치하는지 확인
        assert len(self.hr_data) == len(self.lr_data), "HR and LR datasets must have the same number of samples."
        self.scale = opt['scale'] # 업스케일링 비율 설정

        # HR 데이터 각각에 대한 Min-Max 정규화 값 저장
        self.hr_min = self.hr_data.min()
        self.hr_max = self.hr_data.max()

        print(f"HR data min: {self.hr_min}, max: {self.hr_max}")

    def __getitem__(self, index):
        # 인덱스에 해당하는 HR 및 LR 데이터 로드
        hr = self.hr_data[index]
        lr = self.lr_data[index]

        # 데이터 타입을 float32로 변환
        hr = hr.astype(np.float32)
        lr = lr.astype(np.float32)

        # Min-Max 정규화 적용
        hr = (hr - self.hr_min) / (self.hr_max - self.hr_min)
        lr = (lr - self.hr_min) / (self.hr_max - self.hr_min)

        # 채널 차원 추가 (H, W) -> (C, H, W)
        if hr.ndim == 2:  # (H, W) -> (C, H, W)
            hr = np.expand_dims(hr, axis=0)
        if lr.ndim == 2:  # (H, W) -> (C, H, W)
            lr = np.expand_dims(lr, axis=0)

        # 스케일 검증 (HR이 LR보다 정확히 scale배 크기여야 함)
        h_gt, w_gt = hr.shape[1], hr.shape[2]
        h_lq, w_lq = lr.shape[1], lr.shape[2]
        if h_gt != h_lq * self.scale or w_gt != w_lq * self.scale:
            raise ValueError(
                f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {self.scale}x multiplication of LQ ({h_lq}, {w_lq})."
            )

        # Numpy 배열을 Pytorch Tensor로 변환
        hr_tensor = torch.from_numpy(hr)
        lr_tensor = torch.from_numpy(lr)

        return {'gt': hr_tensor, 'lq': lr_tensor, 'gt_path': str(index), 'lq_path': str(index)}

    def __len__(self):
        return len(self.hr_data)
