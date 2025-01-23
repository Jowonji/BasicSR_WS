import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NumpyPairedDataset(Dataset):
    """Numpy paired dataset for super-resolution.

    This dataset is designed to load paired numpy arrays for low-resolution (LR)
    and high-resolution (HR) data stored in `.npy` format.

    Args:
        opt (dict): Configuration options for the dataset.
    """

    def __init__(self, opt):
        super(NumpyPairedDataset, self).__init__()
        self.opt = opt
        # Load HR and LR data from numpy files
        self.hr_data = np.load(opt['dataroot_gt'])  # Load HR data
        self.lr_data = np.load(opt['dataroot_lq'])  # Load LR data

        # Ensure the number of samples matches
        assert len(self.hr_data) == len(self.lr_data), "HR and LR datasets must have the same number of samples."
        self.scale = opt['scale']

        # Min-Max normalization for LR and HR separately
        self.hr_min = self.hr_data.min()
        self.hr_max = self.hr_data.max()
        self.lr_min = self.lr_data.min()
        self.lr_max = self.lr_data.max()

        print(f"HR data min: {self.hr_min}, max: {self.hr_max}")
        print(f"LR data min: {self.lr_min}, max: {self.lr_max}")

    def __getitem__(self, index):
        # Load HR and LR data
        hr = self.hr_data[index]
        lr = self.lr_data[index]

        # Ensure data is float32
        hr = hr.astype(np.float32)
        lr = lr.astype(np.float32)

        # Min-Max normalization
        hr = (hr - self.hr_min) / (self.hr_max - self.hr_min)
        lr = (lr - self.lr_min) / (self.lr_max - self.lr_min)

        # Add channel dimension if necessary
        if hr.ndim == 2:  # (H, W) -> (C, H, W)
            hr = np.expand_dims(hr, axis=0)
        if lr.ndim == 2:  # (H, W) -> (C, H, W)
            lr = np.expand_dims(lr, axis=0)

        # Verify scale condition
        h_gt, w_gt = hr.shape[1], hr.shape[2]
        h_lq, w_lq = lr.shape[1], lr.shape[2]
        if h_gt != h_lq * self.scale or w_gt != w_lq * self.scale:
            raise ValueError(
                f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {self.scale}x multiplication of LQ ({h_lq}, {w_lq})."
            )

        # Convert numpy arrays to PyTorch tensors
        hr_tensor = torch.from_numpy(hr)
        lr_tensor = torch.from_numpy(lr)

        return {'gt': hr_tensor, 'lq': lr_tensor, 'gt_path': str(index), 'lq_path': str(index)}

    def __len__(self):
        return len(self.hr_data)
