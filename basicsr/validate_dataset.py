from basicsr.data import build_dataset, build_dataloader
import matplotlib.pyplot as plt

def main():
    dataset_opt = {
        'name': 'WindSpeedDataset',
        'type': 'NumpyPairedDataset',
        'dataroot_gt': '/home/wj/works/Wind_Speed_Data/new_data/HR/val_hr.npy',
        'dataroot_lq': '/home/wj/works/Wind_Speed_Data/new_data/LR/val_lr.npy',
        'scale': 5,
        'gt_size': 100,
        'use_hflip': False,
        'use_rot': False,
        'phase': 'val',
        'num_worker_per_gpu': 4,
        'batch_size_per_gpu': 4,
        'pin_memory': True,
        'persistent_workers': True
    }

    dataset = build_dataset(dataset_opt)
    dataloader = build_dataloader(
        dataset=dataset,
        dataset_opt=dataset_opt,
        num_gpu=1,
        dist=False,
        sampler=None,
        seed=42
    )

    print(f"Dataset size: {len(dataset)}")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"LQ batch shape: {batch['lq'].shape}, GT batch shape: {batch['gt'].shape}")

        for sample_idx in range(batch['lq'].shape[0]):
            lq_sample = batch['lq'][sample_idx]
            gt_sample = batch['gt'][sample_idx]
            print(f"  Sample {sample_idx + 1}:")
            print(f"    LQ sample shape: {lq_sample.shape}")
            print(f"    GT sample shape: {gt_sample.shape}")

            if batch_idx == 0 and sample_idx == 0:
                visualize_sample(lq_sample, gt_sample)
        break

def visualize_sample(lq_tensor, gt_tensor):
    """
    배치 내 첫 번째 샘플 데이터를 시각화합니다.

    Args:
        lq_tensor (torch.Tensor): 저해상도(LQ) 데이터 텐서.
        gt_tensor (torch.Tensor): 고해상도(GT) 데이터 텐서.
    """
    try:
        # 디버깅: 텐서 정보 출력
        print(f"LQ tensor type: {type(lq_tensor)}, is_sparse: {lq_tensor.is_sparse if hasattr(lq_tensor, 'is_sparse') else 'N/A'}")
        print(f"GT tensor type: {type(gt_tensor)}, is_sparse: {gt_tensor.is_sparse if hasattr(gt_tensor, 'is_sparse') else 'N/A'}")
        print(f"LQ tensor shape for visualization (before): {lq_tensor.shape}")
        print(f"GT tensor shape for visualization (before): {gt_tensor.shape}")

        # 텐서가 2차원인 경우, 채널 차원을 추가
        if lq_tensor.dim() == 2:  # [H, W] -> [1, H, W]
            lq_tensor = lq_tensor.unsqueeze(0)
        if gt_tensor.dim() == 2:  # [H, W] -> [1, H, W]
            gt_tensor = gt_tensor.unsqueeze(0)

        print(f"LQ tensor shape for visualization (after unsqueeze): {lq_tensor.shape}")
        print(f"GT tensor shape for visualization (after unsqueeze): {gt_tensor.shape}")

        # permute 호출 전 텐서 차원 및 형식 확인
        if not lq_tensor.is_sparse and lq_tensor.dim() == 3:  # (C, H, W) 형식
            lq_image = lq_tensor.permute(1, 2, 0).squeeze().numpy()  # (C, H, W) -> (H, W)
        elif not lq_tensor.is_sparse and lq_tensor.dim() == 2:  # (H, W) 형식
            lq_image = lq_tensor.squeeze().numpy()  # (H, W)
        else:
            raise ValueError(f"LQ tensor format not supported: {lq_tensor.shape}, sparse={lq_tensor.is_sparse}")

        if not gt_tensor.is_sparse and gt_tensor.dim() == 3:  # (C, H, W) 형식
            gt_image = gt_tensor.permute(1, 2, 0).squeeze().numpy()  # (C, H, W) -> (H, W)
        elif not gt_tensor.is_sparse and gt_tensor.dim() == 2:  # (H, W) 형식
            gt_image = gt_tensor.squeeze().numpy()  # (H, W)
        else:
            raise ValueError(f"GT tensor format not supported: {gt_tensor.shape}, sparse={gt_tensor.is_sparse}")

        print(f"LQ image shape for plotting: {lq_image.shape}")
        print(f"GT image shape for plotting: {gt_image.shape}")

        # 이미지 시각화 (viridis 컬러맵 사용)
        plt.subplot(1, 2, 1)
        plt.title("LQ Image")
        plt.imshow(lq_image, cmap='viridis')  # viridis 컬러맵 사용
        plt.colorbar(label='Wind Speed')

        plt.subplot(1, 2, 2)
        plt.title("GT Image")
        plt.imshow(gt_image, cmap='viridis')  # viridis 컬러맵 사용
        plt.colorbar(label='Wind Speed')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")



if __name__ == "__main__":
    main()
