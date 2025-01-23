from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """
    이미지 복원(Image Restoration)을 위한 쌍(pair) 이미지 데이터셋.

    LQ (Low Quality, 저품질) 이미지와 GT (Ground Truth, 고품질) 이미지 쌍을 읽어들입니다.
    이 데이터셋은 세 가지 모드를 지원합니다:
    
    1. **lmdb**:
        - LMDB 파일 형식의 데이터셋을 사용하는 경우.
        - `opt['io_backend']`가 'lmdb'로 설정되어 있어야 합니다.
    2. **meta_info_file**:
        - 메타 정보 파일을 사용하여 경로를 생성하는 경우.
        - `opt['io_backend']`가 'lmdb'가 아니고, `opt['meta_info_file']`이 설정되어 있어야 합니다.
    3. **folder**:
        - 폴더를 스캔하여 경로를 생성하는 경우. 위 두 모드에 해당하지 않는 경우 기본으로 사용됩니다.

    Args:
        opt (dict): 학습 데이터셋 설정 딕셔너리. 주요 키:
            - dataroot_gt (str): GT 데이터의 루트 경로.
            - dataroot_lq (str): LQ 데이터의 루트 경로.
            - meta_info_file (str): 메타 정보 파일 경로.
            - io_backend (dict): IO 백엔드 타입 및 기타 설정.
            - filename_tmpl (str): 각 파일명에 대한 템플릿 (파일 확장자 제외). 기본값: '{}'.
            - gt_size (int): GT 패치 크기 (크롭 크기).
            - use_hflip (bool): 수평 뒤집기 사용 여부.
            - use_rot (bool): 회전 사용 여부 (수직 뒤집기 및 h/w 전환 포함).
            - scale (bool): 스케일, 자동으로 추가됨.
            - phase (str): 'train' 또는 'val'.

    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt # 데이터셋 설정 저장
        # IO 백엔드(client) 설정
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None # 데이터 정규화를 위한 평균값
        self.std = opt['std'] if 'std' in opt else None # 데이터 정규화를 위한 표준편차

        # GT 및 LQ 데이터 경로 설정
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}' # 기본 템플릿

        # IO 백엔드 타입에 따른 경로 초기화
        if self.io_backend_opt['type'] == 'lmdb': # LMDB 모드
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt']
            )
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], 
                ['lq', 'gt'],
                self.opt['meta_info_file'], 
                self.filename_tmpl
            )
        else: # 폴더 스캔 모드
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], 
                ['lq', 'gt'], 
                self.filename_tmpl
            )

    def __getitem__(self, index):
        """
        주어진 인덱스에 해당하는 데이터 샘플(LQ와 GT 이미지 쌍)을 반환하는 함수.

        Args:
            index (int): 데이터셋 내 샘플의 인덱스.

        Returns:
            dict: {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
                - 'lq': 전처리된 LQ 이미지 (torch.Tensor).
                - 'gt': 전처리된 GT 이미지 (torch.Tensor).
                - 'lq_path': LQ 이미지 경로.
                - 'gt_path': GT 이미지 경로.
        """
        # FileClient 초기화
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale'] # 이미지 스케일 설정

        # LQ 및 GT 이미지 로드 (HWC, BGR, [0, 1], float32 형식)
        gt_path = self.paths[index]['gt_path'] # GT 이미지 경로
        img_bytes = self.file_client.get(gt_path, 'gt') # 파일에서 이미지 데이터 읽기
        img_gt = imfrombytes(img_bytes, float32=True) # 바이트 데이터를 이미지로 변환
        lq_path = self.paths[index]['lq_path'] # LQ 이미지 경로
        img_bytes = self.file_client.get(lq_path, 'lq') # 파일에서 이미지 데이터 읽기
        img_lq = imfrombytes(img_bytes, float32=True) # 바이트 데이터를 이미지로 변환

        # 학습 데이터 증강 (train 단계에서만 적용)
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size'] # GT 이미지 크기 설정
            # 랜덤 크롭
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # 플립 및 회전
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # 색 공간 변환 (BGR -> Y 채널, 선택적)
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None] # Y 채널만 추출
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None] # Y 채널만 추출

        # 검증 또는 테스트 시 GT 이미지를 LQ 이미지와 맞추기 위해 크롭 (특히 SR 벤치마크 데이터셋)
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR -> RGB, HWC -> CHW, numpy -> tensor 변환
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # 정규화
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True) # LQ 이미지 정규화
            normalize(img_gt, self.mean, self.std, inplace=True) # GT 이미지 정규화

        return {'lq': img_lq, # 전처리된 LQ 이미지 
                'gt': img_gt, # 전처리된 GT 이미지
                'lq_path': lq_path, # LQ 이미지 경로
                'gt_path': gt_path # GT 이미지 경로
        }

    def __len__(self):
        """
        데이터셋의 총 샘플 수를 반환.
    
        Returns:
            int: 데이터셋 내 샘플의 총 개수.
        """
        return len(self.paths)
