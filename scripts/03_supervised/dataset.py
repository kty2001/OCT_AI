"""
지도학습용 데이터셋: results/02_synthetic_noise/metadata.csv 기반.
6,136 noisy-clean 쌍 (AROI 1,136 + Kermany 5,000) 에서 무작위 패치 샘플링.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).parent.parent.parent
META_CSV = ROOT / "results" / "02_synthetic_noise" / "metadata.csv"


def _load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


class SyntheticPairDataset(Dataset):
    """
    metadata.csv의 (noisy, clean) 쌍을 로드해 랜덤 패치를 반환.
    이미지는 첫 접근 시 메모리에 캐싱 (6,136쌍 × 2장 ≈ 최대 ~3GB).
    메모리가 부족하면 cache=False로 설정.
    """

    def __init__(self, patch_size: int = 128, patches_per_image: int = 4,
                 cache: bool = True):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.cache = cache

        df = pd.read_csv(META_CSV)
        self.noisy_paths = [ROOT / p for p in df["noisy"]]
        self.clean_paths = [ROOT / p for p in df["clean"]]

        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        print(f"Dataset: {len(self.noisy_paths)} pairs "
              f"(AROI {(df['dataset']=='AROI').sum()} + "
              f"Kermany {(df['dataset']=='Kermany').sum()})")

    def __len__(self) -> int:
        return len(self.noisy_paths) * self.patches_per_image

    def _get_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.cache:
            if idx not in self._cache:
                noisy = _load_gray(self.noisy_paths[idx])
                clean = _load_gray(self.clean_paths[idx])
                self._cache[idx] = (noisy, clean)
            return self._cache[idx]
        return _load_gray(self.noisy_paths[idx]), _load_gray(self.clean_paths[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_idx = idx // self.patches_per_image
        noisy, clean = self._get_pair(pair_idx)

        H, W = noisy.shape
        ps = self.patch_size
        top  = np.random.randint(0, H - ps)
        left = np.random.randint(0, W - ps)

        p_noisy = noisy[top:top+ps, left:left+ps].copy()
        p_clean = clean[top:top+ps, left:left+ps].copy()

        if np.random.rand() > 0.5:
            p_noisy = np.fliplr(p_noisy).copy()
            p_clean = np.fliplr(p_clean).copy()
        if np.random.rand() > 0.5:
            p_noisy = np.flipud(p_noisy).copy()
            p_clean = np.flipud(p_clean).copy()

        return torch.from_numpy(p_noisy[None]), torch.from_numpy(p_clean[None])
