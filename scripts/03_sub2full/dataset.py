"""
N2N 프레임쌍 dataset: 같은 세트의 다른 프레임을 noisy-noisy 쌍으로 사용.

SBSDI real experiments 각 세트에는 동일 위치를 반복 스캔한 1~4.tif가 있다.
같은 세트의 두 프레임은 조직 구조는 동일하고 스페클 패턴은 독립적이므로
Noise2Noise 학습 조건을 자연스럽게 만족한다.

39세트 x 12 순서쌍(4P2 = 12) = 468쌍 (양방향 모두 사용)
"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).parent.parent.parent
REAL_DIR = ROOT / "data" / "Final_Publication_2013_SBSDI" / "For real experiments on Humans"


def _load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


class Sub2FullDataset(Dataset):
    """
    학습용 패치 데이터셋.
    같은 세트의 (frame_i, frame_j) i != j 순서쌍을 noisy-noisy 쌍으로 사용.
    에포크마다 무작위 패치를 샘플링.
    """

    def __init__(self, patch_size: int = 128, patches_per_image: int = 8):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []

        set_dirs = sorted(
            [d for d in REAL_DIR.iterdir() if d.is_dir()],
            key=lambda p: int(p.name),
        )
        for set_dir in set_dirs:
            frames = []
            for fname in ("1.tif", "2.tif", "3.tif", "4.tif"):
                fpath = set_dir / fname
                if fpath.exists():
                    frames.append(_load_gray(fpath))
            # 같은 세트 내 모든 순서쌍 (i != j)
            for i in range(len(frames)):
                for j in range(len(frames)):
                    if i != j:
                        self.pairs.append((frames[i], frames[j]))

        n_sets = len(set_dirs)
        print(f"Dataset: {n_sets} sets x 12 ordered pairs = {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp_img, tgt_img = self.pairs[idx // self.patches_per_image]
        H, W = inp_img.shape
        ps = self.patch_size

        top = np.random.randint(0, H - ps)
        left = np.random.randint(0, W - ps)

        p_inp = inp_img[top:top + ps, left:left + ps].copy()
        p_tgt = tgt_img[top:top + ps, left:left + ps].copy()

        if np.random.rand() > 0.5:
            p_inp = np.fliplr(p_inp).copy()
            p_tgt = np.fliplr(p_tgt).copy()
        if np.random.rand() > 0.5:
            p_inp = np.flipud(p_inp).copy()
            p_tgt = np.flipud(p_tgt).copy()

        return torch.from_numpy(p_inp[None]), torch.from_numpy(p_tgt[None])
