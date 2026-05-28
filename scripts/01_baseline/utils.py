"""
SBSDI D1 데이터 로드 및 평가 지표 계산 유틸리티.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "Final_Publication_2013_SBSDI"
SYNTHETIC_DIR = DATA_ROOT / "For synthetic experiments"


def load_pair(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """noisy(test.tif)와 clean(average.tif) 쌍을 float32 [0,1]로 반환."""
    folder = SYNTHETIC_DIR / str(idx)
    noisy = np.array(Image.open(folder / "test.tif").convert("L"), dtype=np.float32) / 255.0
    clean = np.array(Image.open(folder / "average.tif").convert("L"), dtype=np.float32) / 255.0
    return noisy, clean


def load_all_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    """18쌍 전체 로드."""
    return [load_pair(i) for i in range(1, 19)]


def compute_psnr(ref: np.ndarray, img: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, img, data_range=1.0))


def compute_ssim(ref: np.ndarray, img: np.ndarray) -> float:
    return float(structural_similarity(ref, img, data_range=1.0))


def compute_cnr(img: np.ndarray, ref: np.ndarray) -> float:
    """
    CNR = |mean_signal - mean_background| / sqrt(std_signal^2 + std_background^2)

    signal 영역: ref 기준 상위 50% 픽셀
    background 영역: ref 기준 하위 50% 픽셀
    """
    threshold = np.median(ref)
    signal_mask = ref >= threshold
    bg_mask = ~signal_mask

    mu_s = img[signal_mask].mean()
    mu_b = img[bg_mask].mean()
    std_s = img[signal_mask].std()
    std_b = img[bg_mask].std()

    denom = np.sqrt(std_s ** 2 + std_b ** 2)
    if denom < 1e-8:
        return 0.0
    return float(abs(mu_s - mu_b) / denom)


def compute_metrics(clean: np.ndarray, denoised: np.ndarray) -> dict:
    return {
        "psnr": compute_psnr(clean, denoised),
        "ssim": compute_ssim(clean, denoised),
        "cnr": compute_cnr(denoised, clean),
    }
