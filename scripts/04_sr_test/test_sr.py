"""
Real-ESRGAN 사전 학습 모델로 OCT 이미지 1장 Super-Resolution 테스트.

입력: SBSDI D1 세트 1 노이즈 이미지 (450x900)
출력:
  results/04_sr_test/comparison_x2.png   원본 | x2 SR
  results/04_sr_test/comparison_x4.png   원본 | x4 SR
  results/04_sr_test/all_comparison.png  원본 | 클린 | x2 SR | x4 SR
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "01_baseline"))

WEIGHTS_DIR = ROOT / "weights"
DATA_DIR    = ROOT / "data" / "Final_Publication_2013_SBSDI" / "For synthetic experiments" / "1"
OUT_DIR     = ROOT / "results" / "04_sr_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gray_uint8(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def gray_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def bgr_to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def build_upsampler(scale: int, half: bool = True):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale
    )
    weight_name = f"RealESRGAN_x{scale}plus.pth"
    weight_path = WEIGHTS_DIR / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"가중치 없음: {weight_path}")

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weight_path),
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=half,
    )
    return upsampler


def apply_sr(upsampler, img_gray: np.ndarray, outscale: int) -> tuple[np.ndarray, float]:
    img_bgr = gray_to_bgr(img_gray)
    t0 = time.perf_counter()
    output_bgr, _ = upsampler.enhance(img_bgr, outscale=outscale)
    elapsed = time.perf_counter() - t0
    return bgr_to_gray(output_bgr), elapsed


def compute_metrics_at_original_scale(clean: np.ndarray, sr: np.ndarray) -> dict:
    """SR 결과를 원본 크기로 다운스케일 후 PSNR/SSIM 계산."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    sr_down = cv2.resize(sr, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA)
    clean_f = clean.astype(np.float64) / 255.0
    sr_f    = sr_down.astype(np.float64) / 255.0
    psnr = peak_signal_noise_ratio(clean_f, sr_f, data_range=1.0)
    ssim = structural_similarity(clean_f, sr_f, data_range=1.0)
    return {"psnr": psnr, "ssim": ssim}


def save_comparison(images: list[tuple[np.ndarray, str]], out_path: Path, dpi: int = 150):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path.relative_to(ROOT)}")


def main():
    noisy_path = DATA_DIR / "test.tif"
    clean_path = DATA_DIR / "average.tif"

    noisy = load_gray_uint8(noisy_path)
    clean = load_gray_uint8(clean_path)
    print(f"입력 이미지 크기: {noisy.shape}  (H x W)")

    import torch
    half = torch.cuda.is_available()
    device_str = "CUDA (half precision)" if half else "CPU (full precision)"
    print(f"추론 장치: {device_str}")

    results = {}

    for scale in (2, 4):
        print(f"\n[x{scale}] Real-ESRGAN 로드 중...")
        upsampler = build_upsampler(scale=scale, half=half)

        print(f"[x{scale}] 추론 중...")
        sr_img, elapsed = apply_sr(upsampler, noisy, outscale=scale)
        print(f"  출력 크기: {sr_img.shape}  |  처리 시간: {elapsed:.2f}s")

        m = compute_metrics_at_original_scale(clean, sr_img)
        print(f"  PSNR (원본 스케일): {m['psnr']:.4f} dB")
        print(f"  SSIM (원본 스케일): {m['ssim']:.4f}")

        results[scale] = {"sr": sr_img, "elapsed": elapsed, "metrics": m}

        save_comparison(
            [
                (noisy, f"Noisy Input\n{noisy.shape[1]}x{noisy.shape[0]}"),
                (sr_img, f"Real-ESRGAN x{scale}\n{sr_img.shape[1]}x{sr_img.shape[0]}\n"
                         f"PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f}  ({elapsed:.1f}s)"),
            ],
            OUT_DIR / f"comparison_x{scale}.png",
        )

    save_comparison(
        [
            (noisy, f"Noisy Input\n{noisy.shape[1]}x{noisy.shape[0]}"),
            (clean, f"Clean Reference\n(multi-frame avg)"),
            (results[2]["sr"], f"Real-ESRGAN x2\nPSNR={results[2]['metrics']['psnr']:.2f}"),
            (results[4]["sr"], f"Real-ESRGAN x4\nPSNR={results[4]['metrics']['psnr']:.2f}"),
        ],
        OUT_DIR / "all_comparison.png",
    )

    print("\n========== SR 테스트 결과 ==========")
    print(f"{'방법':<20} {'PSNR':>8} {'SSIM':>7} {'시간':>7}")
    print("-" * 45)
    for scale in (2, 4):
        m = results[scale]["metrics"]
        t = results[scale]["elapsed"]
        print(f"Real-ESRGAN x{scale}      {m['psnr']:>8.4f} {m['ssim']:>7.4f} {t:>6.2f}s")
    print("====================================")
    print(f"\n결과 이미지: results/04_sr_test/")


if __name__ == "__main__":
    main()
