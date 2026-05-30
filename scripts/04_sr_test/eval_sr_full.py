"""
Real-ESRGAN 사전 학습 모델로 SBSDI D1 전체 18쌍 평가.

입력: For synthetic experiments/{1..18}/test.tif  (noisy)
      For synthetic experiments/{1..18}/average.tif (clean reference)
출력:
  results/04_sr_test/eval_full_x2.csv    x2 쌍별 지표
  results/04_sr_test/eval_full_x4.csv    x4 쌍별 지표
  results/04_sr_test/eval_full_summary.txt  전체 요약
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "01_baseline"))

WEIGHTS_DIR = ROOT / "weights"
DATA_DIR    = ROOT / "data" / "Final_Publication_2013_SBSDI" / "For synthetic experiments"
OUT_DIR     = ROOT / "results" / "04_sr_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SETS = 18

SRAD_PSNR = 27.50
SRAD_SSIM = 0.652
SRAD_CNR  = 1.220


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
    weight_path = WEIGHTS_DIR / f"RealESRGAN_x{scale}plus.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"가중치 없음: {weight_path}")

    return RealESRGANer(
        scale=scale,
        model_path=str(weight_path),
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=half,
    )


def apply_sr(upsampler, img_gray: np.ndarray, outscale: int) -> tuple[np.ndarray, float]:
    img_bgr = gray_to_bgr(img_gray)
    t0 = time.perf_counter()
    output_bgr, _ = upsampler.enhance(img_bgr, outscale=outscale)
    elapsed = time.perf_counter() - t0
    return bgr_to_gray(output_bgr), elapsed


def compute_psnr_ssim(clean: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    c = clean.astype(np.float64) / 255.0
    r = restored.astype(np.float64) / 255.0
    psnr = peak_signal_noise_ratio(c, r, data_range=1.0)
    ssim = structural_similarity(c, r, data_range=1.0)
    return psnr, ssim


def compute_cnr(img: np.ndarray, ref: np.ndarray) -> float:
    threshold = np.median(ref)
    signal_mask = ref >= threshold
    bg_mask = ~signal_mask
    mu_s  = img[signal_mask].mean()
    mu_b  = img[bg_mask].mean()
    std_s = img[signal_mask].std()
    std_b = img[bg_mask].std()
    denom = np.sqrt(std_s ** 2 + std_b ** 2)
    if denom < 1e-8:
        return 0.0
    return float(abs(mu_s - mu_b) / denom)


def metrics_at_original_scale(clean: np.ndarray, sr: np.ndarray) -> dict:
    sr_down = cv2.resize(sr, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA)
    psnr, ssim = compute_psnr_ssim(clean, sr_down)
    cnr = compute_cnr(sr_down.astype(np.float64) / 255.0,
                      clean.astype(np.float64) / 255.0)
    return {"psnr": psnr, "ssim": ssim, "cnr": cnr}


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"{'세트':>4}  {'PSNR':>8}  {'SSIM':>7}  {'CNR':>7}  {'시간':>6}")
    print(f"{'-' * 46}")
    for r in rows:
        print(f"{r['set']:>4}  {r['psnr']:>8.4f}  {r['ssim']:>7.4f}  {r['cnr']:>7.4f}  {r['elapsed']:>5.2f}s")


def print_summary(rows: list[dict], scale: int) -> None:
    psnr_vals = [r["psnr"] for r in rows]
    ssim_vals = [r["ssim"] for r in rows]
    cnr_vals  = [r["cnr"]  for r in rows]
    t_vals    = [r["elapsed"] for r in rows]

    mean_psnr = np.mean(psnr_vals);  std_psnr = np.std(psnr_vals)
    mean_ssim = np.mean(ssim_vals);  std_ssim = np.std(ssim_vals)
    mean_cnr  = np.mean(cnr_vals);   std_cnr  = np.std(cnr_vals)
    mean_t    = np.mean(t_vals)

    print(f"\n--- Real-ESRGAN x{scale}  평균 (18쌍) ---")
    print(f"  PSNR  {mean_psnr:.4f} +- {std_psnr:.4f}  (SRAD: {SRAD_PSNR})  "
          f"{'[초과]' if mean_psnr > SRAD_PSNR else '[미달]'}")
    print(f"  SSIM  {mean_ssim:.4f} +- {std_ssim:.4f}  (SRAD: {SRAD_SSIM})  "
          f"{'[초과]' if mean_ssim > SRAD_SSIM else '[미달]'}")
    print(f"  CNR   {mean_cnr:.4f} +- {std_cnr:.4f}  (SRAD: {SRAD_CNR})  "
          f"{'[초과]' if mean_cnr > SRAD_CNR else '[미달]'}")
    print(f"  평균 처리 시간: {mean_t:.2f}s/장")

    return {
        "scale": scale,
        "psnr_mean": mean_psnr, "psnr_std": std_psnr,
        "ssim_mean": mean_ssim, "ssim_std": std_ssim,
        "cnr_mean":  mean_cnr,  "cnr_std":  std_cnr,
        "elapsed_mean": mean_t,
    }


def write_summary_file(summaries: list[dict]) -> None:
    lines = ["Real-ESRGAN SBSDI D1 전체 18쌍 평가 결과\n", "=" * 50 + "\n"]
    ref = [
        ("SRAD (기준)", SRAD_PSNR, SRAD_SSIM, SRAD_CNR),
    ]
    for r in ref:
        lines.append(f"{r[0]:<22} PSNR {r[1]:.4f}  SSIM {r[2]:.4f}  CNR {r[3]:.4f}\n")
    lines.append("-" * 50 + "\n")
    for s in summaries:
        tag = f"Real-ESRGAN x{s['scale']}"
        lines.append(
            f"{tag:<22} PSNR {s['psnr_mean']:.4f}+-{s['psnr_std']:.4f}"
            f"  SSIM {s['ssim_mean']:.4f}+-{s['ssim_std']:.4f}"
            f"  CNR {s['cnr_mean']:.4f}+-{s['cnr_std']:.4f}\n"
        )
    path = OUT_DIR / "eval_full_summary.txt"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"\n  요약 저장: {path.relative_to(ROOT)}")


def evaluate_scale(scale: int, half: bool) -> list[dict]:
    print(f"\n[x{scale}] 업샘플러 로드 중...")
    upsampler = build_upsampler(scale=scale, half=half)
    print(f"[x{scale}] 18쌍 평가 시작")

    rows = []
    for i in range(1, N_SETS + 1):
        set_dir = DATA_DIR / str(i)
        noisy = load_gray_uint8(set_dir / "test.tif")
        clean = load_gray_uint8(set_dir / "average.tif")

        sr, elapsed = apply_sr(upsampler, noisy, outscale=scale)
        m = metrics_at_original_scale(clean, sr)

        rows.append({
            "set":     i,
            "psnr":    round(m["psnr"], 6),
            "ssim":    round(m["ssim"], 6),
            "cnr":     round(m["cnr"],  6),
            "elapsed": round(elapsed, 3),
            "h_noisy": noisy.shape[0],
            "w_noisy": noisy.shape[1],
            "h_sr":    sr.shape[0],
            "w_sr":    sr.shape[1],
        })
        print(f"  set {i:2d}/18  PSNR={m['psnr']:.4f}  SSIM={m['ssim']:.4f}"
              f"  CNR={m['cnr']:.4f}  {elapsed:.2f}s", flush=True)

    save_csv(OUT_DIR / f"eval_full_x{scale}.csv", rows)
    print(f"  CSV 저장: results/04_sr_test/eval_full_x{scale}.csv")
    return rows


def main():
    import torch
    half = torch.cuda.is_available()
    device_str = "CUDA (half precision)" if half else "CPU (full precision)"
    print(f"추론 장치: {device_str}")
    print(f"평가 데이터: SBSDI D1  {N_SETS}쌍  (For synthetic experiments/)")

    summaries = []
    for scale in (2, 4):
        rows = evaluate_scale(scale, half)
        print_table(rows, f"Real-ESRGAN x{scale}  쌍별 결과")
        s = print_summary(rows, scale)
        summaries.append(s)

    write_summary_file(summaries)
    print("\n완료.")


if __name__ == "__main__":
    main()
