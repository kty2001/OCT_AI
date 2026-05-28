"""
전통적 스페클 노이즈 제거 방법 구현: NLM, BM3D, SRAD.
입력/출력 모두 float32 [0, 1] 그레이스케일 2D 배열.
"""

import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import bm3d
from scipy.ndimage import uniform_filter


# ---------------------------------------------------------------------------
# NLM (Non-Local Means)
# ---------------------------------------------------------------------------

def denoise_nlm(img: np.ndarray, fast_mode: bool = True) -> np.ndarray:
    sigma = estimate_sigma(img)
    h = 0.8 * sigma if fast_mode else 1.2 * sigma
    return denoise_nl_means(
        img,
        h=h,
        fast_mode=fast_mode,
        patch_size=5,
        patch_distance=6,
        channel_axis=None,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# BM3D
# ---------------------------------------------------------------------------

def denoise_bm3d(img: np.ndarray, sigma_psd: float | None = None) -> np.ndarray:
    if sigma_psd is None:
        sigma_psd = float(estimate_sigma(img))
    result = bm3d.bm3d(img, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# SRAD (Speckle Reducing Anisotropic Diffusion)
# Yu & Acton, IEEE TIP 2002
# ---------------------------------------------------------------------------

def _srad_q0_squared(img: np.ndarray, rho: int = 5) -> float:
    """전체 이미지 초기 q0^2 추정 (균일 영역 기반)."""
    mean = uniform_filter(img, size=rho)
    mean_sq = uniform_filter(img ** 2, size=rho)
    var = mean_sq - mean ** 2
    valid = mean > 1e-8
    q2_vals = var[valid] / (mean[valid] ** 2)
    return float(np.median(q2_vals))


def denoise_srad(
    img: np.ndarray,
    n_iter: int = 100,
    dt: float = 0.1,
    rho: int = 5,
) -> np.ndarray:
    """
    SRAD 반복 확산으로 스페클 제거.
    n_iter: 반복 횟수 (클수록 더 강하게 스무딩)
    dt: 시간 스텝 (0 < dt <= 0.25 권장)
    rho: q0 추정용 윈도우 크기
    """
    u = img.copy().astype(np.float64)
    q0_sq = _srad_q0_squared(u, rho)

    for t in range(1, n_iter + 1):
        # 국소 통계 계산
        mean_local = uniform_filter(u, size=rho)
        mean_sq_local = uniform_filter(u ** 2, size=rho)
        var_local = np.maximum(mean_sq_local - mean_local ** 2, 0.0)

        # 순간 변동 계수 q (Instantaneous Coefficient of Variation)
        denom = mean_local ** 2 + 1e-10
        q_sq = (var_local / denom) / (1.0 + q0_sq / t)
        # q0 감쇠: q0^2(t) = q0^2(0) / (1 + t)

        # 확산 계수 c
        c = 1.0 / (1.0 + (q_sq - q0_sq / t) / (q0_sq / t * (1.0 + q0_sq / t)))
        c = np.clip(c, 0.0, 1.0)

        # 이산 Laplacian (4방향 divergence 근사)
        cn = np.roll(c, -1, axis=0)  # north
        cs = c                        # south (현 픽셀)
        ce = np.roll(c, -1, axis=1)  # east
        cw = c                        # west

        un = np.roll(u, -1, axis=0)
        us = np.roll(u,  1, axis=0)
        ue = np.roll(u, -1, axis=1)
        uw = np.roll(u,  1, axis=1)

        div = (cn * (un - u) + cs * (us - u) +
               ce * (ue - u) + cw * (uw - u))

        u = u + dt * div

    return np.clip(u, 0.0, 1.0).astype(np.float32)


METHODS: dict[str, callable] = {
    "nlm": denoise_nlm,
    "bm3d": denoise_bm3d,
    "srad": denoise_srad,
}
