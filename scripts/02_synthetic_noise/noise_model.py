"""
Gamma-distribution based multiplicative speckle noise synthesis for OCT images.

Model: I = S * N_s + N_a
  N_s ~ Gamma(L, 1/L)  (mean=1, var=1/L)
  N_a ~ N(0, sigma_a^2)
"""
from __future__ import annotations

import numpy as np


def add_speckle(
    image: np.ndarray,
    L: float = 4.0,
    sigma_a: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Gamma-distribution based multiplicative speckle + additive Gaussian noise.

    Args:
        image:   float32 array in [0, 1] - clean OCT B-scan
        L:       number of looks (higher = less noise); typical OCT range: 1-8
        sigma_a: additive noise std
        rng:     numpy random Generator for reproducibility

    Returns:
        Synthetic noisy image as float32 in [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()
    img = np.asarray(image, dtype=np.float32)
    Ns = rng.gamma(shape=L, scale=1.0 / L, size=img.shape).astype(np.float32)
    Na = (rng.standard_normal(img.shape) * sigma_a).astype(np.float32)
    return np.clip(img * Ns + Na, 0.0, 1.0)


def estimate_noise_params(
    noisy: np.ndarray,
    clean: np.ndarray,
    fg_threshold: float = 0.05,
) -> dict[str, float]:
    """
    Estimate speckle noise parameters from a noisy-clean pair (SBSDI D1 style).

    For multiplicative model N_s = I / S, Gamma moment matching:
      L = mean(N_s)^2 / var(N_s)

    Additive noise sigma_a estimated from background (low-intensity) region.

    Args:
        noisy, clean: float32 arrays in [0, 1]
        fg_threshold: minimum clean pixel value for foreground mask

    Returns:
        {'L': float, 'sigma_a': float}
    """
    fg = clean > fg_threshold
    if fg.sum() < 100:
        return {"L": 4.0, "sigma_a": 0.01}

    Ns = noisy[fg] / (clean[fg] + 1e-8)
    mean_Ns = float(Ns.mean())
    var_Ns = float(Ns.var())
    L = (mean_Ns ** 2 / var_Ns) if var_Ns > 1e-9 else 4.0

    bg = clean <= fg_threshold
    sigma_a = float(noisy[bg].std()) if bg.sum() > 100 else 0.01

    return {"L": round(L, 3), "sigma_a": round(sigma_a, 5)}
