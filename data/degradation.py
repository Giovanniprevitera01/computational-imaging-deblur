"""
degradation.py
--------------
Core degradation pipeline shared by ALL methods.
Every method reads the same degraded images saved by generate_degraded.py.

Degradation model:  y = A(x) + n
    A  = Gaussian blur (sigma=2, kernel 9x9)
    n  ~ N(0, sigma_noise^2 * I)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, convolve

# ── Fixed degradation parameters (from project spec) ──────────────────────────
SIGMA_BLUR   = 2
KERNEL_SIZE  = 9
NOISE_LEVELS = [0.005, 0.01, 0.05, 0.1]

# Fixed seed for reproducibility — ALL methods use the same degraded inputs
DEGRADATION_SEED = 42




def blur(img: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Apply Gaussian blur to a float32 RGB image.

    Parameters
    ----------
    img    : (H, W, 3) float32 in [0, 1]
    kernel : optional pre-built kernel; if None, uses default params

    Returns
    -------
    blurred : (H, W, 3) float32
    """
    if kernel is None:
        kernel = get_gaussian_kernel()
    return np.stack(
        [convolve(img[:, :, c], kernel, mode="reflect") for c in range(3)],
        axis=2,
    ).astype(np.float32)


def add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add i.i.d. Gaussian noise.

    Parameters
    ----------
    img   : (H, W, 3) float32 in [0, 1]
    sigma : noise standard deviation
    rng   : numpy random Generator (for reproducibility)

    Returns
    -------
    noisy : (H, W, 3) float32, clipped to [0, 1]
    """
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def degrade(img: np.ndarray, sigma_noise: float, seed: int = DEGRADATION_SEED) -> np.ndarray:
    """
    Full degradation: blur → add noise.

    Parameters
    ----------
    img         : (H, W, 3) float32 in [0, 1]
    sigma_noise : noise level
    seed        : random seed (fixed per image+sigma combination)

    Returns
    -------
    degraded : (H, W, 3) float32 in [0, 1]
    """
    rng = np.random.default_rng(seed)
    blurred = blur(img)
    return add_noise(blurred, sigma_noise, rng)
    
def get_gaussian_kernel(size: int = KERNEL_SIZE, sigma: float = SIGMA_BLUR):
    k = np.zeros((size, size))
    k[size // 2, size // 2] = 1.0
    k = gaussian_filter(k, sigma=sigma)
    return k / k.sum()


def degrade_all_levels(img: np.ndarray) -> dict[float, np.ndarray]:
    """
    Degrade the same image at all four noise levels.

    Returns
    -------
    dict mapping sigma -> degraded array
    """
    return {sigma: degrade(img, sigma) for sigma in NOISE_LEVELS}
