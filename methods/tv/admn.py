"""
admm.py
-------
TV-regularized image deblurring via ADMM (Alternating Direction Method of Multipliers).

Solves:
    min_x  (1/2) ||A x - y||^2  +  lambda * TV(x)

where A is the Gaussian blur operator and TV is the isotropic Total Variation.

ADMM split:
    Introduce  z_h ≈ D_h x,  z_v ≈ D_v x
    Augmented Lagrangian:
        L_rho(x, z, u) = (1/2)||Ax-y||^2 + lambda||z||_{2,1}
                         + (rho/2)||Dx - z + u||^2

    x-update : (A^T A + rho D^T D) x = A^T y + rho D^T (z - u)
               solved iteratively with gradient descent steps
    z-update : soft-thresholding (proximal operator of lambda/rho * ||.||_{2,1})
    u-update : dual ascent

References
----------
    Boyd et al. (2011). Distributed Optimization via ADMM.
    Foundations and Trends in Machine Learning, 3(1), 1–122.
"""

import numpy as np
from tqdm import trange
from .operators import A, At, AtA, D_h, D_v, Dt_h, Dt_v, DtD, tv_norm


# ── Proximal operator ──────────────────────────────────────────────────────────

def _prox_l21(v_h: np.ndarray, v_v: np.ndarray, threshold: float):
    """
    Proximal operator of threshold * ||.||_{2,1} (isotropic TV shrinkage).

    Parameters
    ----------
    v_h, v_v  : horizontal and vertical gradient maps  (H, W, C)
    threshold : lambda / rho

    Returns
    -------
    z_h, z_v : shrunk gradient maps
    """
    norm = np.sqrt(v_h ** 2 + v_v ** 2 + 1e-8)   # (H, W, C)
    scale = np.maximum(0.0, 1.0 - threshold / norm)
    return v_h * scale, v_v * scale


# ── x-update (linear system solve via gradient steps) ─────────────────────────

def _x_update(x: np.ndarray, y: np.ndarray,
               z_h, z_v, u_h, u_v,
               rho: float, n_inner: int = 5, lr: float = 0.1) -> np.ndarray:
    """
    Minimise over x:
        (1/2)||Ax - y||^2  +  (rho/2)||Dx - z + u||^2

    Using n_inner gradient descent steps (closed-form requires FFT; this is
    simpler and converges in practice for n_inner ≈ 3–8).
    """
    rhs = At(y) + rho * (Dt_h(z_h - u_h) + Dt_v(z_v - u_v))
    for _ in range(n_inner):
        grad = AtA(x) - rhs + rho * DtD(x)
        x = x - lr * grad
        x = np.clip(x, 0.0, 1.0)
    return x


# ── Main ADMM solver ───────────────────────────────────────────────────────────

def tv_admm(
    y: np.ndarray,
    lam: float = 0.01,
    rho: float = 1.0,
    n_iter: int = 100,
    n_inner: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """
    TV deblurring via ADMM.

    Parameters
    ----------
    y       : degraded image (H, W, 3) float32 in [0, 1]
    lam     : TV regularization weight (tune on validation set)
    rho     : ADMM penalty parameter (default 1.0, robust to changes)
    n_iter  : number of outer ADMM iterations
    n_inner : gradient descent steps per x-update
    verbose : print TV norm and residuals every 10 iterations

    Returns
    -------
    x : restored image (H, W, 3) float32 in [0, 1]
    """
    # Initialise primal and dual variables
    x   = y.copy()
    z_h = np.zeros_like(x)
    z_v = np.zeros_like(x)
    u_h = np.zeros_like(x)
    u_v = np.zeros_like(x)

    threshold = lam / rho

    iters = trange(n_iter, desc="TV-ADMM", leave=False) if verbose else range(n_iter)

    for it in iters:
        # 1. x-update
        x = _x_update(x, y, z_h, z_v, u_h, u_v, rho, n_inner)

        # 2. z-update (proximal / shrinkage)
        z_h, z_v = _prox_l21(D_h(x) + u_h, D_v(x) + u_v, threshold)

        # 3. u-update (dual ascent)
        u_h = u_h + D_h(x) - z_h
        u_v = u_v + D_v(x) - z_v

        if verbose and it % 10 == 0:
            primal_res = np.linalg.norm(D_h(x) - z_h) + np.linalg.norm(D_v(x) - z_v)
            iters.set_postfix(tv=f"{tv_norm(x):.2f}", res=f"{primal_res:.4f}")

    return np.clip(x, 0.0, 1.0)


# ── Lambda tuning ──────────────────────────────────────────────────────────────

def tune_lambda(
    y_list: list[np.ndarray],
    gt_list: list[np.ndarray],
    lambdas: list[float] | None = None,
    rho: float = 1.0,
    n_iter: int = 80,
) -> tuple[float, float]:
    """
    Grid search for the best lambda on a (small) validation subset.

    Parameters
    ----------
    y_list   : list of degraded images
    gt_list  : list of corresponding ground-truth images
    lambdas  : values to try (default: log-spaced grid)
    rho      : ADMM penalty parameter

    Returns
    -------
    best_lam : float
    best_psnr: float
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from evaluation.metrics import psnr

    if lambdas is None:
        lambdas = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1]

    best_lam, best_score = lambdas[0], -np.inf

    for lam in lambdas:
        scores = []
        for y, gt in zip(y_list, gt_list):
            rec = tv_admm(y, lam=lam, rho=rho, n_iter=n_iter)
            scores.append(psnr(gt, rec))
        mean_psnr = float(np.mean(scores))
        print(f"  λ={lam:.4f}  →  PSNR={mean_psnr:.3f} dB")
        if mean_psnr > best_score:
            best_score, best_lam = mean_psnr, lam

    print(f"\n  Best λ={best_lam}  (PSNR={best_score:.3f} dB)")
    return best_lam, best_score
