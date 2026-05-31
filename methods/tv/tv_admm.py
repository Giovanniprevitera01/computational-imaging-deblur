"""
TV Deblurring via ADMM (Split Bregman).
Minimizza: (1/2)||Ax - y||^2 + lambda * TV(x)
dove A = operatore di blur gaussiano.
"""
import numpy as np
from scipy.ndimage import convolve

def blur_op(x, kernel):
    """Applica blur (operatore A)."""
    return np.stack(
        [convolve(x[:, :, c], kernel, mode="reflect") for c in range(3)],
        axis=2
    )


def tv_admm(y, kernel, lam=0.01, rho=1.0, n_iter=100):
    """
    y:      immagine degradata [H,W,3]
    kernel: kernel PSF
    lam:    regolarizzazione TV
    rho:    parametro ADMM
    """
    x = y.copy()

    z_h = np.zeros_like(x)
    z_v = np.zeros_like(x)
    u_h = np.zeros_like(x)
    u_v = np.zeros_like(x)

    def D_h(x): return np.roll(x, -1, axis=1) - x
    def D_v(x): return np.roll(x, -1, axis=0) - x
    def Dt_h(x): return np.roll(x, 1, axis=1) - x
    def Dt_v(x): return np.roll(x, 1, axis=0) - x

    def shrink(v, t):
        norm = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2 + 1e-8)
        factor = np.maximum(0, 1 - t / norm)
        return v * factor[..., None]

    for _ in range(n_iter):

        # --- x update ---
        Ax = blur_op(x, kernel)
        AT_Ax = blur_op(Ax, kernel)

        rhs = blur_op(y, kernel) + rho * (
            Dt_h(z_h - u_h) + Dt_v(z_v - u_v)
        )

        x = x + 0.1 * (
            rhs - AT_Ax - rho * (Dt_h(D_h(x)) + Dt_v(D_v(x)))
        )

        x = np.clip(x, 0, 1)

        # --- z update ---
        dh = D_h(x) + u_h
        dv = D_v(x) + u_v

        stacked = np.stack([dh, dv], axis=-1)
        z = shrink(stacked, lam / rho)

        z_h = z[..., 0]
        z_v = z[..., 1]

        # --- u update ---
        u_h += D_h(x) - z_h
        u_v += D_v(x) - z_v

    return np.clip(x, 0, 1)


def tune_lambda(y_val, x_gt_val, kernel, lambdas):
    from evaluation.metrics import psnr

    best_lam = None
    best_score = -np.inf

    for lam in lambdas:
        recon = tv_admm(y_val, kernel, lam=lam)
        score = psnr(x_gt_val, recon)

        if score > best_score:
            best_score = score
            best_lam = lam

    return best_lam, best_score
