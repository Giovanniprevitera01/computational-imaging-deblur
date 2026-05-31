"""
operators.py
------------
Linear operators used by the TV-ADMM solver.

    A   : Gaussian blur (forward model)
    A^T : transposed blur (same kernel, convolution is self-adjoint for symmetric kernels)
    D_h : horizontal finite differences
    D_v : vertical finite differences
    D_h^T, D_v^T : their transposes (adjoint differences)
"""

import numpy as np
from scipy.ndimage import convolve
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.degradation import get_gaussian_kernel


# ── Blur operator ──────────────────────────────────────────────────────────────

_KERNEL = None  # cached kernel

def get_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = get_gaussian_kernel()
    return _KERNEL


def A(x: np.ndarray) -> np.ndarray:
    """Gaussian blur: A(x). x is (H,W,3) float32."""
    k = get_kernel()
    return np.stack([convolve(x[:, :, c], k, mode="reflect") for c in range(3)], axis=2)


def At(x: np.ndarray) -> np.ndarray:
    """Transpose of Gaussian blur: A^T(x).
    For symmetric kernels and 'reflect' boundary, A^T ≈ A."""
    return A(x)   # symmetric PSF: A^T = A


def AtA(x: np.ndarray) -> np.ndarray:
    """A^T A (x) — blur applied twice."""
    return At(A(x))


# ── Finite difference operators ────────────────────────────────────────────────

def D_h(x: np.ndarray) -> np.ndarray:
    """Horizontal finite differences (forward): D_h(x)_{i,j} = x_{i,j+1} - x_{i,j}"""
    return np.roll(x, -1, axis=1) - x


def D_v(x: np.ndarray) -> np.ndarray:
    """Vertical finite differences (forward): D_v(x)_{i,j} = x_{i+1,j} - x_{i,j}"""
    return np.roll(x, -1, axis=0) - x


def Dt_h(x: np.ndarray) -> np.ndarray:
    """Adjoint of D_h: D_h^T(x)_{i,j} = x_{i,j-1} - x_{i,j}"""
    return np.roll(x, 1, axis=1) - x


def Dt_v(x: np.ndarray) -> np.ndarray:
    """Adjoint of D_v: D_v^T(x)_{i,j} = x_{i-1,j} - x_{i,j}"""
    return np.roll(x, 1, axis=0) - x


def DtD(x: np.ndarray) -> np.ndarray:
    """D^T D x = D_h^T D_h x + D_v^T D_v x (discrete Laplacian, negative sign)."""
    return Dt_h(D_h(x)) + Dt_v(D_v(x))


# ── TV norm ────────────────────────────────────────────────────────────────────

def tv_norm(x: np.ndarray) -> float:
    """Isotropic Total Variation: sum_ij sqrt((D_h x)^2 + (D_v x)^2)."""
    gh = D_h(x)
    gv = D_v(x)
    return float(np.sum(np.sqrt(gh ** 2 + gv ** 2 + 1e-8)))
