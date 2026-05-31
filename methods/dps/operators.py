"""
operators.py (DPS)
------------------
Forward operator A and its gradient for use inside the DPS sampling loop.

The operator must be differentiable w.r.t. the input x so that PyTorch
can compute the likelihood gradient: ∇_x ||A(x̂₀) - y||²
"""

import torch
import torch.nn.functional as F
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.degradation import get_gaussian_kernel, SIGMA_BLUR, KERNEL_SIZE


def _build_torch_kernel(device: torch.device) -> torch.Tensor:
    """Build a 4D conv kernel (out_c, in_c/groups, kH, kW) for depthwise conv."""
    k_np = get_gaussian_kernel(KERNEL_SIZE, SIGMA_BLUR).astype(np.float32)
    k    = torch.from_numpy(k_np).to(device)
    # Shape: (3, 1, kH, kW) — apply the same kernel to each channel independently
    return k.unsqueeze(0).unsqueeze(0).expand(3, 1, KERNEL_SIZE, KERNEL_SIZE)


_KERNEL_CACHE: dict[torch.device, torch.Tensor] = {}

def get_torch_kernel(device: torch.device) -> torch.Tensor:
    if device not in _KERNEL_CACHE:
        _KERNEL_CACHE[device] = _build_torch_kernel(device)
    return _KERNEL_CACHE[device]


def blur_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Gaussian blur for a batch of images.

    Parameters
    ----------
    x : (B, 3, H, W) float tensor in [0, 1]

    Returns
    -------
    blurred : (B, 3, H, W) float tensor
    """
    kernel = get_torch_kernel(x.device)
    pad    = KERNEL_SIZE // 2
    return F.conv2d(x, kernel, padding=pad, groups=3)


def likelihood_gradient(
    x_t:    torch.Tensor,
    x0_hat: torch.Tensor,
    y:      torch.Tensor,
    zeta:   float = 0.5,
) -> torch.Tensor:
    """
    Compute the DPS likelihood gradient:
        g = ζ · ∇_{x_t} ||A(x̂₀(x_t)) - y||²

    Parameters
    ----------
    x_t    : current noisy sample (B, 3, H, W), requires_grad=True
    x0_hat : denoised estimate from the diffusion model (B, 3, H, W)
    y      : degraded observation (B, 3, H, W)
    zeta   : step size for the likelihood guidance

    Returns
    -------
    grad : gradient w.r.t. x_t, same shape
    """
    # Forward model: blur the predicted clean image
    Ax0 = blur_torch(x0_hat)

    # Likelihood: ||A(x̂₀) - y||²
    likelihood = ((Ax0 - y) ** 2).sum()

    # Gradient through x_t (via x̂₀ which depends on x_t)
    grad = torch.autograd.grad(likelihood, x_t, create_graph=False)[0]

    return zeta * grad
