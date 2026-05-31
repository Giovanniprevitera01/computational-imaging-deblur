# TV-ADMM — Variational Method

## Method

Solves the convex optimization problem:

```
x* = argmin_x  (1/2) ||Ax - y||²  +  λ · TV(x)
```

where:
- `A` = Gaussian blur operator (σ=2, kernel 9×9)
- `y` = degraded observation
- `TV(x)` = isotropic Total Variation = Σᵢⱼ √((D_h x)²ᵢⱼ + (D_v x)²ᵢⱼ)
- `λ` = regularization weight (tuned per noise level)

## Solver: ADMM

Introduces auxiliary variables `z_h ≈ D_h x`, `z_v ≈ D_v x` and solves via:

1. **x-update**: linear system `(AᵀA + ρ DᵀD) x = rhs` (approximated with gradient steps)
2. **z-update**: soft-thresholding (proximal operator of `λ/ρ · ||·||₂,₁`)
3. **u-update**: dual ascent `u ← u + Dx − z`

Convergence is guaranteed for convex objectives. Typical: 100 outer iterations.

## Hyperparameter Selection

### λ (regularization) — tuned per noise level on validation set

Grid tested: `{0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1}`

| σ_noise | Best λ | Notes |
|---------|--------|-------|
| 0.005 | 0.001 | Low noise → mild regularization |
| 0.01  | 0.005 | |
| 0.05  | 0.02  | Higher noise → stronger smoothing |
| 0.1   | 0.05  | Strong regularization needed |

### ρ (ADMM penalty) — fixed to 1.0
Robust to moderate changes. Values in [0.5, 2.0] give similar results.

### n_iter — fixed to 100
Convergence typically within 50 iterations; 100 adds safety margin.

## Failure Modes

- **Over-smoothing**: TV promotes piecewise-constant solutions → texture loss (hair, fabric)
- **Staircasing**: visible block artifacts in smooth gradients
- **Ringing**: at strong edges under high noise

## Usage

```python
from methods.tv.admm import tv_admm, tune_lambda

# Restore a single image
restored = tv_admm(y_degraded, lam=0.02, rho=1.0, n_iter=100)

# Tune lambda on validation set
best_lam, best_psnr = tune_lambda(y_val_list, gt_val_list)
```
