# TV-ADMM

Solves: argmin_x (1/2)||Ax-y||² + λ·TV(x)

via Alternating Direction Method of Multipliers.

## Parameters
- `lam`: regularization weight (tuned per noise level on val set)
- `rho`: ADMM penalty parameter (fixed to 1.0)
- `n_iter`: number of ADMM iterations (default: 100)

## Lambda selection (validation set)
| σ     | Best λ |
|-------|--------|
| 0.005 | 0.001  |
| 0.01  | 0.005  |
| 0.05  | 0.02   |
| 0.1   | 0.05   |
