# Computational Imaging — Deblur & Denoising

**Group U | Academic Year 2024-2025**

Joint Gaussian deblurring and denoising on FFHQ 256×256,
comparing three methodological families:
- **TV-ADMM** — Total Variation via ADMM (variational)
- **NAF-Net** — Nonlinear Activation Free Network (end-to-end)
- **DPS** — Diffusion Posterior Sampling (generative)

## Setup

```bash
git clone https://github.com/TUOUSERNAME/computational-imaging-deblur.git
cd computational-imaging-deblur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download FFHQ 256x256 from Kaggle:
https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only

Place images in `data/raw/ffhq256/`, then run:
```bash
python data/prepare_dataset.py
python data/generate_degraded.py
```

## Usage

```bash
# Run all methods
python experiments/run_all.py

# Single method
python experiments/run_tv.py
python experiments/run_nafnet.py
python experiments/run_dps.py

# Evaluate
python evaluation/evaluate_all.py

# Plot results
python evaluation/plot_results.py
```

## Results

| Method  | σ=0.005 PSNR | σ=0.01 PSNR | σ=0.05 PSNR | σ=0.1 PSNR |
|---------|-------------|------------|------------|-----------|
| TV-ADMM | — | — | — | — |
| NAF-Net | — | — | — | — |
| DPS     | — | — | — | — |



## Repository Structure


## References

- Rudin et al. (1992) — Total Variation
- Chen et al. (2022) — NAF-Net (ECCV)
- Chung et al. (2022) — DPS (ICLR 2023)
- Dhariwal & Nichol (2021) — Guided Diffusion


