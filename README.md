# Computational Imaging — Joint Deblurring & Denoising

**Group U | Academic Year 2025–2026**

> Comparison of three methodological families for joint Gaussian deblurring and denoising on FFHQ 256×256: **TV-ADMM** (variational), **NAF-Net** (end-to-end), and **DPS** (generative).

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Results Summary](#-results-summary)
- [Repository Structure](#-repository-structure)
- [Setup](#-setup)
- [Dataset](#-dataset)
- [Method 1 — TV-ADMM](#-method-1--tv-admm)
- [Method 2 — NAF-Net](#-method-2--naf-net)
- [Method 3 — DPS (Google Colab)](#-method-3--dps-google-colab)
- [Evaluation & Plots](#-evaluation--plots)
- [References](#-references)

---

## 🔬 Project Overview

### Degradation Model

```
y = A(x) + n
```

| Parameter | Value |
|-----------|-------|
| Forward operator A | Gaussian blur |
| Blur σ | 2 |
| Kernel size | 9×9 |
| Noise type | Additive Gaussian |
| Noise levels σ | **0.005, 0.01, 0.05, 0.1** |

> ⚠️ **All methods use identical pre-generated degraded images** — same random seed, same operator A — guaranteeing a fair comparison.

### Methods

| Method | Family | Description |
|--------|--------|-------------|
| **TV-ADMM** | Variational | Total Variation regularization via ADMM |
| **NAF-Net** | End-to-end | Nonlinear Activation Free Network with noise conditioning |
| **DPS** | Generative | Diffusion Posterior Sampling with pre-trained FFHQ model |

---

## 📊 Results Summary

### PSNR (dB) ↑

| Method | σ=0.005 | σ=0.01 | σ=0.05 | σ=0.1 |
|--------|---------|--------|--------|-------|
| **TV-ADMM** | 29.78 | 29.42 | 28.07 | 26.56 |
| **NAF-Net** | 29.46 | 29.37 | 28.07 | 26.64 |
| **DPS** | 25.56 | 24.63 | 24.63 | 24.80 |

### SSIM ↑

| Method | σ=0.005 | σ=0.01 | σ=0.05 | σ=0.1 |
|--------|---------|--------|--------|-------|
| **TV-ADMM** | 0.873 | 0.856 | 0.796 | 0.739 |
| **NAF-Net** | 0.867 | 0.863 | 0.797 | 0.720 |
| **DPS** | 0.790 | 0.810 | 0.737 | 0.690 |

> TV-ADMM and NAF-Net evaluated on **100 test images**. DPS evaluated on **5 test images** per noise level (GPU required — run on Google Colab T4).

---

## 📁 Repository Structure

```
computational-imaging-deblur/
│
├── data/
│   ├── degradation.py          # Blur + noise pipeline (shared by all methods)
│   ├── prepare_dataset.py      # Split FFHQ into train/val/test
│   ├── generate_degraded.py    # Generate and save degraded images
│   └── README.md
│
├── methods/
│   ├── tv/
│   │   ├── tv_admm.py          # TV-ADMM solver
│   │   ├── operators.py        # Blur and finite difference operators
│   │   └── README.md
│   ├── nafnet/
│   │   ├── model.py            # NAF-Net architecture
│   │   ├── train.py            # Training loop
│   │   ├── dataset.py          # PyTorch Dataset
│   │   └── README.md
│   └── dps/
│       ├── sample.py           # DPS sampling (local version)
│       ├── operators.py        # Differentiable blur operator
│       └── README.md
│
├── evaluation/
│   ├── metrics.py              # PSNR, SSIM
│   ├── evaluate_all.py         # Run evaluation on test set
│   └── plot_results.py         # Generate comparison plots
│
├── experiments/
│   ├── run_tv.py               # Run TV-ADMM on test set
│   ├── run_nafnet.py           # Run NAF-Net on test set
│   └── run_all.py              # Run all methods
│
├── notebooks/
│   └── DPS.ipynb               # ← DPS Google Colab notebook
│
├── results/
│   ├── metrics.json            # All numerical results
│   ├── plots/                  # PSNR/SSIM plots
│   └── images/                 # DPS reconstructed images
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Giovanniprevitera01/computational-imaging-deblur.git
cd computational-imaging-deblur

# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Create `__init__.py` files (required)

```bash
touch data/__init__.py
touch methods/__init__.py
touch methods/tv/__init__.py
touch methods/nafnet/__init__.py
touch methods/dps/__init__.py
touch evaluation/__init__.py
touch experiments/__init__.py
```

---

## 📦 Dataset

### Download FFHQ 256×256

```bash
# Option 1 — Kaggle CLI
pip install kaggle
# Place kaggle.json in ~/.kaggle/ then:
kaggle datasets download denislukovnikov/ffhq256-images-only
unzip ffhq256-images-only.zip -d data/raw/ffhq256/

# Option 2 — Manual download
# https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only
# Extract to data/raw/ffhq256/
```

### Prepare the dataset

```bash
# Step 1: Split into train / val / test (70% / 15% / 15%)
python3 data/prepare_dataset.py

# Expected output:
# Found 2000 images — splitting with seed=42
# Train: 1400 | Val: 300 | Test: 300

# Step 2: Generate degraded images for all noise levels
python3 data/generate_degraded.py

# Expected output:
# train: 1400 images × 4 noise levels ... done
# val:   300  images × 4 noise levels ... done
# test:  300  images × 4 noise levels ... done
```

After this step, the structure will be:
```
data/processed/
├── train/clean/        (1400 images)
├── train/degraded/0.005/ ... 0.1/
├── val/clean/          (300 images)
├── val/degraded/...
├── test/clean/         (300 images)
└── test/degraded/...
```

---

## 🔷 Method 1 — TV-ADMM

No training required. Runs directly on the test set.

```bash
# Run on all noise levels (100 images each)
python3 experiments/run_tv.py --max_images 100

# Run on a single noise level
python3 experiments/run_tv.py --sigma 0.05 --max_images 100

# Tune lambda on validation set first (optional)
python3 experiments/run_tv.py --tune --max_images 100
```

**Expected output:**
```
TV-ADMM | σ=0.005 | λ=0.001 | images=100
  PSNR=29.78 dB  SSIM=0.873
TV-ADMM | σ=0.01  | λ=0.005 | images=100
  PSNR=29.42 dB  SSIM=0.856
...
Saved to results/metrics.json
```

**Best λ per noise level (from validation set):**

| σ | Best λ |
|---|--------|
| 0.005 | 0.001 |
| 0.01  | 0.005 |
| 0.05  | 0.020 |
| 0.1   | 0.050 |

---

## 🟢 Method 2 — NAF-Net

### Train the model

```bash
# Full training (CPU: ~2 hours, GPU: ~20 min)
python3 methods/nafnet/train.py --epochs 30 --batch_size 4

# Monitor training:
# Epoch  1 | Loss 0.0842 | Val PSNR 25.12 dB | SSIM 0.712
# Epoch 10 | Loss 0.0412 | Val PSNR 27.55 dB | SSIM 0.823
# ...
# ✓ Best PSNR: 28.56 dB — checkpoint saved

# Checkpoint saved to:
ls checkpoints/nafnet_best.pth
```

### Evaluate

```bash
python3 experiments/run_nafnet.py --max_images 100

# Expected output:
# NAF-Net | σ=0.005 | images=100
#   PSNR=29.46 dB  SSIM=0.867
# ...
```

> **Note:** The model uses noise-level conditioning — a single model handles all 4 noise levels by appending σ as a 4th input channel.

---

## 🟣 Method 3 — DPS (Google Colab)

DPS requires a GPU. We use **Google Colab T4 GPU** (free tier).

> ⏱️ ~60 seconds per image on T4 GPU vs ~3 hours on CPU.

### Step 1 — Open the Colab notebook

Upload `notebooks/DPS.ipynb` to [colab.research.google.com](https://colab.research.google.com):

```
File → Upload notebook → select notebooks/DPS.ipynb
```

**Set runtime to GPU:**
```
Runtime → Change runtime type → Hardware accelerator: T4 GPU
```

### Step 2 — Prepare images to upload

On your local machine, identify the 5 test images to use:

```bash
ls data/processed/test/clean/ | head -5
# 00001.png  00006.png  00013.png  00014.png  00023.png
```

You will need to upload these images in 5 groups:
1. `data/processed/test/clean/` → the 5 clean images
2. `data/processed/test/degraded/0.005/` → same 5 images
3. `data/processed/test/degraded/0.01/` → same 5 images
4. `data/processed/test/degraded/0.05/` → same 5 images
5. `data/processed/test/degraded/0.1/` → same 5 images

> ⚠️ Use **the same 5 images** across all noise levels and clean — this guarantees coherent visual comparisons.

### Step 3 — Run the notebook cells in order

| Cell | What it does | Time |
|------|-------------|------|
| 1 | Install diffusers, scikit-image | ~1 min |
| 2 | Upload images (5 groups) | ~2 min |
| 3 | Download pre-trained DDPM model from HuggingFace | ~2 min |
| 4 | Define DPS sampling function | instant |
| 5 | **Run DPS on all 4 noise levels** | ~20 min |
| 6 | Visualize results and download zip | ~1 min |

**DPS parameters used:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| N steps | 200 | Best quality/speed trade-off |
| ζ (zeta) | 1.0 | Uniform across all σ |
| σ_y | = σ_noise | Likelihood strength |
| Post-processing | Gaussian blend α=0.10-0.15 | Only for σ ≥ 0.05 |

### Step 4 — Download and integrate results

After the notebook completes, download `dps_results.zip` and `dps_metrics.json`.

On your local machine:

```bash
cd ~/path/to/computational-imaging-deblur

# Extract results
unzip ~/Downloads/dps_results.zip -d dps_colab/

# Copy DPS reconstructed images
mkdir -p results/images/dps_0.005 results/images/dps_0.01
mkdir -p results/images/dps_0.05 results/images/dps_0.1

cp -r dps_colab/results/dps_0.005/* results/images/dps_0.005/
cp -r dps_colab/results/dps_0.01/*  results/images/dps_0.01/
cp -r dps_colab/results/dps_0.05/*  results/images/dps_0.05/
cp -r dps_colab/results/dps_0.1/*   results/images/dps_0.1/

# Merge DPS metrics into main metrics.json
python3 - << 'EOF'
import json

with open('results/metrics.json') as f:
    local = json.load(f)

with open('dps_colab/results/dps_metrics.json') as f:
    dps = json.load(f)

local['dps'] = dps['dps']

with open('results/metrics.json', 'w') as f:
    json.dump(local, f, indent=2)

print('metrics.json updated with DPS results!')
EOF
```
![Visual grid All Noise Level](result/plots/visual_grid_1.png)


## 📈 Evaluation & Plots

### Generate comparison plots

```bash
python3 evaluation/plot_results.py
```

This generates:
- `results/plots/psnr_vs_sigma.png` — PSNR curves for all methods
- `results/plots/ssim_vs_sigma.png` — SSIM curves
- `results/plots/combined_comparison.png` — Both metrics side by side

**Expected plots:**
```
Saved: results/plots/psnr_vs_sigma.png
Saved: results/plots/ssim_vs_sigma.png
Saved: results/plots/combined_comparison.png

Method        σ=0.005    σ=0.01    σ=0.05     σ=0.1
TV-ADMM        29.78     29.42     28.07      26.56
NAF-Net        29.46     29.37     28.07      26.64
DPS            25.56     24.63     24.63      24.80
```

### Print full results table

```bash
cat results/metrics.json | python3 -m json.tool
```

---

## 🚀 Quick Start (all steps in order)

```bash
# 1. Setup
git clone https://github.com/Giovanniprevitera01/computational-imaging-deblur.git
cd computational-imaging-deblur
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Dataset
# Download FFHQ to data/raw/ffhq256/ then:
python3 data/prepare_dataset.py
python3 data/generate_degraded.py

# 3. TV-ADMM (no training needed)
python3 experiments/run_tv.py --max_images 100

# 4. NAF-Net
python3 methods/nafnet/train.py --epochs 30 --batch_size 4
python3 experiments/run_nafnet.py --max_images 100

# 5. DPS → run notebooks/DPS.ipynb on Google Colab T4 GPU
#    then integrate results as described above

# 6. Generate plots
python3 evaluation/plot_results.py
```

---

## 📚 References

1. Rudin, L., Osher, S., & Fatemi, E. (1992). *Nonlinear total variation based noise removal algorithms.* Physica D.
2. Boyd, S. et al. (2011). *Distributed optimization via ADMM.* Foundations and Trends in ML.
3. Chen, L. et al. (2022). *Simple Baselines for Image Restoration (NAF-Net).* ECCV 2022.
4. Chung, H. et al. (2022). *Diffusion Posterior Sampling for General Noisy Inverse Problems.* ICLR 2023.
5. Blau, Y. & Michaeli, T. (2018). *The Perception-Distortion Tradeoff.* CVPR 2018.
6. Dhariwal, P. & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021.
7. Karras, T. et al. (2019). *A Style-Based Generator Architecture for GANs (FFHQ).* CVPR 2019.

---

## 📝 Notes on Reproducibility

- All experiments use **fixed random seed = 42**
- All methods use **identical degraded images** generated by `data/generate_degraded.py`
- NAF-Net checkpoint: `checkpoints/nafnet_best.pth` (not included — retrain with `train.py`)
- DPS uses `google/ddpm-celebahq-256` from HuggingFace (downloaded automatically by Colab notebook)
- DPS post-processing: Gaussian blend with α=0.10 (σ=0.05) and α=0.15 (σ=0.1) to reduce high-frequency artefacts — documented in `methods/dps/README.md`
