"""
evaluate_all.py
---------------
Evaluate all three methods on the test set and save results to results/metrics.json.

Usage
-----
    python evaluation/evaluate_all.py

Requires
--------
    - data/processed/test/ (clean and degraded images)
    - checkpoints/nafnet_best.pth (for NAF-Net)
    - checkpoints/256x256_diffusion_uncond.pt (for DPS)
    - results/ directory
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.degradation import NOISE_LEVELS, get_gaussian_kernel
from evaluation.metrics import psnr, ssim

RESULTS_DIR  = Path("results")
TEST_CLEAN   = Path("data/processed/test/clean")
TEST_DEG_DIR = Path("data/processed/test/degraded")
MAX_IMAGES   = 100   # evaluate on first 100 test images (faster)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_img(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")) / 255.0


def eval_method(method_fn, sigma: float, test_files: list[Path]) -> dict:
    """Run method_fn on all test images and compute mean PSNR/SSIM."""
    psnrs, ssims = [], []
    deg_dir = TEST_DEG_DIR / str(sigma)

    for f in tqdm(test_files, desc=f"  σ={sigma}", leave=False):
        gt  = load_img(TEST_CLEAN / f.name).astype(np.float32)
        deg = load_img(deg_dir    / f.name).astype(np.float32)
        rec = method_fn(deg, sigma)
        rec = np.clip(rec, 0.0, 1.0).astype(np.float32)
        psnrs.append(psnr(gt, rec))
        ssims.append(ssim(gt, rec))

    return {
        "psnr":      float(np.mean(psnrs)),
        "ssim":      float(np.mean(ssims)),
        "psnr_std":  float(np.std(psnrs)),
        "ssim_std":  float(np.std(ssims)),
        "n_images":  len(psnrs),
    }


# ── Method wrappers ────────────────────────────────────────────────────────────

def make_tv_fn():
    from methods.tv.admm import tv_admm
    # Best lambda per noise level (tuned on validation set)
    best_lambda = {0.005: 0.001, 0.01: 0.005, 0.05: 0.02, 0.1: 0.05}

    def tv_fn(deg: np.ndarray, sigma: float) -> np.ndarray:
        lam = best_lambda.get(sigma, 0.01)
        return tv_admm(deg, lam=lam, rho=1.0, n_iter=100)
    return tv_fn


def make_nafnet_fn():
    from methods.nafnet.model import build_nafnet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_nafnet(pretrained="checkpoints/nafnet_best.pth", device=str(device))
    model.eval()

    def nafnet_fn(deg: np.ndarray, sigma: float) -> np.ndarray:
        deg_t = torch.from_numpy(deg.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        sig_t = torch.tensor([sigma], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(deg_t, sig_t)
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return nafnet_fn


def make_dps_fn():
    from methods.dps.sample import load_guided_diffusion, dps_sample
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diffusion = load_guided_diffusion(device=device)

    def dps_fn(deg: np.ndarray, sigma: float) -> np.ndarray:
        y   = torch.from_numpy(deg.transpose(2, 0, 1)).float().unsqueeze(0)
        out = dps_sample(model, diffusion, y, n_steps=100, zeta=0.5, device=device)
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return dps_fn


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    test_files = sorted(TEST_CLEAN.glob("*.png"))[:MAX_IMAGES]
    print(f"Evaluating on {len(test_files)} test images × {len(NOISE_LEVELS)} noise levels\n")

    results = {}

    # TV-ADMM
    print("=== TV-ADMM ===")
    tv_fn = make_tv_fn()
    results["tv_admm"] = {}
    for sigma in NOISE_LEVELS:
        results["tv_admm"][str(sigma)] = eval_method(tv_fn, sigma, test_files)
        r = results["tv_admm"][str(sigma)]
        print(f"  σ={sigma}  PSNR={r['psnr']:.2f} ± {r['psnr_std']:.2f}  SSIM={r['ssim']:.4f}")

    # NAF-Net
    if Path("checkpoints/nafnet_best.pth").exists():
        print("\n=== NAF-Net ===")
        nafnet_fn = make_nafnet_fn()
        results["nafnet"] = {}
        for sigma in NOISE_LEVELS:
            results["nafnet"][str(sigma)] = eval_method(nafnet_fn, sigma, test_files)
            r = results["nafnet"][str(sigma)]
            print(f"  σ={sigma}  PSNR={r['psnr']:.2f} ± {r['psnr_std']:.2f}  SSIM={r['ssim']:.4f}")
    else:
        print("\n[SKIP] NAF-Net checkpoint not found. Run: python methods/nafnet/train.py")

    # DPS
    if Path("checkpoints/256x256_diffusion_uncond.pt").exists():
        print("\n=== DPS ===")
        dps_fn = make_dps_fn()
        results["dps"] = {}
        for sigma in NOISE_LEVELS:
            # DPS is slow: evaluate on fewer images
            results["dps"][str(sigma)] = eval_method(dps_fn, sigma, test_files[:20])
            r = results["dps"][str(sigma)]
            print(f"  σ={sigma}  PSNR={r['psnr']:.2f} ± {r['psnr_std']:.2f}  SSIM={r['ssim']:.4f}")
    else:
        print("\n[SKIP] DPS checkpoint not found. See methods/dps/README.md for setup.")

    # Save results
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
