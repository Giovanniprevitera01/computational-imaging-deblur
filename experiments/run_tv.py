"""
Run TV-ADMM on the test set for all noise levels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.degradation import NOISE_LEVELS, get_gaussian_kernel
from methods.tv.tv_admm import tv_admm, tune_lambda
from evaluation.metrics import psnr, ssim


RESULTS_DIR  = Path("results")
TEST_CLEAN   = Path("data/processed/test/clean")
TEST_DEG_DIR = Path("data/processed/test/degraded")
VAL_CLEAN    = Path("data/processed/val/clean")
VAL_DEG_DIR  = Path("data/processed/val/degraded")


BEST_LAMBDA = {
    0.005: 0.001,
    0.01:  0.005,
    0.05:  0.02,
    0.1:   0.05,
}


def load_img(path):
    return np.array(Image.open(path).convert("RGB")) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=None,
                        help="Single noise level (default: all)")
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    sigmas = [args.sigma] if args.sigma else NOISE_LEVELS
    test_files = sorted(TEST_CLEAN.glob("*.png"))[:args.max_images]

    results = {}

    # 🔥 BLUR IS FIXED (from project spec)
    kernel = get_gaussian_kernel()

    for sigma in sigmas:

        lam = BEST_LAMBDA.get(sigma, 0.01)

        if args.tune:
            print(f"\nTuning λ for σ={sigma}...")
            val_files = sorted(VAL_CLEAN.glob("*.png"))[:30]

            y_list = [
                load_img(VAL_DEG_DIR / str(sigma) / f.name).astype(np.float32)
                for f in val_files
            ]
            gt_list = [
                load_img(f).astype(np.float32)
                for f in val_files
            ]

            lam, _ = tune_lambda(y_list, gt_list, kernel, [0.001, 0.005, 0.01, 0.02, 0.05])

        print(f"\nTV-ADMM | σ={sigma} | λ={lam} | images={len(test_files)}")

        psnrs, ssims = [], []

        for f in tqdm(test_files, desc=f"σ={sigma}"):

            gt  = load_img(TEST_CLEAN / f.name).astype(np.float32)
            deg = load_img(TEST_DEG_DIR / str(sigma) / f.name).astype(np.float32)

            rec = tv_admm(deg, kernel, lam=lam, rho=1.0, n_iter=100)

            rec = np.clip(rec, 0.0, 1.0).astype(np.float32)

            psnrs.append(psnr(gt, rec))
            ssims.append(ssim(gt, rec))

        results[str(sigma)] = {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)),
            "psnr_std": float(np.std(psnrs)),
            "ssim_std": float(np.std(ssims)),
            "lambda": lam,
            "n_images": len(psnrs),
        }

        print(f"  PSNR={results[str(sigma)]['psnr']:.2f} dB  "
              f"SSIM={results[str(sigma)]['ssim']:.4f}")

    out_path = RESULTS_DIR / "metrics.json"

    all_results = {}
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)

    all_results["tv_admm"] = results

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
