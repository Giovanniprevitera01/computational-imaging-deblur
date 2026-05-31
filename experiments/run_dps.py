"""
run_dps.py
----------
Evaluate DPS on the test set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.degradation import NOISE_LEVELS
from methods.dps.sample import load_model_and_diffusion, dps_sample
from evaluation.metrics import psnr, ssim

RESULTS_DIR  = Path("results")
TEST_CLEAN   = Path("data/processed/test/clean")
TEST_DEG_DIR = Path("data/processed/test/degraded")
CHECKPOINT   = "checkpoints/256x256_diffusion_uncond.pt"


def load_img(path):
    return np.array(Image.open(path).convert("RGB")) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma",      type=float, default=None)
    parser.add_argument("--max_images", type=int,   default=20)
    parser.add_argument("--steps",      type=int,   default=100)
    parser.add_argument("--zeta",       type=float, default=0.5)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint non trovato: {args.checkpoint}")
        print("Vedi methods/dps/README.md per il download.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Steps: {args.steps} | ζ: {args.zeta}")
    print("Caricamento modello guided-diffusion...")
    model, diffusion = load_model_and_diffusion(args.checkpoint, device)

    RESULTS_DIR.mkdir(exist_ok=True)
    sigmas     = [args.sigma] if args.sigma else NOISE_LEVELS
    test_files = sorted(TEST_CLEAN.glob("*.png"))[:args.max_images]
    results    = {}

    for sigma in sigmas:
        deg_dir = TEST_DEG_DIR / str(sigma)
        print(f"\nDPS | σ={sigma} | steps={args.steps} | ζ={args.zeta} | images={len(test_files)}")
        psnrs, ssims = [], []

        for f in tqdm(test_files, desc=f"  σ={sigma}"):
            gt  = load_img(TEST_CLEAN / f.name).astype(np.float32)
            deg = load_img(deg_dir    / f.name).astype(np.float32)

            y = torch.from_numpy(deg.transpose(2, 0, 1)).float().unsqueeze(0)
            out = dps_sample(model, diffusion, y,
                 noise_std=sigma,
                 n_steps=args.steps, zeta=args.zeta, device=device)
            rec = np.clip(
                out.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1
            ).astype(np.float32)

            psnrs.append(psnr(gt, rec))
            ssims.append(ssim(gt, rec))

        results[str(sigma)] = {
            "psnr":     float(np.mean(psnrs)),
            "ssim":     float(np.mean(ssims)),
            "psnr_std": float(np.std(psnrs)),
            "ssim_std": float(np.std(ssims)),
            "n_images": len(psnrs),
            "steps":    args.steps,
            "zeta":     args.zeta,
        }
        print(f"  → PSNR={results[str(sigma)]['psnr']:.2f} dB  "
              f"SSIM={results[str(sigma)]['ssim']:.4f}")

    out_path    = RESULTS_DIR / "metrics.json"
    all_results = {}
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    all_results["dps"] = results

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSalvato in {out_path}")


if __name__ == "__main__":
    main()
