"""
visual_grid.py — griglia: Original / Degraded / TV / NAF-Net / DPS
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
from data.degradation import get_gaussian_kernel
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path



from methods.tv.tv_admm import tv_admm
from methods.nafnet.model import NAFNet
from data.degradation import NOISE_LEVELS

CHECKPOINT = Path("checkpoints/nafnet_best.pth")
TEST_CLEAN = Path("data/processed/test/clean")
TEST_DEG   = Path("data/processed/test/degraded")
DPS_DIR    = Path("results/images")
OUT_DIR    = Path("results/plots")
OUT_DIR.mkdir(exist_ok=True)

BEST_LAMBDA = {0.005: 0.001, 0.01: 0.005, 0.05: 0.02, 0.1: 0.05}

def load_img(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

def load_nafnet():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    width = state["intro.weight"].shape[0]
    enc_blks, dec_blks = [], []
    s = 0
    while any(f"encoders.{s}.{i}.beta" in state for i in range(10)):
        enc_blks.append(sum(1 for i in range(10) if f"encoders.{s}.{i}.beta" in state))
        s += 1
    s = 0
    while any(f"decoders.{s}.{i}.beta" in state for i in range(10)):
        dec_blks.append(sum(1 for i in range(10) if f"decoders.{s}.{i}.beta" in state))
        s += 1
    model = NAFNet(inp_channels=4, out_channels=3, width=width,
                   enc_blks=enc_blks, dec_blks=dec_blks)
    model.load_state_dict(state)
    model.eval()
    return model

def nafnet_infer(model, deg, sigma):
    t = torch.from_numpy(deg.transpose(2,0,1)).float().unsqueeze(0)
    s = torch.tensor([sigma])
    with torch.no_grad():
        out = model(t, s)
    return out.squeeze(0).permute(1,2,0).numpy().clip(0,1).astype(np.float32)

def main():
    test_files = sorted(TEST_CLEAN.glob("*.png"))[:3]
    model      = load_nafnet()

    col_names  = ["Original", "Degraded", "TV-ADMM", "NAF-Net", "DPS"]
    col_colors = ["#2C3E50", "#7F8C8D", "#2471A3", "#27AE60", "#8E44AD"]
    kernel = get_gaussian_kernel()

    for img_idx, test_img in enumerate(test_files):
        fig, axes = plt.subplots(
            len(NOISE_LEVELS), 5,
            figsize=(20, 4 * len(NOISE_LEVELS))
        )
        fig.suptitle(f"Method Comparison — {test_img.name}",
                     fontsize=14, fontweight="bold")

        for row, sigma in enumerate(NOISE_LEVELS):
            gt  = load_img(test_img)
            deg = load_img(TEST_DEG / str(sigma) / test_img.name)
            tv = tv_admm(deg, kernel, lam=BEST_LAMBDA[sigma])
            naf = nafnet_infer(model, deg, sigma)

            dps_path = DPS_DIR / f"dps_{sigma}" / test_img.name
            dps = load_img(dps_path) if dps_path.exists() else np.zeros_like(gt)

            for col, (img, name, color) in enumerate(zip(
                [gt, deg, tv, naf, dps],
                col_names, col_colors
            )):
                ax = axes[row, col]
                ax.imshow(img.clip(0,1))
                ax.axis("off")
                if row == 0:
                    ax.set_title(name, fontsize=12,
                                 fontweight="bold", color=color)
                if col == 0:
                    ax.set_ylabel(f"σ={sigma}", fontsize=11,
                                  rotation=90, labelpad=10)

        plt.tight_layout()
        out = OUT_DIR / f"visual_grid_{img_idx+1}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Salvato: {out}")

    print("Griglia visiva completata!")

if __name__ == "__main__":
    main()
