"""
train.py (CPU-OPTIMIZED)
------------------------
Fast and lightweight NAF-Net training for university project.

Balanced for:
- CPU training
- reasonable quality
- acceptable training time
- good PSNR progression

Usage
-----
python3 -m methods.nafnet.train
python3 -m methods.nafnet.train --epochs 15 --batch_size 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..")
)

from methods.nafnet.model import NAFNet
from methods.nafnet.dataset import FFHQRestorationDataset
from evaluation.metrics import psnr_torch, ssim_torch


# -------------------------------------------------
# Combined Loss
# -------------------------------------------------
class L1SSIMLoss(nn.Module):

    def __init__(self, ssim_weight=0.1):
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):

        l1 = nn.functional.l1_loss(pred, target)

        ssim = ssim_torch(pred, target)

        return l1 + self.ssim_weight * (1.0 - ssim)


# -------------------------------------------------
# Training
# -------------------------------------------------
def train(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 60)
    print("NAF-Net Training (CPU Optimized)")
    print(f"Device: {device}")
    print("=" * 60 + "\n")

    # -------------------------------------------------
    # DATASETS
    # -------------------------------------------------
    print("Loading datasets...")

    train_ds_full = FFHQRestorationDataset(
        "train",
        augment=True
    )

    val_ds_full = FFHQRestorationDataset(
        "val",
        augment=False
    )

    # smaller subset for CPU
    rng = np.random.RandomState(42)

    train_indices = rng.choice(
        len(train_ds_full),
        size=min(800, len(train_ds_full)),
        replace=False
    )

    val_indices = rng.choice(
        len(val_ds_full),
        size=min(120, len(val_ds_full)),
        replace=False
    )

    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}\n")

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    print("Building lightweight NAF-Net...")

    model = NAFNet(
        inp_channels=4,
        out_channels=3,
        width=12,
        enc_blks=[1, 1, 1],
        dec_blks=[1, 1, 1],
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {n_params:,}\n")

    # -------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.7
    )

    criterion = L1SSIMLoss(ssim_weight=0.1)

    # -------------------------------------------------
    # CHECKPOINTS
    # -------------------------------------------------
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    history = {
        "train_loss": [],
        "val_psnr": [],
        "val_ssim": []
    }

    best_psnr = 0.0

    # -------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        model.train()

        train_losses = []

        pbar = tqdm(
            train_dl,
            desc=f"Epoch {epoch:02d}/train",
            leave=False
        )

        # IMPORTANT:
        # limit batches for CPU speed
        max_batches = 120

        for batch_idx, (deg, clean, sigma) in enumerate(pbar):

            if batch_idx >= max_batches:
                break

            deg = deg.to(device)
            clean = clean.to(device)
            sigma = sigma.to(device)

            pred = model(deg, sigma)

            loss = criterion(pred, clean)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_losses.append(loss.item())

            pbar.set_postfix(
                loss=f"{np.mean(train_losses):.4f}"
            )

        scheduler.step()

        mean_loss = np.mean(train_losses)

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------
        model.eval()

        val_psnrs = []
        val_ssims = []

        with torch.no_grad():

            for deg, clean, sigma in tqdm(
                val_dl,
                desc=f"Epoch {epoch:02d}/val",
                leave=False
            ):

                deg = deg.to(device)
                clean = clean.to(device)
                sigma = sigma.to(device)

                pred = model(deg, sigma)

                val_psnrs.append(
                    psnr_torch(clean, pred).item()
                )

                val_ssims.append(
                    ssim_torch(clean, pred).item()
                )

        mean_psnr = np.mean(val_psnrs)
        mean_ssim = np.mean(val_ssims)

        history["train_loss"].append(float(mean_loss))
        history["val_psnr"].append(float(mean_psnr))
        history["val_ssim"].append(float(mean_ssim))

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {mean_loss:.4f} | "
            f"PSNR {mean_psnr:.2f} dB | "
            f"SSIM {mean_ssim:.4f}"
        )

        # save best
        if mean_psnr > best_psnr:

            best_psnr = mean_psnr

            torch.save(
                model.state_dict(),
                ckpt_dir / "nafnet_best.pth"
            )

            print(
                f"   ✓ Best model saved "
                f"(PSNR {best_psnr:.2f} dB)"
            )

        # save latest
        torch.save(
            model.state_dict(),
            ckpt_dir / "nafnet_last.pth"
        )

    # -------------------------------------------------
    # SAVE HISTORY
    # -------------------------------------------------
    with open(
        ckpt_dir / "nafnet_history.json",
        "w"
    ) as f:

        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoint: {ckpt_dir / 'nafnet_best.pth'}")
    print("=" * 60 + "\n")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    train(args)
