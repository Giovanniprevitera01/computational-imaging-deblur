"""
run_nafnet.py — valutazione NAF-Net sul test set
Carica automaticamente l'architettura dal checkpoint.
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
from methods.nafnet.model import NAFNet
from evaluation.metrics import psnr, ssim

RESULTS_DIR  = Path("results")
TEST_CLEAN   = Path("data/processed/test/clean")
TEST_DEG_DIR = Path("data/processed/test/degraded")
CHECKPOINT   = Path("checkpoints/nafnet_best.pth")


def load_img(path):
    return np.array(Image.open(path).convert("RGB")) / 255.0


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Carica il modello leggendo i parametri direttamente dal checkpoint,
    senza dover specificare l'architettura a mano.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Ricava width dal primo layer
    width = state["intro.weight"].shape[0]

    # Ricava enc_blks contando i layer per ogni stage
    enc_blks = []
    stage = 0
    while True:
        count = 0
        while f"encoders.{stage}.{count}.beta" in state:
            count += 1
        if count == 0:
            break
        enc_blks.append(count)
        stage += 1

    # dec_blks stessa logica
    dec_blks = []
    stage = 0
    while True:
        count = 0
        while f"decoders.{stage}.{count}.beta" in state:
            count += 1
        if count == 0:
            break
        dec_blks.append(count)
        stage += 1

    print(f"  Architettura rilevata: width={width}, enc_blks={enc_blks}, dec_blks={dec_blks}")

    model = NAFNet(
        inp_channels=4,
        out_channels=3,
        width=width,
        enc_blks=enc_blks,
        dec_blks=dec_blks,
    ).to(device)

    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma",      type=float, default=None)
    parser.add_argument("--max_images", type=int,   default=100)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint non trovato: {args.checkpoint}")
        print("Allena prima: python3 methods/nafnet/train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Checkpoint: {args.checkpoint}")

    model = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    print("Modello caricato.\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    sigmas     = [args.sigma] if args.sigma else NOISE_LEVELS
    test_files = sorted(TEST_CLEAN.glob("*.png"))[:args.max_images]
    results    = {}

    for sigma in sigmas:
        deg_dir = TEST_DEG_DIR / str(sigma)
        print(f"NAF-Net | σ={sigma} | images={len(test_files)}")
        psnrs, ssims = [], []

        for f in tqdm(test_files, desc=f"  σ={sigma}"):
            gt  = load_img(TEST_CLEAN / f.name).astype(np.float32)
            deg = load_img(deg_dir    / f.name).astype(np.float32)

            deg_t = torch.from_numpy(deg.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            sig_t = torch.tensor([sigma], dtype=torch.float32).to(device)

            with torch.no_grad():
                out = model(deg_t, sig_t)

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
        }
        print(f"  → PSNR={results[str(sigma)]['psnr']:.2f} dB  "
              f"SSIM={results[str(sigma)]['ssim']:.4f}\n")

    out_path    = RESULTS_DIR / "metrics.json"
    all_results = {}
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    all_results["nafnet"] = results

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Risultati salvati in {out_path}")


if __name__ == "__main__":
    main()
