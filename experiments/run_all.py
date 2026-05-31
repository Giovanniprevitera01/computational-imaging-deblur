"""
Esegui tutti i metodi su tutti i livelli di rumore e salva i risultati.
"""
import numpy as np
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from data.degradation import degrade, get_gaussian_kernel, NOISE_LEVELS
from methods.tv_admm import tv_admm
from evaluation.metrics import psnr, ssim

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def run_tv(test_dir, kernel):
    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1]
    results = {}
    for sigma in NOISE_LEVELS:
        print(f"\n--- TV | sigma={sigma} ---")
        psnrs, ssims = [], []
        for f in tqdm(sorted(Path(test_dir).glob("*.png"))[:100]):
            img = np.array(Image.open(f).convert("RGB")) / 255.0
            deg = degrade(img.astype(np.float32), sigma)
            rec = tv_admm(deg, kernel, lam=0.01)   # usa il lambda ottimale!
            psnrs.append(psnr(img, rec))
            ssims.append(ssim(img, rec))
        results[sigma] = {"psnr": np.mean(psnrs), "ssim": np.mean(ssims)}
        print(f"  PSNR={results[sigma]['psnr']:.2f} | SSIM={results[sigma]['ssim']:.4f}")
    return results

def main():
    kernel  = get_gaussian_kernel()
    test_dir = "data/processed/test/clean"

    all_results = {}
    all_results["tv"] = run_tv(test_dir, kernel)
    # Aggiungi NAF-Net e DPS analogamente

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nRisultati salvati in results/results.json")

if __name__ == "__main__":
    main()
