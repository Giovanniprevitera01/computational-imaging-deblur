"""
generate_degraded.py
--------------------
Generate degraded images (blur + noise) for all splits and noise levels.
Must be run AFTER prepare_dataset.py.

All methods will read from these pre-generated files to guarantee
that every method sees exactly the same degraded inputs.

Usage
-----
    python data/generate_degraded.py

Output
------
    data/processed/{split}/degraded/{sigma}/  for each split and sigma
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from degradation import degrade, NOISE_LEVELS, DEGRADATION_SEED

PROCESSED_DIR = Path("data/processed")
SPLITS        = ["train", "val", "test"]


def main():
    for split in SPLITS:
        clean_dir = PROCESSED_DIR / split / "clean"
        if not clean_dir.exists():
            print(f"[WARNING] {clean_dir} not found — run prepare_dataset.py first")
            continue

        image_files = sorted(clean_dir.glob("*.png")) + sorted(clean_dir.glob("*.jpg"))
        print(f"\n{split}: {len(image_files)} images × {len(NOISE_LEVELS)} noise levels")

        for sigma in NOISE_LEVELS:
            out_dir = PROCESSED_DIR / split / "degraded" / str(sigma)
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, f in enumerate(tqdm(image_files, desc=f"  σ={sigma}")):
                img_np = np.array(Image.open(f).convert("RGB")) / 255.0
                img_np = img_np.astype(np.float32)

                # Unique seed per image per sigma for reproducibility
                seed = DEGRADATION_SEED + i + int(sigma * 10000)
                deg  = degrade(img_np, sigma, seed=seed)

                # Save as PNG (lossless)
                deg_uint8 = (deg * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(deg_uint8).save(out_dir / f.name)

    print("\nDegradation complete!")


if __name__ == "__main__":
    main()
