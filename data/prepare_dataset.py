import os, random, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_DIR   = Path("data/raw/ffhq256")      # dove Kaggle scarica le immagini
OUT_DIR   = Path("data/processed")
N_IMAGES  = 2000                          # usa 2000 immagini su 70k

def split_images(all_files, ratios=(0.70, 0.15, 0.15)):
    random.shuffle(all_files)
    n = len(all_files)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return (all_files[:n_train],
            all_files[n_train:n_train+n_val],
            all_files[n_train+n_val:])

def main():
    all_files = sorted(RAW_DIR.glob("*.png"))[:N_IMAGES]
    train, val, test = split_images(all_files)

    for split_name, files in [("train", train), ("val", val), ("test", test)]:
        for subdir in ["clean", "degraded"]:
            (OUT_DIR / split_name / subdir).mkdir(parents=True, exist_ok=True)
        for f in tqdm(files, desc=split_name):
            img = Image.open(f).convert("RGB").resize((256, 256), Image.LANCZOS)
            img.save(OUT_DIR / split_name / "clean" / f.name)

    print("Dataset pronto!")

if __name__ == "__main__":
    main()
