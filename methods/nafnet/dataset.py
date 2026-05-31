"""
dataset.py
----------
PyTorch Dataset for loading degraded/clean pairs from the pre-generated files.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.degradation import NOISE_LEVELS


class FFHQRestorationDataset(Dataset):
    """
    Dataset that loads clean and corresponding degraded images from disk.

    Parameters
    ----------
    split       : 'train', 'val', or 'test'
    processed_dir : path to data/processed/
    noise_levels : subset of noise levels to include (default: all 4)
    augment     : if True, apply random horizontal/vertical flips (train only)
    """

    def __init__(
        self,
        split: str = "train",
        processed_dir: str | Path = "data/processed",
        noise_levels: list[float] | None = None,
        augment: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.noise_levels  = noise_levels or NOISE_LEVELS
        self.augment       = augment

        clean_dir = self.processed_dir / split / "clean"
        self.clean_files = sorted(clean_dir.glob("*.png")) + sorted(clean_dir.glob("*.jpg"))

        if len(self.clean_files) == 0:
            raise FileNotFoundError(
                f"No images found in {clean_dir}. "
                "Run data/prepare_dataset.py and data/generate_degraded.py first."
            )

        # Each image appears once per noise level
        self.samples = [
            (f, sigma)
            for f in self.clean_files
            for sigma in self.noise_levels
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_path, sigma = self.samples[idx]

        # Load clean image
        clean_np = np.array(Image.open(clean_path).convert("RGB")) / 255.0
        clean_np = clean_np.astype(np.float32)

        # Load corresponding degraded image (pre-generated, same for all methods)
        deg_dir  = self.processed_dir / clean_path.parent.parent.name / "degraded" / str(sigma)
        deg_path = deg_dir / clean_path.name
        deg_np   = np.array(Image.open(deg_path).convert("RGB")) / 255.0
        deg_np   = deg_np.astype(np.float32)

        # To tensor (H,W,C) → (C,H,W)
        clean = torch.from_numpy(clean_np.transpose(2, 0, 1))
        deg   = torch.from_numpy(deg_np.transpose(2, 0, 1))

        # Data augmentation (train only)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                clean = torch.flip(clean, dims=[2])  # horizontal flip
                deg   = torch.flip(deg,   dims=[2])
            if torch.rand(1).item() > 0.5:
                clean = torch.flip(clean, dims=[1])  # vertical flip
                deg   = torch.flip(deg,   dims=[1])

        sigma_t = torch.tensor(sigma, dtype=torch.float32)
        return deg, clean, sigma_t
