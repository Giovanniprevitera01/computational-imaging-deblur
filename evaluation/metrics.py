import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def psnr(gt, pred):
    """numpy float32 [0,1] → PSNR dB."""
    return peak_signal_noise_ratio(gt, pred, data_range=1.0)

def ssim(gt, pred):
    """numpy float32 [0,1] → SSIM."""
    return structural_similarity(gt, pred, channel_axis=2, data_range=1.0)

def psnr_torch(gt, pred):
    """Tensor (B,C,H,W) [0,1] → PSNR medio del batch."""
    mse = ((gt - pred)**2).mean(dim=[1,2,3])
    return (10 * torch.log10(1.0 / mse)).mean()

def ssim_torch(gt, pred, window_size=11):
    """Approssimazione SSIM differenziabile."""
    C1, C2 = 0.01**2, 0.03**2
    mu1 = torch.nn.functional.avg_pool2d(gt,   window_size, stride=1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    s1  = torch.nn.functional.avg_pool2d(gt*gt,     window_size, stride=1, padding=window_size//2) - mu1_sq
    s2  = torch.nn.functional.avg_pool2d(pred*pred, window_size, stride=1, padding=window_size//2) - mu2_sq
    s12 = torch.nn.functional.avg_pool2d(gt*pred,   window_size, stride=1, padding=window_size//2) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*s12 + C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return ssim_map.mean()

def evaluate_all(gt_dir, pred_dir):
    """Valuta tutte le immagini in due directory e ritorna media PSNR/SSIM."""
    from pathlib import Path
    from PIL import Image
    gt_files   = sorted(Path(gt_dir).glob("*.png"))
    pred_files = sorted(Path(pred_dir).glob("*.png"))
    psnrs, ssims = [], []
    for gf, pf in zip(gt_files, pred_files):
        gt   = np.array(Image.open(gf).convert("RGB")) / 255.0
        pred = np.array(Image.open(pf).convert("RGB")) / 255.0
        psnrs.append(psnr(gt, pred))
        ssims.append(ssim(gt, pred))
    return {"psnr": np.mean(psnrs), "ssim": np.mean(ssims),
            "psnr_std": np.std(psnrs), "ssim_std": np.std(ssims)}
