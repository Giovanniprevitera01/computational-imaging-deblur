# NAF-Net

Nonlinear Activation Free Network (Chen et al., ECCV 2022).

## Architecture choice
NAF-Net was chosen over UNet and ViT because:
1. Designed specifically for image restoration (deblur + denoise)
2. Simple Gate replaces all non-linearities — faster and more stable
3. Outperforms UNet on SIDD and GoPro benchmarks
4. ViT requires more data and compute for comparable results

## Noise conditioning
Input: 4 channels (RGB + σ map). Single model covers all noise levels.

## Training
```bash
python methods/nafnet/train.py
```
Best checkpoint saved to `checkpoints/nafnet_best.pth`.
