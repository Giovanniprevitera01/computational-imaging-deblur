# Data

## Structure
- `raw/ffhq256/` — original FFHQ images (not tracked by git, download manually)
- `processed/train/clean/` — training clean images
- `processed/train/degraded/{sigma}/` — degraded images per noise level
- `processed/val/` — validation split
- `processed/test/` — test split

## Download
```bash
kaggle datasets download denislukovnikov/ffhq256-images-only
unzip ffhq256-images-only.zip -d data/raw/ffhq256/
```

## Generate
```bash
python data/prepare_dataset.py   # split in train/val/test
python data/generate_degraded.py # crea le immagini degradate
```
