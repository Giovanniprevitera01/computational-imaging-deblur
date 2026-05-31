# DPS — Diffusion Posterior Sampling

Chung et al., ICLR 2023: https://arxiv.org/abs/2209.14687

## Setup del checkpoint

1. Installa guided-diffusion:
```bash
pip install git+https://github.com/openai/guided-diffusion.git
```

2. Scarica il checkpoint FFHQ 256×256:
```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt \
     -P checkpoints/
```

3. Verifica:
