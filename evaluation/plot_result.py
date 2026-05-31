import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_FILE = Path("results/metrics.json")
PLOTS_DIR    = Path("results/plots")
PLOTS_DIR.mkdir(exist_ok=True)

with open(RESULTS_FILE) as f:
    results = json.load(f)

SIGMAS  = [0.005, 0.01, 0.05, 0.1]
METHODS = {
    "tv_admm": {"label": "TV-ADMM",  "color": "#2471A3", "marker": "o", "ls": "--"},
    "nafnet":  {"label": "NAF-Net",  "color": "#27AE60", "marker": "s", "ls": "-"},
    "dps":     {"label": "DPS",      "color": "#8E44AD", "marker": "^", "ls": "-."},
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      12,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
})

# ── Plot 1: PSNR vs sigma ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("TV-ADMM vs NAF-Net vs DPS — FFHQ 256×256",
             fontsize=13, fontweight="bold")

for method, style in METHODS.items():
    if method not in results:
        continue
    psnrs = [results[method].get(str(s), {}).get("psnr", None) for s in SIGMAS]
    ssims = [results[method].get(str(s), {}).get("ssim", None) for s in SIGMAS]

    valid_p = [(s, p) for s, p in zip(SIGMAS, psnrs) if p is not None]
    valid_s = [(s, s2) for s, s2 in zip(SIGMAS, ssims) if s2 is not None]

    if valid_p:
        xs, ys = zip(*valid_p)
        axes[0].plot(xs, ys, label=style["label"], color=style["color"],
                     marker=style["marker"], linestyle=style["ls"])

    if valid_s:
        xs, ys = zip(*valid_s)
        axes[1].plot(xs, ys, label=style["label"], color=style["color"],
                     marker=style["marker"], linestyle=style["ls"])

for ax, ylabel, title in zip(axes,
    ["PSNR (dB) ↑", "SSIM ↑"],
    ["PSNR vs Noise Level", "SSIM vs Noise Level"]):
    ax.set_xscale("log")
    ax.set_xticks(SIGMAS)
    ax.set_xticklabels([str(s) for s in SIGMAS])
    ax.set_xlabel("Noise level σ")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "combined_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(PLOTS_DIR / "combined_comparison.pdf", dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: combined_comparison.png")

# ── Plot 2: PSNR separato ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for method, style in METHODS.items():
    if method not in results:
        continue
    psnrs = [results[method].get(str(s), {}).get("psnr", None) for s in SIGMAS]
    valid = [(s, p) for s, p in zip(SIGMAS, psnrs) if p is not None]
    if valid:
        xs, ys = zip(*valid)
        ax.plot(xs, ys, label=style["label"], color=style["color"],
                marker=style["marker"], linestyle=style["ls"])

ax.set_xscale("log")
ax.set_xticks(SIGMAS)
ax.set_xticklabels([str(s) for s in SIGMAS])
ax.set_xlabel("Noise level σ")
ax.set_ylabel("PSNR (dB) ↑")
ax.set_title("PSNR vs Noise Level", fontweight="bold")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "psnr_vs_sigma.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: psnr_vs_sigma.png")

# ── Plot 3: SSIM separato ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for method, style in METHODS.items():
    if method not in results:
        continue
    ssims = [results[method].get(str(s), {}).get("ssim", None) for s in SIGMAS]
    valid = [(s, s2) for s, s2 in zip(SIGMAS, ssims) if s2 is not None]
    if valid:
        xs, ys = zip(*valid)
        ax.plot(xs, ys, label=style["label"], color=style["color"],
                marker=style["marker"], linestyle=style["ls"])

ax.set_xscale("log")
ax.set_xticks(SIGMAS)
ax.set_xticklabels([str(s) for s in SIGMAS])
ax.set_xlabel("Noise level σ")
ax.set_ylabel("SSIM ↑")
ax.set_title("SSIM vs Noise Level", fontweight="bold")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ssim_vs_sigma.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: ssim_vs_sigma.png")

# ── Stampa tabella risultati ──────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Method':<12} {'σ=0.005':>10} {'σ=0.01':>10} {'σ=0.05':>10} {'σ=0.1':>10}")
print("="*70)
for method, style in METHODS.items():
    if method not in results:
        continue
    psnrs = [results[method].get(str(s), {}).get("psnr", 0) for s in SIGMAS]
    print(f"{style['label']:<12} " + " ".join(f"{p:>10.2f}" for p in psnrs))
print("="*70)
print(f"\n{'Method':<12} {'σ=0.005':>10} {'σ=0.01':>10} {'σ=0.05':>10} {'σ=0.1':>10}")
print("="*70)
for method, style in METHODS.items():
    if method not in results:
        continue
    ssims = [results[method].get(str(s), {}).get("ssim", 0) for s in SIGMAS]
    print(f"{style['label']:<12} " + " ".join(f"{s:>10.4f}" for s in ssims))
print("="*70)

print("\nTutti i plot salvati in results/plots/")
