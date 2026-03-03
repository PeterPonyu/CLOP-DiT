#!/usr/bin/env python3
"""Compare v6.3 vs v6.4.1 training results — comprehensive dashboard."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/home/zeyufu/Desktop/CLOP-DiT/results")
OUT_DIR.mkdir(exist_ok=True)

# Load histories
with open("/home/zeyufu/Desktop/CLOP-DiT/models/checkpoints/v63_backup/clop_history.json") as f:
    h63 = json.load(f)
with open("/home/zeyufu/Desktop/CLOP-DiT/models/checkpoints/clop_history.json") as f:
    h641 = json.load(f)

e63 = np.arange(1, len(h63["train_loss"]) + 1)
e641 = np.arange(1, len(h641["train_loss"]) + 1)

# ── Figure: 6-panel comparison ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("CLOP v6.3 vs v6.4.1 Comparison", fontsize=16, fontweight="bold")

# Panel 1: Val Proto Accuracy
ax = axes[0, 0]
ax.plot(e63, [v*100 for v in h63["val_proto_acc"]], "b-", linewidth=1.5, label="v6.3")
ax.plot(e641, [v*100 for v in h641["val_proto_acc"]], "r-", linewidth=1.5, label="v6.4.1")
ax.axhline(y=100/102, color="gray", linestyle="--", alpha=0.5, label="Random (~1%)")

best63 = max(h63["val_proto_acc"])
best641 = max(h641["val_proto_acc"])
best63_ep = h63["val_proto_acc"].index(best63) + 1
best641_ep = h641["val_proto_acc"].index(best641) + 1
ax.scatter([best63_ep], [best63*100], color="blue", s=80, zorder=5, marker="*")
ax.scatter([best641_ep], [best641*100], color="red", s=80, zorder=5, marker="*")

ax.set_xlabel("Epoch")
ax.set_ylabel("Val Proto Acc (%)")
ax.set_title(f"Val Proto Acc\nv6.3: {best63*100:.1f}% (ep{best63_ep}) | v6.4.1: {best641*100:.1f}% (ep{best641_ep})")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Val Proto Top-5
ax = axes[0, 1]
if "val_proto_top5" in h641:
    ax.plot(e641, [v*100 for v in h641["val_proto_top5"]], "r-", linewidth=1.5, label="v6.4.1 Top-5")
    best_top5 = max(h641["val_proto_top5"])
    ax.set_title(f"Val Proto Top-5 Acc (v6.4.1 only)\nBest: {best_top5*100:.1f}%")
ax.set_xlabel("Epoch")
ax.set_ylabel("Top-5 Acc (%)")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Train vs Val Proto Acc (v6.4.1)
ax = axes[0, 2]
if "train_proto_acc" in h641:
    ax.plot(e641, [v*100 for v in h641["train_proto_acc"]], "r--", linewidth=1.5, label="v6.4.1 Train Proto")
    ax.plot(e641, [v*100 for v in h641["val_proto_acc"]], "r-", linewidth=1.5, label="v6.4.1 Val Proto")
    ax.fill_between(e641,
                     [v*100 for v in h641["val_proto_acc"]],
                     [v*100 for v in h641["train_proto_acc"]],
                     alpha=0.15, color="red")
ax.set_xlabel("Epoch")
ax.set_ylabel("Proto Acc (%)")
ax.set_title("Train-Val Gap (v6.4.1)")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Train/Val Loss
ax = axes[1, 0]
ax.plot(e63, h63["train_loss"], "b--", linewidth=1, alpha=0.7, label="v6.3 Train")
ax.plot(e63, h63["val_loss"], "b-", linewidth=1.5, label="v6.3 Val")
ax.plot(e641, h641["train_loss"], "r--", linewidth=1, alpha=0.7, label="v6.4.1 Train")
ax.plot(e641, h641["val_loss"], "r-", linewidth=1.5, label="v6.4.1 Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Train/Val Loss Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 5: Temperature
ax = axes[1, 1]
ax.plot(e63, h63["temperature"], "b-", linewidth=1.5, label="v6.3")
ax.plot(e641, h641["temperature"], "r-", linewidth=1.5, label="v6.4.1")
ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="Cap (20)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Temperature")
ax.set_title("Learned Temperature")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 6: Summary table
ax = axes[1, 2]
ax.axis("off")
summary = [
    ["", "v6.3", "v6.4.1", "Change"],
    ["Val Proto Acc", f"{best63*100:.1f}%", f"{best641*100:.1f}%", f"{(best641-best63)*100:+.1f}%"],
    ["Best Epoch", str(best63_ep), str(best641_ep), ""],
    ["Total Epochs", str(len(e63)), str(len(e641)), ""],
    ["Layers", "3", "3", "="],
    ["Dropout", "0.2", "0.3", "+0.1"],
    ["Cell Noise", "0.05", "0.1", "2x"],
    ["Weight Decay", "0.05", "0.08", "1.6x"],
    ["LR", "1e-3", "5e-4", "0.5x"],
    ["Cohesion", "0.1", "0.2", "2x"],
    ["Model Params", "3.2M", "3.2M", "="],
    ["Train Proto*", "87.2%", "55.3%", "-31.9%"],
    ["Gap (T/V)*", "8.3x", "5.1x", "-3.2x"],
]
table = ax.table(cellText=summary, loc="center", cellLoc="center", colWidths=[0.28, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#D6E4F0")
ax.set_title("v6.3 → v6.4.1 Summary\n(*at best checkpoint)", fontsize=11, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "v63_vs_v641_comparison.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved {OUT_DIR / 'v63_vs_v641_comparison.png'}")

# ── Figure 2: Overfitting analysis ──
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Overfitting Analysis: v6.3 vs v6.4.1", fontsize=14, fontweight="bold")

# v6.3 gap
gap63 = [v - t for v, t in zip(h63["val_loss"], h63["train_loss"])]
ax1.fill_between(e63, 0, gap63, alpha=0.3, color="blue")
ax1.plot(e63, gap63, "b-", linewidth=1.5, label="v6.3 Gap")
gap641 = [v - t for v, t in zip(h641["val_loss"], h641["train_loss"])]
ax1.fill_between(e641, 0, gap641, alpha=0.3, color="red")
ax1.plot(e641, gap641, "r-", linewidth=1.5, label="v6.4.1 Gap")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Val Loss - Train Loss")
ax1.set_title("Generalization Gap (Loss)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Proto acc bar comparison
categories = ["Train\nProto", "Val\nProto", "Val\nTop-5", "Val\nTop-10"]
v63_vals = [87.2, 10.5, 29.4, 41.2]
v641_vals = [55.3, 10.8, 28.8, 39.6]

x = np.arange(len(categories))
w = 0.35
ax2.bar(x - w/2, v63_vals, w, label="v6.3", color="#4472C4", alpha=0.8)
ax2.bar(x + w/2, v641_vals, w, label="v6.4.1", color="#ED7D31", alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Metric Comparison at Best Checkpoint")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Add values on bars
for i, (v63, v641) in enumerate(zip(v63_vals, v641_vals)):
    ax2.text(i - w/2, v63 + 1, f"{v63:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2.text(i + w/2, v641 + 1, f"{v641:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig2.savefig(OUT_DIR / "v63_vs_v641_overfitting.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved {OUT_DIR / 'v63_vs_v641_overfitting.png'}")

plt.close("all")
print("\nDone. All comparison figures saved.")
