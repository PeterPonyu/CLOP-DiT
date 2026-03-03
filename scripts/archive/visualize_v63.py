#!/usr/bin/env python3
"""Visualize v6.3 CLOP training results — comprehensive dashboard."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HISTORY = Path("models/checkpoints/clop_history.json")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

with open(HISTORY) as f:
    h = json.load(f)

epochs = np.arange(1, len(h["train_loss"]) + 1)
n_epochs = len(epochs)

# ── Figure 1: 6-panel training dashboard ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("CLOP v6.3 Training Dashboard (PrototypeSigLIP)", fontsize=16, fontweight="bold")

# Panel 1: Train/Val Loss
ax = axes[0, 0]
ax.plot(epochs, h["train_loss"], "b-", linewidth=1.5, label="Train Loss")
ax.plot(epochs, h["val_loss"], "r-", linewidth=1.5, label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Train vs Val Loss")
ax.legend()
ax.grid(True, alpha=0.3)
# Annotate the gap
mid = n_epochs // 2
gap_mid = h["val_loss"][mid] - h["train_loss"][mid]
ax.annotate(f"Gap ≈ {gap_mid:.3f}", xy=(mid+1, h["val_loss"][mid]),
            fontsize=9, color="darkred")

# Panel 2: Val Proto Accuracy (the key metric)
ax = axes[0, 1]
proto_acc = h.get("val_proto_acc", [0]*n_epochs)
ax.plot(epochs, [v*100 for v in proto_acc], "g-", linewidth=2, label="Val Proto Acc")
best_idx = np.argmax(proto_acc)
best_val = proto_acc[best_idx] * 100
ax.axhline(y=100/102, color="gray", linestyle="--", alpha=0.5, label="Random (~1%)")
ax.scatter([best_idx+1], [best_val], color="red", s=100, zorder=5,
           label=f"Best: {best_val:.1f}% (ep {best_idx+1})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Val Prototype Accuracy (key metric)")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Val Individual Accuracy (misleading metric for reference)
ax = axes[0, 2]
ax.plot(epochs, [v*100 for v in h["val_acc"]], "m-", linewidth=1.5, label="Val Individual Acc")
ax.axhline(y=100/1024, color="gray", linestyle="--", alpha=0.5, label="Random (~0.1%)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Val Individual Acc (misleading for PrototypeSigLIP)")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Temperature
ax = axes[1, 0]
ax.plot(epochs, h["temperature"], "orange", linewidth=2)
ax.axhline(y=20.0, color="red", linestyle="--", alpha=0.5, label="Max cap (20)")
ax.axhline(y=14.0, color="blue", linestyle="--", alpha=0.5, label="Init (14)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Temperature (logit scale)")
ax.set_title("Learned Temperature")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 5: Train-Val Loss Gap (overfitting indicator)
ax = axes[1, 1]
gap = [v - t for v, t in zip(h["val_loss"], h["train_loss"])]
ax.fill_between(epochs, 0, gap, alpha=0.3, color="red")
ax.plot(epochs, gap, "r-", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Loss - Train Loss")
ax.set_title("Generalization Gap (overfitting)")
ax.grid(True, alpha=0.3)

# Panel 6: Proto Acc vs Loss correlation
ax = axes[1, 2]
sc = ax.scatter(h["val_loss"], [v*100 for v in proto_acc],
                c=epochs, cmap="viridis", s=30, edgecolors="k", linewidths=0.3)
plt.colorbar(sc, ax=ax, label="Epoch")
ax.set_xlabel("Val Loss")
ax.set_ylabel("Val Proto Acc (%)")
ax.set_title("Proto Acc vs Val Loss (colored by epoch)")
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "v63_dashboard.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved {OUT_DIR / 'v63_dashboard.png'}")

# ── Figure 2: Detailed metric summary table ──────────────────
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.axis("off")

# Compute summary stats
summary_data = [
    ["Total Epochs", f"{n_epochs}"],
    ["Best Epoch (proto_acc)", f"{best_idx+1}"],
    ["Best Val Proto Acc", f"{best_val:.1f}% (random ≈ 1.0%)"],
    ["Best Val Indiv Acc", f"{max(h['val_acc'])*100:.1f}% (random ≈ 0.1%)"],
    ["Final Train Loss", f"{h['train_loss'][-1]:.4f}"],
    ["Final Val Loss", f"{h['val_loss'][-1]:.4f}"],
    ["Val Loss at Best Epoch", f"{h['val_loss'][best_idx]:.4f}"],
    ["Train-Val Gap (final)", f"{h['val_loss'][-1] - h['train_loss'][-1]:.4f}"],
    ["Temperature", f"14.0 → 20.0 (hit cap at epoch 5)"],
    ["Early Stopping", f"Patience=30, stopped at epoch {n_epochs}"],
    ["Groups per Val Batch", f"~102 (from 1024 cells)"],
    ["Improvement over Random", f"{best_val / (100/102):.1f}x"],
]

table = ax2.table(
    cellText=summary_data,
    colLabels=["Metric", "Value"],
    loc="center",
    cellLoc="left",
    colWidths=[0.4, 0.6],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#D6E4F0")

ax2.set_title("CLOP v6.3 Training Summary", fontsize=14, fontweight="bold", pad=20)
fig2.savefig(OUT_DIR / "v63_summary.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved {OUT_DIR / 'v63_summary.png'}")

# ── Figure 3: Proto Acc progression with key milestones ──────
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(epochs, [v*100 for v in proto_acc], "g-", linewidth=2.5, label="Val Proto Acc")
ax3.fill_between(epochs, 0, [v*100 for v in proto_acc], alpha=0.1, color="green")

# Annotate milestones
milestones = [
    (1, "Start: 1.1%"),
    (5, "T hits cap"),
    (10, "7.0%"),
    (20, "7.9%"),
    (32, f"Best: {best_val:.1f}%"),
]
for ep, label in milestones:
    if ep <= n_epochs:
        idx = ep - 1
        ax3.annotate(label, xy=(ep, proto_acc[idx]*100),
                     xytext=(ep+3, proto_acc[idx]*100 + 0.8),
                     fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="black", lw=1),
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

ax3.axhline(y=100/102, color="gray", linestyle="--", alpha=0.5, label="Random (~1%)")
ax3.axvline(x=best_idx+1, color="red", linestyle=":", alpha=0.5, label=f"Best epoch ({best_idx+1})")
ax3.axvline(x=n_epochs, color="purple", linestyle=":", alpha=0.5, label=f"Early stop ({n_epochs})")
ax3.set_xlabel("Epoch", fontsize=12)
ax3.set_ylabel("Prototype Accuracy (%)", fontsize=12)
ax3.set_title("CLOP v6.3 Prototype Accuracy Progression", fontsize=14, fontweight="bold")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)
ax3.set_ylim(bottom=0)

fig3.savefig(OUT_DIR / "v63_proto_acc_progression.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved {OUT_DIR / 'v63_proto_acc_progression.png'}")

plt.close("all")
print("\nDone. All figures saved to results/")
