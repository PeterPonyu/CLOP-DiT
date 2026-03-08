# visualizer.py — Visualization tools for CLOP-DiT
"""
Visualization utilities for:
    1. CLOP alignment space (text vs cell projections)
    2. DiT generated embeddings vs real
    3. Training curves
    4. Generated expression heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional, List, Dict

# Publication-quality style (Nature/Cell convention)
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


class EmbeddingVisualizer:
    """Visualization suite for CLOP-DiT embeddings.

    Parameters
    ----------
    save_dir : str
        Directory to save figures.
    dpi : int
        Figure resolution.
    """

    def __init__(self, save_dir: str = "figures", dpi: int = 300):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_alignment_space(
        self,
        text_proj: np.ndarray,
        cell_proj: np.ndarray,
        labels: Optional[List[str]] = None,
        filename: str = "alignment_space.png",
        method: str = "umap",
    ):
        """Visualize CLOP alignment space with text and cell projections.

        Parameters
        ----------
        text_proj : (N, D) text projections
        cell_proj : (N, D) cell projections
        labels : list of str, optional
            Sample labels for coloring.
        filename : str
        method : str
            Dimensionality reduction ('umap' or 'tsne').
        """
        combined = np.concatenate([text_proj, cell_proj], axis=0)
        N = text_proj.shape[0]

        if method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)

        coords = reducer.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Text points
        ax.scatter(
            coords[:N, 0], coords[:N, 1],
            c="tab:blue", marker="^", s=50, alpha=0.7, label="Text", edgecolors="white", linewidths=0.5
        )
        # Cell points
        ax.scatter(
            coords[N:, 0], coords[N:, 1],
            c="tab:orange", marker="o", s=30, alpha=0.7, label="Cell", edgecolors="white", linewidths=0.5
        )

        # Draw lines between matched pairs
        for i in range(min(N, 100)):  # Limit lines for readability
            ax.plot(
                [coords[i, 0], coords[N + i, 0]],
                [coords[i, 1], coords[N + i, 1]],
                "k-", alpha=0.1, linewidth=0.5,
            )

        ax.legend(fontsize=12, frameon=True)
        ax.set_title("CLOP Alignment Space", fontsize=14, fontweight="bold")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_generation_comparison(
        self,
        real: np.ndarray,
        generated: np.ndarray,
        labels_real: Optional[np.ndarray] = None,
        labels_gen: Optional[np.ndarray] = None,
        filename: str = "generation_comparison.png",
        method: str = "umap",
    ):
        """Compare real vs generated embeddings in 2D.

        Parameters
        ----------
        real : (N, D) real cell embeddings
        generated : (M, D) generated cell embeddings
        labels_real : optional labels for coloring real cells
        labels_gen : optional labels for coloring generated cells
        filename : str
        method : str
        """
        combined = np.concatenate([real, generated], axis=0)
        N_real = real.shape[0]

        if method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)

        coords = reducer.fit_transform(combined)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Real
        ax = axes[0]
        c = labels_real if labels_real is not None else "tab:blue"
        ax.scatter(coords[:N_real, 0], coords[:N_real, 1], c=c, s=5, alpha=0.5, cmap="tab20")
        ax.set_title("Real Cells", fontsize=14, fontweight="bold")

        # Panel 2: Generated
        ax = axes[1]
        c = labels_gen if labels_gen is not None else "tab:red"
        ax.scatter(coords[N_real:, 0], coords[N_real:, 1], c=c, s=5, alpha=0.5, cmap="tab20")
        ax.set_title("Generated Cells", fontsize=14, fontweight="bold")

        # Panel 3: Overlay
        ax = axes[2]
        ax.scatter(coords[:N_real, 0], coords[:N_real, 1], c="tab:blue", s=5, alpha=0.3, label="Real")
        ax.scatter(coords[N_real:, 0], coords[N_real:, 1], c="tab:red", s=5, alpha=0.3, label="Generated")
        ax.legend(fontsize=11, markerscale=5)
        ax.set_title("Overlay", fontsize=14, fontweight="bold")

        for ax in axes:
            ax.set_xlabel(f"{method.upper()} 1")
            ax.set_ylabel(f"{method.upper()} 2")

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_training_curves(
        self,
        history: Dict,
        filename: str = "training_curves.png",
        title: str = "Training Progress",
    ):
        """Plot training and validation loss curves.

        Parameters
        ----------
        history : dict
            Training history with 'train_loss', 'val_loss', etc.
        filename : str
        title : str
        """
        n_plots = sum(1 for k in history if history[k])
        fig, axes = plt.subplots(1, min(n_plots, 4), figsize=(5 * min(n_plots, 4), 4))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0
        for key, values in history.items():
            if not values or plot_idx >= 4:
                continue
            ax = axes[plot_idx]
            ax.plot(values, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key.replace("_", " ").title())
            ax.set_title(key.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # suptitle removed per revision; title information moved to LaTeX caption
        # fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_dimension_distributions(
        self,
        real: np.ndarray,
        generated: np.ndarray,
        dims: Optional[List[int]] = None,
        filename: str = "dim_distributions.png",
    ):
        """Compare per-dimension distributions between real and generated.

        Parameters
        ----------
        real : (N, D)
        generated : (M, D)
        dims : list of int, optional
            Which dimensions to plot. Default: first 12.
        filename : str
        """
        if dims is None:
            dims = list(range(min(12, real.shape[1])))

        n_cols = 4
        n_rows = (len(dims) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        for i, d in enumerate(dims):
            ax = axes[i]
            ax.hist(real[:, d], bins=50, alpha=0.5, density=True, label="Real", color="tab:blue")
            ax.hist(generated[:, d], bins=50, alpha=0.5, density=True, label="Generated", color="tab:red")
            ax.set_title(f"Dim {d}", fontsize=10)
            ax.legend(fontsize=8)

        for i in range(len(dims), len(axes)):
            axes[i].set_visible(False)

        # suptitle removed per revision; title information moved to LaTeX caption
        # plt.suptitle("Per-Dimension Distribution Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()
