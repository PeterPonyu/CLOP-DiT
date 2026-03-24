#!/usr/bin/env python3
# rare_cell_augmentation.py — Data augmentation utility study for rare cell types
"""
Demonstrate CLOP-DiT's utility as a data augmentation engine for rare cell types.

Study design:
  1. Identify rare cell types (< threshold fraction) in reference data.
  2. Extreme-downsample rare types to simulate scarcity.
  3. Generate synthetic cells at increasing augmentation ratios (1x, 5x, 10x).
  4. Train logistic regression classifiers with/without augmentation.
  5. Measure downstream detection F1 per rare type.

Produces:
  - results/rare_cell_augmentation/augmentation_results.json
  - results/rare_cell_augmentation/fig_augmentation_utility.pdf

Usage (CLI):
    python -m src.experiments.rare_cell_augmentation \
        --reference_h5ad data/processed_h5ad/reference.h5ad \
        --output_dir results/rare_cell_augmentation \
        --threshold 0.01 \
        --num_cells 50 \
        --ratios 1 5 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import anndata as ad
import scanpy as sc

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Project path setup ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.visualization.style import COLORS as VIZ_COLORS, apply_style, save_with_vcd

# ── Load CLOPDiTInference from numbered script via importlib ───────────
_INFERENCE_SCRIPT = _PROJECT_ROOT / "scripts" / "inference" / "05_inference.py"
_spec = importlib.util.spec_from_file_location("inference", str(_INFERENCE_SCRIPT))
_inference_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inference_mod)
CLOPDiTInference = _inference_mod.CLOPDiTInference

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════
# Default prompts for common rare cell types
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_RARE_TYPE_PROMPTS: dict[str, str] = {
    "Mast cells": (
        "Mast cells in the tumor microenvironment. Tissue-resident mast cells "
        "with granule-associated gene expression including TPSAB1 and KIT."
    ),
    "Plasmacytoid dendritic cells": (
        "Plasmacytoid dendritic cells from human tissue. pDCs expressing "
        "interferon response genes including IRF7, LILRA4, and CLEC4C."
    ),
    "Plasma cells": (
        "Plasma cells producing immunoglobulins. Terminally differentiated "
        "B cells with high expression of JCHAIN, MZB1, and XBP1."
    ),
    "Basophils": (
        "Basophils from human peripheral blood or tissue. Rare granulocytes "
        "expressing GATA2, CPA3, and HDC."
    ),
    "Megakaryocytes": (
        "Megakaryocytes from human bone marrow. Large polyploid cells "
        "expressing PF4, PPBP, and GP9."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _to_dense(x) -> np.ndarray:
    """Convert sparse matrix to dense float64 array."""
    import scipy.sparse as sp
    if sp.issparse(x):
        return np.asarray(x.toarray(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _get_embeddings(adata: ad.AnnData, embedding_key: str = "X_clop_dit") -> np.ndarray:
    """Extract embeddings from AnnData, falling back to .X if key not in .obsm.

    Parameters
    ----------
    adata : AnnData
        Input data.
    embedding_key : str
        Key in .obsm to try first.

    Returns
    -------
    (N, D) array of embeddings.
    """
    if embedding_key in adata.obsm:
        return np.asarray(adata.obsm[embedding_key])

    # Fall back to PCA of expression
    logger.info(
        "Embedding key '%s' not found; computing PCA(50) from .X for classification.",
        embedding_key,
    )
    from sklearn.decomposition import PCA

    X = _to_dense(adata.X)
    # Replace NaN/Inf that can arise from scGPT decode or outer-join padding
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Drop zero-variance columns to avoid NaN from PCA standardisation
    col_var = np.var(X, axis=0)
    keep = col_var > 1e-12
    if keep.sum() < X.shape[1]:
        logger.info(
            "Dropping %d zero-variance genes before PCA (%d → %d).",
            X.shape[1] - int(keep.sum()), X.shape[1], int(keep.sum()),
        )
        X = X[:, keep]
    n_components = min(50, X.shape[0], X.shape[1])
    if n_components == 0:
        return np.zeros((len(adata), 1), dtype=np.float64)
    features = PCA(n_components=n_components, random_state=42).fit_transform(X)
    # Safety: ensure no residual NaN after PCA
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


# ═══════════════════════════════════════════════════════════════════════
# RareCellAugmenter
# ═══════════════════════════════════════════════════════════════════════

class RareCellAugmenter:
    """Data augmentation study for rare cell types using CLOP-DiT generation.

    Parameters
    ----------
    cell_type_key : str
        Column in adata.obs containing cell type labels.
    embedding_key : str
        Key in adata.obsm for cell embeddings.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cell_type_key: str = "cell_type",
        embedding_key: str = "X_clop_dit",
        seed: int = 42,
    ):
        self.cell_type_key = cell_type_key
        self.embedding_key = embedding_key
        self.seed = seed

    # ── Step 1: Identify rare types ──────────────────────────────────

    def identify_rare_types(
        self,
        adata: ad.AnnData,
        threshold_fraction: float = 0.01,
    ) -> list[str]:
        """Find cell types with fewer than threshold_fraction of total cells.

        Parameters
        ----------
        adata : AnnData
            Reference dataset.
        threshold_fraction : float
            Cell types below this fraction of total cells are considered rare.

        Returns
        -------
        List of rare cell type names, sorted by count (ascending).
        """
        type_counts = adata.obs[self.cell_type_key].value_counts()
        total = len(adata)
        threshold = total * threshold_fraction

        rare_types = type_counts[type_counts < threshold].index.tolist()
        rare_types.sort(key=lambda t: type_counts[t])

        logger.info(
            "Identified %d rare types (< %.1f%% of %d cells): %s",
            len(rare_types),
            threshold_fraction * 100,
            total,
            rare_types,
        )
        return rare_types

    # ── Step 2: Downsample ───────────────────────────────────────────

    def downsample_rare_types(
        self,
        adata: ad.AnnData,
        rare_types: list[str],
        target_count: int = 10,
    ) -> ad.AnnData:
        """Extreme-downsample rare types to simulate data scarcity.

        Non-rare types are kept intact. For each rare type, randomly sample
        min(target_count, actual_count) cells.

        Parameters
        ----------
        adata : AnnData
            Full reference dataset.
        rare_types : list of str
            Cell types to downsample.
        target_count : int
            Target number of cells per rare type.

        Returns
        -------
        AnnData with downsampled rare types.
        """
        rng = np.random.default_rng(self.seed)

        keep_indices = []

        for ct in adata.obs[self.cell_type_key].unique():
            mask = adata.obs[self.cell_type_key] == ct
            indices = np.where(mask.values)[0]

            if ct in rare_types:
                n_keep = min(target_count, len(indices))
                chosen = rng.choice(indices, size=n_keep, replace=False)
                keep_indices.extend(chosen)
                logger.info(
                    "  Downsampled '%s': %d -> %d cells", ct, len(indices), n_keep,
                )
            else:
                keep_indices.extend(indices)

        keep_indices = sorted(keep_indices)
        downsampled = adata[keep_indices].copy()

        logger.info(
            "Downsampled dataset: %d -> %d cells", len(adata), len(downsampled),
        )
        return downsampled

    # ── Step 3: Generate augmentation ────────────────────────────────

    def generate_augmentation(
        self,
        rare_types: list[str],
        reference_adata: ad.AnnData,
        pipeline: CLOPDiTInference,
        ratios: list[int] | None = None,
        base_count: int = 10,
        prompt_templates: dict[str, str] | None = None,
    ) -> dict[int, ad.AnnData]:
        """Generate synthetic cells for rare types at multiple augmentation ratios.

        Parameters
        ----------
        rare_types : list of str
            Cell types to augment.
        reference_adata : AnnData
            Reference for scGPT gene vocabulary.
        pipeline : CLOPDiTInference
            Loaded inference pipeline.
        ratios : list of int
            Augmentation ratios (multiples of base_count to generate).
        base_count : int
            Base number of cells (the downsampled count).
        prompt_templates : dict mapping cell type -> prompt text, optional.
            Falls back to DEFAULT_RARE_TYPE_PROMPTS or a generic template.

        Returns
        -------
        dict mapping ratio -> AnnData of generated cells (all rare types
        concatenated, with cell_type in .obs).
        """
        if ratios is None:
            ratios = [1, 5, 10]
        if prompt_templates is None:
            prompt_templates = {}

        results: dict[int, ad.AnnData] = {}

        for ratio in ratios:
            adatas_per_type = []

            for ct in rare_types:
                num_gen = base_count * ratio

                # Build prompt
                if ct in prompt_templates:
                    prompt = prompt_templates[ct]
                elif ct in DEFAULT_RARE_TYPE_PROMPTS:
                    prompt = DEFAULT_RARE_TYPE_PROMPTS[ct]
                else:
                    prompt = (
                        f"{ct} cells from human tissue. {ct} with characteristic "
                        f"gene expression markers and transcriptomic signature."
                    )

                logger.info(
                    "  Generating %dx augmentation for '%s': %d cells",
                    ratio, ct, num_gen,
                )

                gen_adata = pipeline.generate_adata(
                    prompt=prompt,
                    num_cells=num_gen,
                    decode_expression=True,
                    reference_adata=reference_adata,
                )
                gen_adata.obs[self.cell_type_key] = ct
                gen_adata.obs["source"] = "synthetic"
                adatas_per_type.append(gen_adata)

            if adatas_per_type:
                combined = ad.concat(adatas_per_type, join="outer")
                combined.obs_names_make_unique()
                results[ratio] = combined
                logger.info(
                    "  Ratio %dx: generated %d total synthetic cells.",
                    ratio, len(combined),
                )

        return results

    # ── Step 4: Train classifier ─────────────────────────────────────

    def train_classifier(
        self,
        train_adata: ad.AnnData,
        test_adata: ad.AnnData,
    ) -> dict:
        """Train logistic regression on cell type labels and evaluate on test set.

        Uses cell embeddings (from .obsm or PCA fallback) as features.

        Parameters
        ----------
        train_adata : AnnData
            Training data with cell type labels.
        test_adata : AnnData
            Test data with cell type labels.

        Returns
        -------
        dict with "overall_f1_macro", "overall_f1_weighted", "per_type_f1",
        and "classification_report".
        """
        # Extract features — always use PCA on expression to keep train/test
        # in the same feature space.  We fit PCA on train and transform test.
        from sklearn.decomposition import PCA

        X_train_raw = _to_dense(train_adata.X)
        X_test_raw = _to_dense(test_adata.X)

        # Align columns: intersect var_names (gene sets may differ after concat)
        train_vars = list(train_adata.var_names)
        test_vars = list(test_adata.var_names)
        common = sorted(set(train_vars) & set(test_vars))
        if len(common) < len(train_vars) or len(common) < len(test_vars):
            logger.info(
                "Aligning gene sets: train=%d, test=%d, common=%d",
                len(train_vars), len(test_vars), len(common),
            )
            tr_idx = [train_vars.index(g) for g in common]
            te_idx = [test_vars.index(g) for g in common]
            X_train_raw = X_train_raw[:, tr_idx]
            X_test_raw = X_test_raw[:, te_idx]

        X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Drop zero-variance columns (based on train)
        col_var = np.var(X_train_raw, axis=0)
        keep = col_var > 1e-12
        X_train_raw = X_train_raw[:, keep]
        X_test_raw = X_test_raw[:, keep]

        n_components = min(50, X_train_raw.shape[0], X_train_raw.shape[1])
        pca = PCA(n_components=n_components, random_state=self.seed)
        X_train = pca.fit_transform(X_train_raw)
        X_test = pca.transform(X_test_raw)

        # Final safety net
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        y_train = train_adata.obs[self.cell_type_key].values
        y_test = test_adata.obs[self.cell_type_key].values

        # Encode labels
        le = LabelEncoder()
        le.fit(np.concatenate([y_train, y_test]))
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        # Train logistic regression
        clf = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=self.seed,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train_enc)

        # Predict
        y_pred = clf.predict(X_test)

        # Compute metrics
        f1_macro = float(f1_score(y_test_enc, y_pred, average="macro", zero_division=0))
        f1_weighted = float(
            f1_score(y_test_enc, y_pred, average="weighted", zero_division=0),
        )

        # Per-type F1
        all_labels = le.classes_
        f1_per_class = f1_score(
            y_test_enc, y_pred, average=None, labels=range(len(all_labels)), zero_division=0,
        )
        per_type_f1 = {str(ct): float(f1_per_class[i]) for i, ct in enumerate(all_labels)}

        report = classification_report(
            y_test_enc, y_pred, target_names=[str(c) for c in all_labels], zero_division=0,
        )

        return {
            "overall_f1_macro": f1_macro,
            "overall_f1_weighted": f1_weighted,
            "per_type_f1": per_type_f1,
            "classification_report": report,
        }

    # ── Step 5: Evaluate augmentation ────────────────────────────────

    def evaluate_augmentation(
        self,
        real_adata: ad.AnnData,
        augmented_adatas: dict[int, ad.AnnData],
        test_adata: ad.AnnData,
    ) -> dict:
        """Compare classifier performance with/without augmentation.

        Parameters
        ----------
        real_adata : AnnData
            Downsampled real training data.
        augmented_adatas : dict
            Mapping ratio -> AnnData of synthetic cells.
        test_adata : AnnData
            Held-out test data.

        Returns
        -------
        dict with "baseline" (no augmentation) and per-ratio results.
        """
        results: Dict[str, Any] = {}

        # Baseline: real only
        logger.info("Training baseline classifier (real data only)...")
        real_adata_train = real_adata.copy()
        real_adata_train.obs["source"] = "real"
        baseline = self.train_classifier(real_adata_train, test_adata)
        results["baseline"] = baseline
        logger.info("  Baseline F1 (macro): %.4f", baseline["overall_f1_macro"])

        # Augmented: real + synthetic at each ratio
        for ratio, syn_adata in sorted(augmented_adatas.items()):
            logger.info("Training augmented classifier (ratio=%dx)...", ratio)

            # Combine real + synthetic (inner join to keep only shared genes)
            combined = ad.concat(
                [real_adata_train, syn_adata], join="inner",
            )
            combined.obs_names_make_unique()
            # Safety: fill any residual NaN
            combined.X = np.nan_to_num(
                _to_dense(combined.X), nan=0.0, posinf=0.0, neginf=0.0,
            )

            aug_result = self.train_classifier(combined, test_adata)
            results[f"ratio_{ratio}x"] = aug_result
            logger.info(
                "  Ratio %dx F1 (macro): %.4f", ratio, aug_result["overall_f1_macro"],
            )

        return results

    # ── Step 6: End-to-end study ─────────────────────────────────────

    def run_full_study(
        self,
        reference_h5ad: str | Path,
        output_dir: str | Path,
        pipeline: CLOPDiTInference,
        threshold_fraction: float = 0.01,
        target_downsample: int = 10,
        ratios: list[int] | None = None,
        test_fraction: float = 0.2,
        prompt_templates: dict[str, str] | None = None,
    ) -> dict:
        """Run the complete rare cell augmentation study end-to-end.

        Parameters
        ----------
        reference_h5ad : str or Path
            Path to reference AnnData.
        output_dir : str or Path
            Output directory.
        pipeline : CLOPDiTInference
            Loaded inference pipeline.
        threshold_fraction : float
            Fraction below which types are rare.
        target_downsample : int
            Target count for downsampled rare types.
        ratios : list of int
            Augmentation ratios.
        test_fraction : float
            Fraction of data held out for testing.
        prompt_templates : dict, optional
            Custom prompts per cell type.

        Returns
        -------
        dict with all study results.
        """
        if ratios is None:
            ratios = [1, 5, 10]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load reference
        logger.info("Loading reference data: %s", reference_h5ad)
        adata = sc.read_h5ad(reference_h5ad)
        logger.info("Reference: %d cells, %d genes", adata.n_obs, adata.n_vars)

        # If cell_type column is missing, try to populate from subcluster metadata
        if self.cell_type_key not in adata.obs.columns:
            meta_path = Path(reference_h5ad).parent / "subcluster_metadata.json"
            dataset_key = Path(reference_h5ad).stem.replace("_processed", "")
            if meta_path.exists():
                import json as _json
                _meta = _json.load(open(meta_path))
                if dataset_key in _meta and "clusters" in _meta[dataset_key]:
                    ct_map = ["Unknown"] * adata.n_obs
                    for cid, info in _meta[dataset_key]["clusters"].items():
                        for idx in info.get("cell_indices", []):
                            if idx < adata.n_obs:
                                ct_map[idx] = info["cell_type"]
                    adata.obs[self.cell_type_key] = ct_map
                    adata.obs[self.cell_type_key] = adata.obs[self.cell_type_key].astype("category")
                    logger.info("Populated '%s' from subcluster metadata: %d types",
                                self.cell_type_key, adata.obs[self.cell_type_key].nunique())
                else:
                    raise KeyError(f"'{self.cell_type_key}' not in adata.obs and dataset_key "
                                   f"'{dataset_key}' not found in subcluster metadata")
            else:
                raise KeyError(f"'{self.cell_type_key}' not found in adata.obs and no "
                               f"subcluster metadata at {meta_path}")

        # Step 1: Identify rare types
        rare_types = self.identify_rare_types(adata, threshold_fraction)

        if not rare_types:
            logger.warning(
                "No rare types found at threshold %.2f%%. "
                "Consider increasing --threshold.",
                threshold_fraction * 100,
            )
            return {"rare_types": [], "error": "No rare types found."}

        # Split train/test (stratified)
        rng = np.random.default_rng(self.seed)
        type_col = self.cell_type_key
        train_idx, test_idx = train_test_split(
            np.arange(len(adata)),
            test_size=test_fraction,
            stratify=adata.obs[type_col].values,
            random_state=self.seed,
        )
        train_adata = adata[train_idx].copy()
        test_adata = adata[test_idx].copy()
        logger.info(
            "Train/test split: %d / %d cells", len(train_adata), len(test_adata),
        )

        # Step 2: Downsample rare types in training set
        downsampled = self.downsample_rare_types(train_adata, rare_types, target_downsample)

        # Step 3: Generate augmentation
        augmented_adatas = self.generate_augmentation(
            rare_types=rare_types,
            reference_adata=adata,
            pipeline=pipeline,
            ratios=ratios,
            base_count=target_downsample,
            prompt_templates=prompt_templates,
        )

        # Step 4-5: Evaluate
        eval_results = self.evaluate_augmentation(
            real_adata=downsampled,
            augmented_adatas=augmented_adatas,
            test_adata=test_adata,
        )

        # Assemble full results
        study_results = {
            "rare_types": rare_types,
            "threshold_fraction": threshold_fraction,
            "target_downsample": target_downsample,
            "ratios": ratios,
            "num_train": len(train_adata),
            "num_test": len(test_adata),
            "num_downsampled": len(downsampled),
            "evaluation": eval_results,
        }

        # Step 6: Report
        self.generate_report(study_results, output_dir)

        return study_results

    # ── Reporting ────────────────────────────────────────────────────

    def generate_report(
        self,
        results: dict,
        output_dir: str | Path,
    ) -> Path:
        """Save JSON results and generate the augmentation utility figure.

        Produces fig_augmentation_utility.pdf: line plot with x=augmentation
        ratio, y=detection F1, one line per rare cell type.

        Parameters
        ----------
        results : dict
            Study results from run_full_study or evaluate_augmentation.
        output_dir : str or Path
            Output directory.

        Returns
        -------
        Path to the saved figure.
        """
        apply_style()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Save JSON ────────────────────────────────────────────────
        json_path = output_dir / "augmentation_results.json"

        # Strip classification_report strings for cleaner JSON
        serialisable = json.loads(json.dumps(results, default=str))
        with open(json_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Saved augmentation results to %s", json_path)

        # ── Extract per-type F1 across ratios ────────────────────────
        eval_data = results.get("evaluation", results)
        rare_types = results.get("rare_types", [])

        # Collect data points: {cell_type: [(ratio_label, f1), ...]}
        type_f1_curves: Dict[str, list] = {ct: [] for ct in rare_types}

        # Baseline (ratio=0)
        baseline = eval_data.get("baseline", {})
        baseline_per_type = baseline.get("per_type_f1", {})
        for ct in rare_types:
            f1_val = baseline_per_type.get(ct, 0.0)
            type_f1_curves[ct].append((0, f1_val))

        # Each augmentation ratio
        ratios = results.get("ratios", [1, 5, 10])
        for ratio in sorted(ratios):
            key = f"ratio_{ratio}x"
            ratio_result = eval_data.get(key, {})
            per_type = ratio_result.get("per_type_f1", {})
            for ct in rare_types:
                f1_val = per_type.get(ct, 0.0)
                type_f1_curves[ct].append((ratio, f1_val))

        if not rare_types or not type_f1_curves:
            logger.warning("No rare type data to plot. Skipping figure.")
            return json_path

        # ── Plot ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5.5, 3.8))

        # Colour cycle from VIZ_COLORS supplemented by a tab10 palette
        base_colors = [
            VIZ_COLORS["real"],
            VIZ_COLORS["generated"],
            VIZ_COLORS["good"],
            VIZ_COLORS["warn"],
            VIZ_COLORS["accent"],
            VIZ_COLORS["neutral"],
            VIZ_COLORS["baseline_gauss"],
            VIZ_COLORS["baseline_shuffle"],
        ]

        for i, ct in enumerate(rare_types):
            points = sorted(type_f1_curves[ct], key=lambda p: p[0])
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]

            color = base_colors[i % len(base_colors)]
            ax.plot(
                x_vals, y_vals,
                marker="o", markersize=5, linewidth=1.5,
                color=color, label=ct,
            )

        ax.set_xlabel("Augmentation Ratio")
        ax.set_ylabel("Detection F1 Score")
        ax.set_title("Rare Cell Type Augmentation Utility", fontsize=11, fontweight="bold")

        # X-axis: show 0 (baseline) plus ratios
        all_x = sorted({0} | set(ratios))
        ax.set_xticks(all_x)
        x_labels = ["Baseline\n(0x)"] + [f"{r}x" for r in all_x if r > 0]
        ax.set_xticklabels(x_labels)

        ax.set_ylim(-0.05, 1.05)
        ax.legend(
            loc="best", fontsize=8, frameon=True, framealpha=0.9,
            edgecolor="0.8",
        )

        fig_path = output_dir / "fig_augmentation_utility.pdf"
        save_with_vcd(fig, fig_path, close=True)
        logger.info("Saved augmentation utility figure to %s", fig_path)

        return fig_path


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Command-line interface for rare cell augmentation study."""
    parser = argparse.ArgumentParser(
        description="CLOP-DiT Rare Cell Type Augmentation Study",
    )
    parser.add_argument(
        "--reference_h5ad",
        type=str,
        required=True,
        help="Path to reference AnnData (.h5ad) file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/rare_cell_augmentation",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Fraction threshold for rare type identification (default: 0.01).",
    )
    parser.add_argument(
        "--num_cells",
        type=int,
        default=10,
        help="Base number of cells per rare type after downsampling.",
    )
    parser.add_argument(
        "--ratios",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Augmentation ratios (multiples of base count).",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing.",
    )
    parser.add_argument("--cell_type_key", type=str, default="cell_type")
    parser.add_argument("--embedding_key", type=str, default="X_clop_dit")
    parser.add_argument(
        "--dit_checkpoint",
        type=str,
        default="models/checkpoints/dit_best.pth",
    )
    parser.add_argument(
        "--clop_checkpoint",
        type=str,
        default="models/checkpoints/clop_best.pth",
    )
    parser.add_argument(
        "--scgpt_model_dir",
        type=str,
        default="models/scgpt_human",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting rare cell augmentation study.")

    # Initialise pipeline
    pipeline = CLOPDiTInference(
        dit_checkpoint=args.dit_checkpoint,
        clop_checkpoint=args.clop_checkpoint,
        scgpt_model_dir=args.scgpt_model_dir,
        device=args.device,
    )

    # Initialise augmenter
    augmenter = RareCellAugmenter(
        cell_type_key=args.cell_type_key,
        embedding_key=args.embedding_key,
        seed=args.seed,
    )

    # Run study
    results = augmenter.run_full_study(
        reference_h5ad=args.reference_h5ad,
        output_dir=args.output_dir,
        pipeline=pipeline,
        threshold_fraction=args.threshold,
        target_downsample=args.num_cells,
        ratios=args.ratios,
        test_fraction=args.test_fraction,
    )

    logger.info("Rare cell augmentation study complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
