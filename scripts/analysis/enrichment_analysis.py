#!/usr/bin/env python3
"""GO enrichment analysis of DE concordance between real and generated cells.

For each biologically meaningful contrast, perform Wilcoxon DE on both real and
generated expression, then compare enriched GO terms to quantify pathway-level
concordance beyond individual gene overlap.

Outputs:
    results/enrichment_concordance.json   — per-contrast GO overlap stats
    results/figures/enrichment_concordance.png/.pdf  — summary figure

Dependencies:
    pip install gseapy   (for Enrichr or local GO enrichment)

Usage:
    python scripts/analysis/enrichment_analysis.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualization.style import (
    apply_style, COLORS, FONT_TITLE, FONT_LABEL, FONT_SMALL,
    add_panel_label, save_with_vcd,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Contrasts used in DE concordance (same as Fig 18).
# Labels in the expression arrays are *integer IDs* that map to full captions
# in ``data/cached_latents/text_captions_deduplicated.json``.
CONTRASTS_BY_ID = [
    (0, 6, "CD8+ T cells vs CD4+ T cells"),
    (5, 7, "Macrophages vs Monocytes"),
    (8, 3, "Epithelial cells vs Fibroblasts"),
]

TOP_DE_GENES = 100
LFC_THRESHOLD = 0.25
PVAL_THRESHOLD = 0.05

# GO term databases (used in offline mode or with gseapy)
GO_LIBRARIES = ["GO_Biological_Process_2023", "GO_Molecular_Function_2023"]


# ── helpers ──────────────────────────────────────────────────────────


def _load_expression_data():
    """Load real and generated expression with labels (integer IDs)."""
    real = np.load(RESULTS / "real_expression.npy")
    real_labels = np.load(RESULTS / "real_expression_labels.npy", allow_pickle=True).astype(int)
    gen = np.load(RESULTS / "generated_expression.npy")
    gen_labels = np.load(RESULTS / "generated_expression_labels.npy", allow_pickle=True).astype(int)

    # Gene names
    gene_names_path = RESULTS / "expression_gene_names.json"
    if gene_names_path.exists():
        with open(gene_names_path) as f:
            gene_names = json.load(f)
    else:
        gene_names = [f"Gene_{i}" for i in range(real.shape[1])]

    return real, real_labels, gen, gen_labels, gene_names


def _wilcoxon_de(
    expr: np.ndarray,
    labels: np.ndarray,
    group_a_id: int,
    group_b_id: int,
    gene_names: list[str],
) -> dict:
    """Wilcoxon rank-sum DE between two groups (identified by integer ID).

    Returns dict with gene-level stats: lfc, pval, padj, gene_name.
    """
    mask_a = labels == group_a_id
    mask_b = labels == group_b_id
    X_a = expr[mask_a]
    X_b = expr[mask_b]

    if X_a.shape[0] < 3 or X_b.shape[0] < 3:
        return {"genes": [], "lfc": [], "pval": [], "n_a": int(mask_a.sum()), "n_b": int(mask_b.sum())}

    mean_a = X_a.mean(axis=0)
    mean_b = X_b.mean(axis=0)
    lfc = mean_a - mean_b  # log fold-change (data assumed log-scale)

    pvals = np.ones(expr.shape[1])
    for g in range(expr.shape[1]):
        if np.std(X_a[:, g]) == 0 and np.std(X_b[:, g]) == 0:
            continue
        try:
            _, p = stats.ranksums(X_a[:, g], X_b[:, g])
            pvals[g] = p
        except ValueError:
            pass

    # BH correction
    from numpy import argsort
    n_genes = len(pvals)
    ranked = argsort(pvals)
    padj = np.ones(n_genes)
    for i, idx in enumerate(ranked):
        padj[idx] = pvals[idx] * n_genes / (i + 1)
    padj = np.minimum(padj, 1.0)
    # Enforce monotonicity
    for i in range(n_genes - 2, -1, -1):
        padj[ranked[i]] = min(padj[ranked[i]], padj[ranked[i + 1]] if i + 1 < n_genes else 1.0)

    return {
        "genes": gene_names,
        "lfc": lfc.tolist(),
        "pval": pvals.tolist(),
        "padj": padj.tolist(),
        "n_a": int(mask_a.sum()),
        "n_b": int(mask_b.sum()),
    }


def _get_top_de_genes(de_result: dict, n: int = TOP_DE_GENES) -> list[str]:
    """Get top DE genes by absolute LFC, filtered by p-value."""
    if not de_result["genes"]:
        return []
    lfc = np.abs(de_result["lfc"])
    padj = np.array(de_result["padj"])
    sig = padj < PVAL_THRESHOLD
    if sig.sum() == 0:
        sig = np.ones(len(lfc), dtype=bool)  # fallback if nothing significant
    indices = np.where(sig)[0]
    top_idx = indices[np.argsort(lfc[indices])[-n:]]
    return [de_result["genes"][i] for i in top_idx]


def _enrichment_overlap(
    real_de_genes: list[str],
    gen_de_genes: list[str],
) -> dict:
    """Compute gene-set overlap statistics."""
    real_set = set(real_de_genes)
    gen_set = set(gen_de_genes)
    intersection = real_set & gen_set
    union = real_set | gen_set
    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        "n_real_de": len(real_set),
        "n_gen_de": len(gen_set),
        "n_overlap": len(intersection),
        "jaccard": jaccard,
        "overlap_genes": sorted(intersection),
    }


def _try_go_enrichment(gene_list: list[str], library: str = "GO_Biological_Process_2023"):
    """Try GO enrichment via gseapy (Enrichr). Returns term list or None."""
    try:
        import gseapy as gp
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=library,
            organism="human",
            outdir=None,
            no_plot=True,
        )
        if enr.results is not None and len(enr.results) > 0:
            sig = enr.results[enr.results["Adjusted P-value"] < 0.05]
            return sig["Term"].tolist()[:50]
    except (ImportError, Exception):
        pass
    return None


# ── main ─────────────────────────────────────────────────────────────


def run_enrichment_analysis():
    """Run full enrichment concordance analysis."""
    real, real_labels, gen, gen_labels, gene_names = _load_expression_data()

    results = {}
    for id_a, id_b, contrast_name in CONTRASTS_BY_ID:
        print(f"\nContrast: {contrast_name}")

        # DE on real
        de_real = _wilcoxon_de(real, real_labels, id_a, id_b, gene_names)
        de_gen = _wilcoxon_de(gen, gen_labels, id_a, id_b, gene_names)

        if not de_real["genes"] or not de_gen["genes"]:
            print(f"  Skipped (insufficient cells: real n_a={de_real['n_a']}, gen n_a={de_gen['n_a']})")
            continue

        # Top DE gene overlap
        real_top = _get_top_de_genes(de_real, TOP_DE_GENES)
        gen_top = _get_top_de_genes(de_gen, TOP_DE_GENES)
        overlap = _enrichment_overlap(real_top, gen_top)

        # LFC correlation
        lfc_real = np.array(de_real["lfc"])
        lfc_gen = np.array(de_gen["lfc"])
        mask = np.isfinite(lfc_real) & np.isfinite(lfc_gen)
        pearson_r = float(np.corrcoef(lfc_real[mask], lfc_gen[mask])[0, 1]) if mask.sum() > 2 else 0.0
        spearman_r = float(stats.spearmanr(lfc_real[mask], lfc_gen[mask]).correlation) if mask.sum() > 2 else 0.0

        # Sign agreement
        sign_agree = float(np.mean(np.sign(lfc_real[mask]) == np.sign(lfc_gen[mask]))) if mask.sum() > 0 else 0.0

        # GO enrichment (if gseapy available)
        go_real_terms = _try_go_enrichment(real_top)
        go_gen_terms = _try_go_enrichment(gen_top)
        go_overlap = None
        if go_real_terms and go_gen_terms:
            go_real_set = set(go_real_terms)
            go_gen_set = set(go_gen_terms)
            go_inter = go_real_set & go_gen_set
            go_union = go_real_set | go_gen_set
            go_overlap = {
                "n_real_terms": len(go_real_set),
                "n_gen_terms": len(go_gen_set),
                "n_shared": len(go_inter),
                "jaccard": len(go_inter) / len(go_union) if go_union else 0.0,
                "shared_terms": sorted(go_inter)[:20],
            }

        results[contrast_name] = {
            "de_overlap": overlap,
            "lfc_pearson": pearson_r,
            "lfc_spearman": spearman_r,
            "sign_agreement": sign_agree,
            "go_enrichment_overlap": go_overlap,
        }

        print(f"  DE Jaccard@{TOP_DE_GENES}: {overlap['jaccard']:.3f}")
        print(f"  LFC Pearson: {pearson_r:.3f}, Spearman: {spearman_r:.3f}")
        print(f"  Sign agreement: {sign_agree:.1%}")
        if go_overlap:
            print(f"  GO term overlap: {go_overlap['n_shared']}/{go_overlap['n_real_terms']} terms")

    return results


def make_figure(results: dict):
    """Summary figure for enrichment analysis."""
    apply_style()

    contrasts = list(results.keys())
    n = len(contrasts)
    if n == 0:
        print("No contrasts to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Short labels
    short = [c.split(" vs ")[0][:12] + "\nvs\n" + c.split(" vs ")[1][:12] for c in contrasts]

    # Panel (a): DE gene overlap (Jaccard)
    ax = axes[0]
    jaccards = [results[c]["de_overlap"]["jaccard"] for c in contrasts]
    ax.bar(range(n), jaccards, color=COLORS.get("generated", "#4C72B0"),
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, fontsize=FONT_SMALL - 1)
    ax.set_ylabel(f"Jaccard @ top-{TOP_DE_GENES}", fontsize=FONT_LABEL)
    ax.set_title("DE gene overlap", fontsize=FONT_TITLE)
    ax.set_ylim(0, max(jaccards) * 1.3 if jaccards else 1)
    add_panel_label(ax, "a")

    # Panel (b): LFC correlation
    ax2 = axes[1]
    pearsons = [results[c]["lfc_pearson"] for c in contrasts]
    spearmans = [results[c]["lfc_spearman"] for c in contrasts]
    x = np.arange(n)
    w = 0.35
    ax2.bar(x - w / 2, pearsons, w, label="Pearson", color=COLORS.get("real", "#55A868"),
            edgecolor="black", linewidth=0.6)
    ax2.bar(x + w / 2, spearmans, w, label="Spearman", color=COLORS.get("generated", "#4C72B0"),
            edgecolor="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=FONT_SMALL - 1)
    ax2.set_ylabel("Correlation", fontsize=FONT_LABEL)
    ax2.set_title("LFC concordance", fontsize=FONT_TITLE)
    ax2.legend(fontsize=FONT_SMALL)
    add_panel_label(ax2, "b")

    # Panel (c): Sign agreement
    ax3 = axes[2]
    signs = [results[c]["sign_agreement"] for c in contrasts]
    ax3.bar(range(n), signs, color=COLORS.get("baseline", "#C44E52"),
            edgecolor="black", linewidth=0.6)
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(short, fontsize=FONT_SMALL - 1)
    ax3.set_ylabel("Fraction agreed", fontsize=FONT_LABEL)
    ax3.set_title("LFC sign agreement", fontsize=FONT_TITLE)
    ax3.set_ylim(0, 1.05)
    ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, label="Chance")
    ax3.legend(fontsize=FONT_SMALL)
    add_panel_label(ax3, "c")

    fig.tight_layout()
    out = str(FIG_DIR / "enrichment_concordance")
    save_with_vcd(fig, out)
    plt.close(fig)
    print(f"\nFigure saved: {out}.png / .pdf")


def main():
    print("=" * 60)
    print("Enrichment concordance analysis")
    print("=" * 60)

    results = run_enrichment_analysis()

    # Save JSON
    out_path = RESULTS / "enrichment_concordance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    make_figure(results)
    print("Done.")


if __name__ == "__main__":
    main()
