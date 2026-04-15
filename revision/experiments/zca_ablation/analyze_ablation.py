"""ZCA ablation — analyze CLOP alignment metrics across conditions.

Compares three preprocessing conditions (whiten, center_norm, none)
by loading their training history JSON and extracting best-epoch metrics.

Output:
  - zca_ablation_summary.json (machine-readable)
  - zca_ablation_preview.txt  (human-readable)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

CONDITIONS = ["whiten", "center_norm", "none"]
CKPT_ROOT = REPO_ROOT / "models" / "revision" / "zca_ablation"

# Mapping from history JSON keys to readable metric names
HISTORY_KEYS = {
    "proto_acc": "val_proto_acc",
    "tc_align": "val_text_cell_align",
    "separation": "val_inter_sep",
    "cos_sim": "val_mean_cosine_sim",
    "intra_cohesion": "val_intra_cohesion",
    "val_loss": "val_loss",
    "train_loss": "train_loss",
    "train_proto_acc": "train_proto_acc",
}


def load_history(cond: str) -> dict | None:
    """Load clop_history.json for a condition."""
    hist_path = CKPT_ROOT / cond / "clop_history.json"
    if hist_path.exists():
        return json.loads(hist_path.read_text())
    return None


def extract_best_metrics(hist: dict) -> dict:
    """Extract the best-epoch metrics from training history.

    History uses flat lists: val_proto_acc, val_text_cell_align,
    val_inter_sep, val_mean_cosine_sim, etc. (100 items each).
    Quality score = 0.3*proto + 0.3*tc_align + 0.2*sep + 0.2*cos_sim.
    """
    result = {}

    # Compute quality score per epoch
    proto = np.array(hist.get("val_proto_acc", []))
    tc_align = np.array(hist.get("val_text_cell_align", []))
    sep = np.array(hist.get("val_inter_sep", []))
    cos_sim = np.array(hist.get("val_mean_cosine_sim", []))

    n_epochs = len(proto)
    if n_epochs == 0:
        return result

    quality = 0.3 * proto + 0.3 * tc_align + 0.2 * sep + 0.2 * cos_sim
    best_idx = int(np.argmax(quality))

    result["best_epoch"] = best_idx + 1
    result["val_quality_score"] = round(float(quality[best_idx]), 4)
    result["val_proto_acc"] = round(float(proto[best_idx]), 4)
    result["val_tc_align"] = round(float(tc_align[best_idx]), 4)
    result["val_separation"] = round(float(sep[best_idx]), 4)
    result["val_cos_sim"] = round(float(cos_sim[best_idx]), 4)

    # Additional metrics at best epoch
    for key_name, hist_key in HISTORY_KEYS.items():
        if hist_key in hist and len(hist[hist_key]) > best_idx:
            result[key_name] = round(float(hist[hist_key][best_idx]), 4)

    # Final epoch metrics
    result["final_val_loss"] = round(float(hist["val_loss"][-1]), 4) if "val_loss" in hist else None
    result["final_train_loss"] = round(float(hist["train_loss"][-1]), 4) if "train_loss" in hist else None

    # Convergence trajectory: also report final-epoch quality
    if n_epochs > 1:
        result["final_quality_score"] = round(float(quality[-1]), 4)

    return result


def load_best_checkpoint_metrics(cond: str) -> dict | None:
    """Fallback: load metrics from the clop_best.pth checkpoint."""
    import torch

    ckpt_path = CKPT_ROOT / cond / "clop_best.pth"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", -1)
    result = {"best_epoch": epoch + 1}
    for key in ["proto_acc", "text_cell_align", "inter_sep", "mean_cosine_sim"]:
        if key in metrics:
            result[f"val_{key}"] = round(float(metrics[key]), 4)
    return result


def load_preprocessing_stats(cond: str) -> dict | None:
    """Load preprocessing cosine stats for context."""
    if cond == "center_norm":
        stats_path = (REPO_ROOT / "data" / "cached_latents_ablation_centernorm"
                      / "preprocessing_stats_preprocessed.json")
    elif cond == "whiten":
        stats_path = (REPO_ROOT / "data" / "cached_latents"
                      / "preprocessing_stats_preprocessed.json")
    else:
        return None

    if stats_path.exists():
        return json.loads(stats_path.read_text())
    return None


def main() -> None:
    rows = []

    for cond in CONDITIONS:
        print(f"\n--- Condition: {cond} ---")
        hist = load_history(cond)
        if hist:
            metrics = extract_best_metrics(hist)
            print(f"  Loaded from clop_history.json (best epoch: {metrics.get('best_epoch')})")
        else:
            metrics = load_best_checkpoint_metrics(cond)
            if metrics:
                print(f"  Loaded from clop_best.pth (best epoch: {metrics.get('best_epoch')})")
            else:
                print(f"  [SKIP] No history or checkpoint found for {cond}")
                continue

        pre_stats = load_preprocessing_stats(cond)
        if pre_stats:
            metrics["text_cosine_raw"] = pre_stats.get("text", {}).get("raw_cosine_mean")
            metrics["text_cosine_processed"] = pre_stats.get("text", {}).get("processed_cosine_mean")
            metrics["cell_cosine_raw"] = pre_stats.get("cell", {}).get("raw_cosine_mean")
            metrics["cell_cosine_processed"] = pre_stats.get("cell", {}).get("processed_cosine_mean")

        metrics["condition"] = cond
        rows.append(metrics)

    if not rows:
        print("\nNo results found. Run run_ablation.sh first.")
        return

    # Compute ratios relative to whiten baseline
    whiten_row = next((r for r in rows if r["condition"] == "whiten"), None)
    if whiten_row:
        for r in rows:
            if r["condition"] == "whiten":
                r["quality_ratio_vs_whiten"] = 1.0
            elif "val_quality_score" in r and "val_quality_score" in whiten_row:
                base = whiten_row["val_quality_score"]
                r["quality_ratio_vs_whiten"] = (
                    round(r["val_quality_score"] / base, 4) if base > 0 else None
                )

    # Save JSON summary
    out = {
        "experiment": "ZCA_ablation",
        "conditions": CONDITIONS,
        "metric_definitions": {
            "quality_score": "0.3*proto_acc + 0.3*tc_align + 0.2*separation + 0.2*cos_sim",
            "prototype_accuracy": "fraction of text prototypes whose nearest cell centroid is correct type",
            "text_cell_alignment": "mean cosine between matched text-cell pairs in projected space",
            "separation": "mean distance between non-matched pairs (higher = better)",
            "cosine_similarity": "mean cosine similarity of positive pairs",
        },
        "results": rows,
    }

    (HERE / "zca_ablation_summary.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    # Human-readable preview
    lines = [
        "ZCA Ablation — CLOP alignment metrics across preprocessing conditions",
        "=" * 72,
        "",
    ]

    header = (
        f"{'condition':<14s} {'quality':>8s} {'proto_acc':>10s} "
        f"{'tc_align':>10s} {'sep':>8s} {'cos_sim':>8s} {'best_ep':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        lines.append(
            f"{r['condition']:<14s} "
            f"{r.get('val_quality_score', '—'):>8} "
            f"{r.get('val_proto_acc', '—'):>10} "
            f"{r.get('val_tc_align', '—'):>10} "
            f"{r.get('val_separation', '—'):>8} "
            f"{r.get('val_cos_sim', '—'):>8} "
            f"{r.get('best_epoch', '—'):>8}"
        )

    if whiten_row and len(rows) > 1:
        lines.append("")
        lines.append("Ratios vs whiten (ZCA) baseline:")
        for r in rows:
            ratio = r.get("quality_ratio_vs_whiten")
            if ratio is not None:
                lines.append(f"  {r['condition']:<14s} quality_ratio = {ratio:.3f}")

    # Preprocessing cosine stats
    lines.append("")
    lines.append("Embedding pairwise cosine similarity (pre/post preprocessing):")
    for r in rows:
        if "text_cosine_raw" in r and r["text_cosine_raw"] is not None:
            lines.append(
                f"  {r['condition']:<14s} text: {r['text_cosine_raw']:.4f} -> "
                f"{r['text_cosine_processed']:.4f}   "
                f"cell: {r['cell_cosine_raw']:.4f} -> "
                f"{r['cell_cosine_processed']:.4f}"
            )
        elif r["condition"] == "none":
            lines.append(f"  {'none':<14s} (no preprocessing applied)")

    preview = "\n".join(lines) + "\n"
    (HERE / "zca_ablation_preview.txt").write_text(preview)
    print("\n" + preview)


if __name__ == "__main__":
    main()
