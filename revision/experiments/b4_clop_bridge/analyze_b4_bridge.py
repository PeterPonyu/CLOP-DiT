"""B4 Phase 4: Compare generation metrics across CLOP ablation variants.

Loads generation_metrics.json from each variant's results directory and
produces a comparative summary + verdict on whether CLOP ablation
conclusions transfer to the full pipeline.

Run after run_b4_evaluate.sh completes.
"""

from __future__ import annotations

import json
import sys
import logging
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VARIANTS = ["abl_baseline", "no_cohesion", "no_cell_noise"]

# Stage-1 proto_acc from ablation suite (for transfer comparison)
STAGE1_PROTO_ACC = {
    "abl_baseline": 0.7617,
    "no_cohesion": 0.8641,
    "no_cell_noise": 0.8623,
}


def main() -> None:
    results = {}
    for v in VARIANTS:
        metrics_path = HERE / f"results_{v}" / "generation_metrics.json"
        if not metrics_path.exists():
            logger.warning(f"  [SKIP] {v}: no metrics at {metrics_path}")
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        results[v] = m
        logger.info(f"Loaded {v}: {len(m.get('per_type', {}))} types evaluated")

    if len(results) < 2:
        logger.error("Need at least 2 variants to compare")
        return

    # Extract headline metrics per variant
    rows = []
    for v in VARIANTS:
        if v not in results:
            continue
        m = results[v]
        overall = m.get("overall", {})
        summary = m.get("summary", {})
        row = {
            "variant": v,
            "stage1_proto_acc": STAGE1_PROTO_ACC.get(v),
            "frechet_distance": overall.get("frechet_distance"),
            "coverage": overall.get("coverage"),
            "mmd": overall.get("mmd"),
            "mean_centroid_cosine": summary.get("mean_centroid_cosine"),
            "min_centroid_cosine": summary.get("min_centroid_cosine"),
            "mean_fd_per_type": summary.get("mean_fd"),
        }
        rows.append(row)

    if not rows:
        logger.error("No valid results")
        return

    # Compute transfer verdict
    # Key question: does the Stage-1 ranking (no_cohesion ≈ no_cell_noise > baseline
    # in proto_acc) match the Stage-2 ranking in generation quality?
    baseline_row = next((r for r in rows if r["variant"] == "abl_baseline"), None)
    transfer_verdicts = []
    if baseline_row:
        for row in rows:
            if row["variant"] == "abl_baseline":
                continue
            s1_delta = (row["stage1_proto_acc"] or 0) - (baseline_row["stage1_proto_acc"] or 0)
            s2_fd_delta = None
            s2_cos_delta = None

            if row["frechet_distance"] is not None and baseline_row["frechet_distance"] is not None:
                s2_fd_delta = row["frechet_distance"] - baseline_row["frechet_distance"]
            if row["mean_centroid_cosine"] is not None and baseline_row["mean_centroid_cosine"] is not None:
                s2_cos_delta = row["mean_centroid_cosine"] - baseline_row["mean_centroid_cosine"]

            # Determine if direction matches
            # Higher proto_acc at Stage-1 should → lower FD and higher centroid cosine at Stage-2
            fd_direction = "matches" if (s2_fd_delta is not None and s2_fd_delta < 0) else \
                          "reverses" if (s2_fd_delta is not None and s2_fd_delta > 0) else "unknown"
            cos_direction = "matches" if (s2_cos_delta is not None and s2_cos_delta > 0) else \
                           "reverses" if (s2_cos_delta is not None and s2_cos_delta < 0) else "unknown"

            # Magnitude check (is the effect practically meaningful?)
            fd_magnitude = abs(s2_fd_delta) if s2_fd_delta is not None else 0
            cos_magnitude = abs(s2_cos_delta) if s2_cos_delta is not None else 0

            if fd_magnitude < 0.01 * abs(baseline_row["frechet_distance"] or 1) and \
               cos_magnitude < 0.005:
                verdict = "DISAPPEARS"
                detail = "Stage-1 effect does not propagate to generation quality"
            elif fd_direction == "matches" and cos_direction == "matches":
                verdict = "CONFIRMED"
                detail = "Stage-1 improvement transfers to better generation"
            elif fd_direction == "reverses" or cos_direction == "reverses":
                verdict = "REVERSED"
                detail = "Stage-1 improvement does not translate — generation quality worsens"
            else:
                verdict = "MIXED"
                detail = "Inconsistent signals across metrics"

            transfer_verdicts.append({
                "variant": row["variant"],
                "stage1_proto_acc_delta": round(s1_delta, 4) if s1_delta else None,
                "stage2_fd_delta": round(s2_fd_delta, 4) if s2_fd_delta is not None else None,
                "stage2_cos_delta": round(s2_cos_delta, 4) if s2_cos_delta is not None else None,
                "fd_direction": fd_direction,
                "cos_direction": cos_direction,
                "verdict": verdict,
                "detail": detail,
            })

    # Overall verdict
    if transfer_verdicts:
        all_verdicts = [tv["verdict"] for tv in transfer_verdicts]
        if all(v == "CONFIRMED" for v in all_verdicts):
            overall_verdict = "FULL_TRANSFER"
            overall_detail = "All Stage-1 ablation improvements transfer to full pipeline."
        elif all(v == "DISAPPEARS" for v in all_verdicts):
            overall_verdict = "NO_TRANSFER"
            overall_detail = "Stage-1 ablation effects vanish at the generation stage. CLOP ablation conclusions are Stage-1-specific."
        elif any(v == "REVERSED" for v in all_verdicts):
            overall_verdict = "PARTIAL_REVERSAL"
            overall_detail = "Some Stage-1 improvements reverse at the generation stage. CLOP ablation study is not a reliable proxy for end-to-end quality."
        else:
            overall_verdict = "MIXED"
            overall_detail = "Mixed signals across variants. Some effects transfer, some don't."
    else:
        overall_verdict = "INSUFFICIENT_DATA"
        overall_detail = "Not enough variants to assess transfer."

    # Save output
    out = {
        "experiment": "b4_clop_bridge",
        "n_variants": len(rows),
        "per_variant": rows,
        "transfer_verdicts": transfer_verdicts,
        "overall_verdict": overall_verdict,
        "overall_detail": overall_detail,
    }
    (HERE / "b4_bridge_comparison.json").write_text(json.dumps(out, indent=2) + "\n")

    # Human-readable preview
    lines = [
        "B4 — CLOP ablation → full pipeline bridge comparison",
        "=" * 60,
        "",
        f"{'variant':<20s} {'proto_acc':>10s} {'FD':>10s} {'coverage':>10s} {'centroid_cos':>12s}",
        "-" * 62,
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<20s} "
            f"{row['stage1_proto_acc']:>10.4f} "
            f"{row['frechet_distance']:>10.4f} "
            f"{row['coverage']:>10.4f} "
            f"{row['mean_centroid_cosine']:>12.4f}"
        )

    lines.append("")
    lines.append("Transfer verdicts:")
    for tv in transfer_verdicts:
        lines.append(
            f"  {tv['variant']}: {tv['verdict']} "
            f"(FD Δ={tv['stage2_fd_delta']}, cos Δ={tv['stage2_cos_delta']}, "
            f"FD {tv['fd_direction']}, cos {tv['cos_direction']})"
        )

    lines.append("")
    lines.append(f"OVERALL: {overall_verdict}")
    lines.append(overall_detail)

    preview = "\n".join(lines) + "\n"
    (HERE / "b4_bridge_comparison_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
