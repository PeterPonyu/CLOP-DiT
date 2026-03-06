#!/usr/bin/env python3
"""Run downstream biology analyses for all registered methods with expression artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.baseline_registry import get_method_specs, load_method_metadata
from src.evaluation.downstream_biology import run_all_downstream
from src.utils.paths import CACHE_DIR, RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream biology for baseline methods")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--caption-json", default=None)
    parser.add_argument("--contrasts-json", default=None)
    parser.add_argument("--include-primary", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir or RESULTS_DIR)
    output_dir = Path(args.output_dir or (results_dir / "downstream"))
    caption_json = Path(args.caption_json or (CACHE_DIR / "text_captions_deduplicated.json"))
    output_dir.mkdir(parents=True, exist_ok=True)

    type_names = {}
    if caption_json.exists():
        with open(caption_json) as f:
            raw = json.load(f)
        for k, v in raw.items():
            name = v.split(" are ")[0] if " are " in v else v[:50]
            type_names[int(k)] = name

    contrasts = None
    if args.contrasts_json:
        with open(args.contrasts_json) as f:
            contrasts = [tuple(item) for item in json.load(f)]

    for spec in get_method_specs(results_dir):
        if spec.root_generated and not args.include_primary:
            continue
        if not spec.capabilities.has_expression:
            continue

        expr_path = spec.expression_path(results_dir)
        expr_labels = spec.expression_labels_path(results_dir)
        gene_names = results_dir / "expression_gene_names.json"
        if not expr_path.exists() or not expr_labels.exists() or not gene_names.exists():
            continue

        metadata = load_method_metadata(spec, results_dir)
        print(f"Running downstream biology for {spec.display_name}...")
        run_all_downstream(
            type_names=type_names,
            output_dir=str(output_dir),
            output_prefix=spec.slug,
            contrasts=contrasts,
            real_expr_path=str(results_dir / "real_expression.npy"),
            gen_expr_path=str(expr_path),
            real_labels_path=str(results_dir / "real_expression_labels.npy"),
            gen_labels_path=str(expr_labels),
            gene_names_path=str(gene_names),
        )
        metadata["downstream_outputs"] = str(output_dir)
        meta_path = spec.metadata_path(results_dir)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
