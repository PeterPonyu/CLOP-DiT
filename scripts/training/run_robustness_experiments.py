#!/usr/bin/env python3
"""Plan or execute robustness experiments for CLOP-DiT.

This script is intentionally orchestration-first: by default it writes a
machine-readable experiment plan so robustness studies are explicit and
reproducible before expensive runs are launched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _parse_csv_ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _subsample_cells_per_type(base: int, frac: float) -> int:
    return max(10, int(round(base * frac)))


def build_plan(
    base_output_dir: Path,
    seeds: list[int],
    subsample_fracs: list[float],
    base_cells_per_type: int,
    dataset_manifest: str,
    prompt_variants_json: str | None,
) -> list[dict]:
    experiments = []

    for seed in seeds:
        exp_id = f"seed_{seed}"
        out_dir = base_output_dir / exp_id
        experiments.append(
            {
                "id": exp_id,
                "category": "seed_stability",
                "seed": seed,
                "command": [
                    "python",
                    "scripts/08_biological_validation.py",
                    "--dataset_manifest",
                    dataset_manifest,
                    "--output_dir",
                    str(out_dir),
                    "--seed",
                    str(seed),
                ],
            }
        )

    for frac in subsample_fracs:
        exp_id = f"subsample_{str(frac).replace('.', 'p')}"
        out_dir = base_output_dir / exp_id
        experiments.append(
            {
                "id": exp_id,
                "category": "subsampling",
                "subsample_fraction": frac,
                "command": [
                    "python",
                    "scripts/08_biological_validation.py",
                    "--dataset_manifest",
                    dataset_manifest,
                    "--output_dir",
                    str(out_dir),
                    "--num_cells_per_type",
                    str(_subsample_cells_per_type(base_cells_per_type, frac)),
                ],
            }
        )

    if prompt_variants_json:
        with open(prompt_variants_json) as f:
            prompt_variants = json.load(f)
        for variant_name, prompt_file in prompt_variants.items():
            exp_id = f"prompt_{variant_name}"
            out_dir = base_output_dir / exp_id
            experiments.append(
                {
                    "id": exp_id,
                    "category": "caption_sensitivity",
                    "prompt_variant": variant_name,
                    "command": [
                        "python",
                        "scripts/08_biological_validation.py",
                        "--dataset_manifest",
                        dataset_manifest,
                        "--output_dir",
                        str(out_dir),
                        "--prompt_file",
                        str(prompt_file),
                    ],
                }
            )

    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run robustness experiments")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", default="results/robustness")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--subsample-fracs", default="1.0,0.5,0.25")
    parser.add_argument("--base-cells-per-type", type=int, default=200)
    parser.add_argument("--prompt-variants-json", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(
        base_output_dir=output_dir,
        seeds=_parse_csv_ints(args.seeds),
        subsample_fracs=_parse_csv_floats(args.subsample_fracs),
        base_cells_per_type=args.base_cells_per_type,
        dataset_manifest=args.dataset_manifest,
        prompt_variants_json=args.prompt_variants_json,
    )

    plan_path = output_dir / "experiment_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    if not args.execute:
        print(f"Wrote robustness plan to {plan_path}")
        return

    for exp in plan:
        print(f"\n== Running {exp['id']} ==")
        exp_dir = output_dir / exp["id"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "console.log"
        with open(log_path, "w") as log_f:
            result = subprocess.run(exp["command"], stdout=log_f, stderr=subprocess.STDOUT)
        exp["return_code"] = result.returncode
        exp["log_path"] = str(log_path)

    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)


if __name__ == "__main__":
    main()
