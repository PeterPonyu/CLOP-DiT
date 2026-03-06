#!/usr/bin/env python3
"""Train an scVI baseline and export benchmark metadata/artifacts.

This script is intentionally conservative: it records a pinned artifact
contract and prepares latent outputs when `scvi-tools` is available, but it
does not force a comparison in the benchmark unless the expected artifacts
exist under `results/baselines/scvi_latent/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _update_manifest(output_dir: Path, metadata: dict) -> None:
    manifest_path = output_dir.parent / "manifest.json"
    manifest = {"artifact_contract_version": 1, "methods": {}}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest["methods"][metadata["slug"]] = metadata
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or document an scVI latent baseline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--expression", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--gene-names", default=None)
    parser.add_argument("--output-dir", default="results/baselines/scvi_latent")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expr_path = Path(args.expression or cfg.get("expression", "results/real_expression.npy"))
    lbl_path = Path(args.labels or cfg.get("labels", "results/real_expression_labels.npy"))
    gene_path = Path(args.gene_names or cfg.get("gene_names", "results/expression_gene_names.json"))
    latent_dim = args.latent_dim or cfg.get("latent_dim", 32)
    epochs = args.epochs or cfg.get("epochs", 200)

    metadata = {
        "slug": "scvi_latent",
        "display_name": "scVI Latent",
        "family": "external_baseline",
        "source": "scvi-tools",
        "artifact_contract_version": 1,
        "expression": str(expr_path),
        "labels": str(lbl_path),
        "gene_names": str(gene_path),
        "latent_dim": latent_dim,
        "epochs": epochs,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "notes": (
            "This baseline is intended for expression-side modeling with scvi-tools. "
            "To make it directly comparable in the current embedding benchmark, the "
            "exported artifacts must be written to results/baselines/scvi_latent/"
            "{embeddings.npy,labels.npy} in the shared benchmark space."
        ),
    }

    try:
        import anndata as ad
        import scvi
    except Exception as exc:
        metadata["status"] = "dependency_missing"
        metadata["dependency_error"] = str(exc)
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        _update_manifest(output_dir, metadata)
        print(
            f"[scvi_baseline] scvi-tools not available ({exc}). "
            "Wrote dependency_missing metadata. Skipping training."
        )
        return

    if args.dry_run:
        metadata["status"] = "dry_run"
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        _update_manifest(output_dir, metadata)
        return

    if not expr_path.exists() or not lbl_path.exists() or not gene_path.exists():
        raise FileNotFoundError(f"Missing required input(s): {expr_path}, {lbl_path}, {gene_path}")

    X = np.load(expr_path).astype(np.float32)
    y = np.load(lbl_path).astype(np.int64)
    with open(gene_path) as f:
        gene_names = json.load(f)

    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs["label"] = y.astype(str)

    scvi.settings.seed = args.seed
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=latent_dim)
    model.train(max_epochs=epochs)

    latent = model.get_latent_representation(adata)
    np.save(output_dir / "embeddings.npy", latent.astype(np.float32))
    np.save(output_dir / "labels.npy", y.astype(np.int64))
    metadata["status"] = "completed"
    metadata["scvi_version"] = getattr(scvi, "__version__", "unknown")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    _update_manifest(output_dir, metadata)


if __name__ == "__main__":
    main()
