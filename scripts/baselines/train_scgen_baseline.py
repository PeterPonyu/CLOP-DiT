#!/usr/bin/env python3
"""Train a scGen-equivalent baseline using scVI's conditional VAE.

scGen (Lotfollahi et al., Nat Methods 2019) is built on top of a conditional VAE.
Due to API incompatibilities between the scgen package and modern scvi-tools,
we implement scGen's core functionality directly: a conditional scVI model
with cell-type labels as the batch key (enabling label-conditioned generation).

This approach captures scGen's key capability: learning a shared latent space
where cell types are separable, enabling conditional generation by decoding
from per-type latent distributions.

Artifact contract: results/baselines/scgen/{embeddings.npy, labels.npy, metadata.json}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser(description="Train scGen-equivalent baseline")
    parser.add_argument("--expression", default="results/real_expression.npy")
    parser.add_argument("--labels", default="results/real_expression_labels.npy")
    parser.add_argument("--gene-names", default="results/expression_gene_names.json")
    parser.add_argument("--output-dir", default="results/baselines/scgen")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "slug": "scgen",
        "display_name": "scGen",
        "family": "external_baseline",
        "source": "scgen-via-scvi",
        "artifact_contract_version": 1,
        "expression": str(args.expression),
        "labels": str(args.labels),
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "seed": args.seed,
        "notes": (
            "scGen-equivalent: conditional scVI with cell-type labels as batch key. "
            "Captures scGen's core capability (label-conditioned latent space) using "
            "scvi-tools API due to scgen package incompatibility."
        ),
    }

    try:
        import anndata as ad
        import scvi
    except ImportError as exc:
        metadata["status"] = "dependency_missing"
        metadata["dependency_error"] = str(exc)
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        _update_manifest(output_dir, metadata)
        print(f"[scgen_baseline] dependency missing ({exc}). Skipping.")
        return

    expr_path = Path(args.expression)
    lbl_path = Path(args.labels)
    gene_path = Path(args.gene_names)

    if not expr_path.exists() or not lbl_path.exists():
        raise FileNotFoundError(f"Missing: {expr_path}, {lbl_path}")

    print("[scgen_baseline] Loading expression data...")
    X = np.load(expr_path).astype(np.float32)
    y = np.load(lbl_path).astype(np.int64)
    with open(gene_path) as f:
        gene_names = json.load(f)

    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs["cell_type"] = y.astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    n_types = len(np.unique(y))
    print(f"[scgen_baseline] Training conditional scVI (scGen-equivalent) on "
          f"{X.shape[0]} cells, {X.shape[1]} genes, {n_types} types, "
          f"{args.epochs} epochs, latent_dim={args.latent_dim}...")

    scvi.settings.seed = args.seed

    # Use cell_type as categorical covariate for conditional modeling
    scvi.model.SCVI.setup_anndata(
        adata,
        categorical_covariate_keys=["cell_type"],
    )
    model = scvi.model.SCVI(
        adata,
        n_latent=args.latent_dim,
        n_layers=2,
        n_hidden=128,
    )
    model.train(max_epochs=args.epochs, early_stopping=True)

    # Export latent representation
    latent = model.get_latent_representation(adata)
    np.save(output_dir / "embeddings.npy", latent.astype(np.float32))
    np.save(output_dir / "labels.npy", y.astype(np.int64))

    metadata["status"] = "completed"
    metadata["scvi_version"] = getattr(scvi, "__version__", "unknown")
    metadata["latent_dim"] = latent.shape[1]
    metadata["n_cells"] = int(X.shape[0])
    metadata["n_genes"] = int(X.shape[1])
    metadata["n_types"] = n_types
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    _update_manifest(output_dir, metadata)
    print(f"[scgen_baseline] Done. Latent shape: {latent.shape}")
    print(f"[scgen_baseline] Saved to {output_dir}")


if __name__ == "__main__":
    main()
