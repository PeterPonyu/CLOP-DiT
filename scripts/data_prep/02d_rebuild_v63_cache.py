#!/usr/bin/env python3
# 02d_rebuild_v63_cache.py — Rebuild cache for v6.3 (template anchor + enriched augmentation)
"""
CLOP-DiT v6.3: Fix text anchor drift from v6.2.

Problem (v6.2):
    Enriched texts shifted alignment anchor from cell-type features to
    dataset-specific features (logFC values, dataset-specific pathways,
    Source: context). Same cell type across datasets got very different
    BiomedBERT embeddings → cross-dataset generalization destroyed.
    val_acc: 3.6% (v5.2) → 0.8% (v6.2)

Fix (v6.3):
    1. PRIMARY text: Clean template (organism + tissue + cell type + markers)
       - Strip "Source: ..." suffix (dataset-specific context)
       - Strip "(Leiden cluster N, n=X cells, res=Y)" (arbitrary numbering)
       - Keep: organism, tissue, disease context, cell type, canonical markers
    2. VARIANTS: Enriched texts used ONLY as augmentation (30% sampling)
       - Strip "Source: ..." from variants too
       - Keep pathway/GO/marker detail as augmentation noise
    3. Temperature fix (done in config): max_temperature 100→20

Usage:
    python scripts/02d_rebuild_v63_cache.py
    python scripts/02d_rebuild_v63_cache.py --skip_variants
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Text Cleaning
# ═══════════════════════════════════════════════════════════════

def clean_template_text(text: str) -> str:
    """Clean a template text to be cell-type centric.

    Removes:
        1. "(Leiden cluster N, n=X cells, res=Y.YY)" — dataset-specific
        2. "Source: ..." suffix — dataset-specific context

    Keeps:
        - Organism + tissue + disease context
        - Cell type name
        - Canonical marker gene names
    """
    # 1. Remove "(Leiden cluster N, n=X cells, res=Y.YY)"
    text = re.sub(
        r'\s*\(Leiden cluster \d+,\s*n=\d+ cells,\s*res=[\d.]+\)',
        '',
        text
    )

    # 2. Remove ". Source: ..." onwards
    source_idx = text.find('. Source:')
    if source_idx >= 0:
        text = text[:source_idx] + '.'
    else:
        # Also try without the leading period
        source_idx = text.find('Source:')
        if source_idx > 0 and text[source_idx - 1] in ' .':
            text = text[:source_idx].rstrip('. ') + '.'

    # 3. Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_variant_text(text: str) -> str:
    """Clean an enriched variant text.

    Removes:
        - "Source: ..." suffix
        - Keeps everything else (pathway/GO/marker details as augmentation noise)
    """
    # Remove "Source: ..." onwards — find last occurrence
    # Enriched texts have "Source:" at the end
    source_idx = text.rfind('Source:')
    if source_idx >= 0:
        text = text[:source_idx].rstrip('. ') + '.'

    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════
#  Main rebuild
# ═══════════════════════════════════════════════════════════════

def rebuild_primary_texts(cache_dir: str, device: str = "cuda"):
    """Rebuild primary text embeddings from cleaned template texts.

    Steps:
        1. Load template texts from subcluster_metadata.json
        2. Clean them (strip Source:, Leiden cluster info)
        3. Map each group_id → cleaned template text
        4. Encode through BiomedBERT
        5. Save new text_embeddings_unique.npy + text_strings.json
        6. Rebuild full text_embeddings.npy from unique + group_ids
    """
    cache_dir = Path(cache_dir)

    from src.data_pipeline.cache_builder import LatentCacheBuilder

    # ── Load metadata ──
    template_meta_path = cache_dir.parent / "processed_h5ad" / "subcluster_metadata.json"
    enriched_meta_path = cache_dir.parent / "processed_h5ad" / "subcluster_metadata_enriched.json"
    metadata_path = cache_dir / "metadata.json"

    with open(template_meta_path) as f:
        template_meta = json.load(f)
    with open(metadata_path) as f:
        id_to_text = json.load(f)

    # ── Load existing dedup structure ──
    text_group_ids = np.load(cache_dir / "text_group_ids.npy")
    sample_ids = np.load(cache_dir / "sample_ids.npy")
    n_cells = len(text_group_ids)
    n_unique = int(text_group_ids.max()) + 1

    logger.info(f"Cells: {n_cells}, Unique groups: {n_unique}, Datasets: {len(id_to_text)}")

    # ── Build cell → template text mapping ──
    # Same logic as 02c but using template_meta
    all_cell_texts = [""] * n_cells

    for sid_str, ds_text in id_to_text.items():
        sid = int(sid_str)
        mask = sample_ids == sid
        cell_indices_global = np.where(mask)[0]

        if len(cell_indices_global) == 0:
            continue

        # Find matching dataset in template metadata
        ds_match = None
        for ds_id, ds_info in template_meta.items():
            if ds_info["dataset_text"][:80] == ds_text[:80]:
                ds_match = (ds_id, ds_info)
                break

        if ds_match is None:
            # Fallback: clean the dataset-level text
            clean = clean_template_text(ds_text)
            for gi in cell_indices_global:
                all_cell_texts[gi] = clean
            continue

        ds_id, ds_info = ds_match

        # Default: dataset-level text (for orphan cells)
        clean_ds = clean_template_text(ds_info["dataset_text"])
        for gi in cell_indices_global:
            all_cell_texts[gi] = clean_ds

        # Override with cluster-level texts
        start_idx = int(cell_indices_global[0])
        for cid, cinfo in ds_info.get("clusters", {}).items():
            cluster_text = cinfo.get("text", ds_info["dataset_text"])
            clean_cluster = clean_template_text(cluster_text)

            for local_idx in cinfo.get("cell_indices", []):
                global_idx = start_idx + local_idx
                if global_idx < n_cells and sample_ids[global_idx] == sid:
                    all_cell_texts[global_idx] = clean_cluster

    # ── Map group_id → text string ──
    # For each group_id, pick the text of the first cell in that group
    new_text_strings = {}
    for gid in range(n_unique):
        # Find first cell with this group_id
        cell_idx = np.where(text_group_ids == gid)[0]
        if len(cell_idx) > 0:
            new_text_strings[str(gid)] = all_cell_texts[cell_idx[0]]
        else:
            logger.warning(f"Group {gid} has no cells!")
            new_text_strings[str(gid)] = ""

    # Validate: check for empty texts
    n_empty = sum(1 for v in new_text_strings.values() if not v.strip())
    logger.info(f"Template texts: {len(new_text_strings)} groups, {n_empty} empty")

    # Show some examples
    for gid_str in list(new_text_strings.keys())[:3]:
        logger.info(f"  Group {gid_str}: {new_text_strings[gid_str][:120]}...")

    # Statistics
    lengths = [len(v) for v in new_text_strings.values()]
    logger.info(f"  Text length: mean={np.mean(lengths):.0f}, "
                f"min={np.min(lengths)}, max={np.max(lengths)}")

    # ── Encode through BiomedBERT ──
    texts_ordered = [new_text_strings[str(gid)] for gid in range(n_unique)]
    builder = LatentCacheBuilder(cache_dir=str(cache_dir), device=device)

    logger.info(f"Encoding {n_unique} cleaned template texts through BiomedBERT...")
    new_emb = builder.encode_texts(texts_ordered)
    logger.info(f"  Encoded shape: {new_emb.shape}")

    # ── Save ──
    np.save(cache_dir / "text_embeddings_unique.npy", new_emb)
    logger.info(f"  Saved text_embeddings_unique.npy: {new_emb.shape}")

    # Rebuild full duplicated embeddings (needed for ZCA whitening)
    text_emb_full = new_emb[text_group_ids]
    np.save(cache_dir / "text_embeddings.npy", text_emb_full)
    logger.info(f"  Saved text_embeddings.npy: {text_emb_full.shape}")

    # Save text strings
    with open(cache_dir / "text_strings.json", "w") as f:
        json.dump(new_text_strings, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved text_strings.json: {len(new_text_strings)} entries")

    # Check cross-dataset text similarity
    from numpy.linalg import norm as np_norm
    emb_normed = new_emb / (np_norm(new_emb, axis=1, keepdims=True) + 1e-8)
    cos_mat = emb_normed @ emb_normed.T
    mask_diag = ~np.eye(n_unique, dtype=bool)
    logger.info(f"  Raw embedding pairwise cosine: mean={cos_mat[mask_diag].mean():.4f}")

    return new_emb, new_text_strings


def rebuild_variant_texts(cache_dir: str, device: str = "cuda"):
    """Rebuild variant text embeddings from cleaned enriched variants.

    Uses enriched variants from subcluster_metadata_enriched.json but
    strips "Source: ..." from each variant.
    """
    cache_dir = Path(cache_dir)

    from src.data_pipeline.cache_builder import LatentCacheBuilder

    # ── Load enriched metadata ──
    enriched_meta_path = cache_dir.parent / "processed_h5ad" / "subcluster_metadata_enriched.json"
    metadata_path = cache_dir / "metadata.json"

    with open(enriched_meta_path) as f:
        enriched_meta = json.load(f)
    with open(metadata_path) as f:
        id_to_text = json.load(f)

    # Load group mapping
    text_group_ids = np.load(cache_dir / "text_group_ids.npy")
    sample_ids = np.load(cache_dir / "sample_ids.npy")
    n_cells = len(text_group_ids)
    n_unique = int(text_group_ids.max()) + 1

    # ── Build group_id → cleaned variants mapping ──
    # Iterate over datasets and clusters, collect variants per group_id
    group_variants = {}  # group_id → list of cleaned variant texts

    for sid_str, ds_text in id_to_text.items():
        sid = int(sid_str)
        mask = sample_ids == sid
        cell_indices_global = np.where(mask)[0]

        if len(cell_indices_global) == 0:
            continue

        # Find matching dataset in enriched metadata
        ds_match = None
        for ds_id, ds_info in enriched_meta.items():
            if ds_info["dataset_text"][:80] == ds_text[:80]:
                ds_match = (ds_id, ds_info)
                break

        if ds_match is None:
            continue

        ds_id, ds_info = ds_match
        start_idx = int(cell_indices_global[0])

        for cid, cinfo in ds_info.get("clusters", {}).items():
            variants = cinfo.get("text_variants", [])
            if not variants:
                continue

            # Clean each variant
            cleaned = [clean_variant_text(v) for v in variants]

            # Find the group_id for this cluster's cells
            for local_idx in cinfo.get("cell_indices", []):
                global_idx = start_idx + local_idx
                if global_idx < n_cells and sample_ids[global_idx] == sid:
                    gid = int(text_group_ids[global_idx])
                    if gid not in group_variants:
                        group_variants[gid] = cleaned
                    break  # Only need one cell to identify the group

    logger.info(f"Found variants for {len(group_variants)} groups")

    # ── Flatten and encode ──
    all_variant_texts = []
    variant_group_map = []  # [(group_id, variant_idx), ...]

    for gid in sorted(group_variants.keys()):
        for vi, vtext in enumerate(group_variants[gid]):
            all_variant_texts.append(vtext)
            variant_group_map.append([gid, vi])

    logger.info(f"Total variant texts: {len(all_variant_texts)}")

    # Show examples
    for i in range(min(3, len(all_variant_texts))):
        logger.info(f"  Variant {i}: {all_variant_texts[i][:120]}...")

    # Encode through BiomedBERT
    builder = LatentCacheBuilder(cache_dir=str(cache_dir), device=device)
    logger.info(f"Encoding {len(all_variant_texts)} cleaned variants through BiomedBERT...")
    variant_embs = builder.encode_texts(all_variant_texts)

    # Save
    np.save(cache_dir / "text_variant_embeddings.npy", variant_embs)
    with open(cache_dir / "text_variant_map.json", "w") as f:
        json.dump(variant_group_map, f)

    # Save cleaned variant strings
    new_variants_json = {}
    for gid in sorted(group_variants.keys()):
        new_variants_json[str(gid)] = group_variants[gid]
    with open(cache_dir / "text_variants.json", "w") as f:
        json.dump(new_variants_json, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved {variant_embs.shape} variant embeddings + map")

    return variant_embs


def apply_whitening(cache_dir: str):
    """Apply ZCA whitening to primary + variant text embeddings.

    Re-runs the full whitening pipeline, then applies the TEXT transform
    to unique and variant embeddings too.
    """
    cache_dir = Path(cache_dir)

    from src.data_pipeline.embedding_preprocessor import (
        EmbeddingPreprocessor,
        preprocess_cached_embeddings,
    )

    # ── Step 1: Run full whitening pipeline (fits on unique texts) ──
    logger.info("Running ZCA whitening pipeline...")
    stats = preprocess_cached_embeddings(
        cache_dir=str(cache_dir),
        text_method="whiten",
        cell_method="whiten",
    )
    logger.info(f"Text cosine: {stats['text']['raw_cosine_mean']:.4f} → "
                f"{stats['text']['processed_cosine_mean']:.4f}")
    logger.info(f"Cell cosine: {stats['cell']['raw_cosine_mean']:.4f} → "
                f"{stats['cell']['processed_cosine_mean']:.4f}")

    # ── Step 2: Apply whitening to unique text embeddings ──
    logger.info("Applying whitening to unique text embeddings...")
    text_pre = EmbeddingPreprocessor.load(
        str(cache_dir / "text_preprocessor_preprocessed.npz")
    )
    unique_emb = np.load(cache_dir / "text_embeddings_unique.npy")
    unique_pp = text_pre.transform(unique_emb)
    np.save(cache_dir / "text_embeddings_unique_preprocessed.npy", unique_pp)
    logger.info(f"  Saved unique preprocessed: {unique_pp.shape}")

    # ── Step 3: Apply whitening to variant embeddings ──
    variant_path = cache_dir / "text_variant_embeddings.npy"
    if variant_path.exists():
        logger.info("Applying whitening to variant text embeddings...")
        variant_emb = np.load(variant_path)
        variant_pp = text_pre.transform(variant_emb)
        np.save(variant_path, variant_pp)  # Overwrite with whitened
        logger.info(f"  Whitened {variant_pp.shape[0]} variant embeddings")

        # Check quality
        from numpy.linalg import norm as np_norm
        sub = variant_pp[:min(200, len(variant_pp))]
        sub_n = sub / (np_norm(sub, axis=1, keepdims=True) + 1e-8)
        cos_mat = sub_n @ sub_n.T
        m = ~np.eye(len(sub), dtype=bool)
        logger.info(f"  Variant pairwise cosine: mean={cos_mat[m].mean():.4f}")

        # Primary-variant cosine (how different are variants from their primaries?)
        variant_map_path = cache_dir / "text_variant_map.json"
        if variant_map_path.exists():
            with open(variant_map_path) as f:
                vmap = json.load(f)
            # For each variant, compute cosine to its primary
            cos_list = []
            for vi, (gid, _) in enumerate(vmap):
                prim = unique_pp[gid]
                var = variant_pp[vi]
                pn = prim / (np.linalg.norm(prim) + 1e-8)
                vn = var / (np.linalg.norm(var) + 1e-8)
                cos_list.append(float(pn @ vn))
            logger.info(f"  Primary-variant cosine: mean={np.mean(cos_list):.4f}, "
                        f"std={np.std(cos_list):.4f}")


def update_manifest(cache_dir: str):
    """Update manifest for v6.3."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest["format_version"] = "6.3"
    manifest["description"] = (
        "v6.3: Template-anchored alignment (template primary + enriched augmentation). "
        "Stripped Source: and Leiden cluster info from templates. "
        "Temperature capped at 20 (down from 100)."
    )
    manifest["deduplicated"] = True

    unique_path = cache_dir / "text_embeddings_unique.npy"
    if unique_path.exists():
        manifest["num_unique_texts"] = int(np.load(unique_path).shape[0])

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Updated manifest: {json.dumps(manifest, indent=2)}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild cache for v6.3 (template anchor + enriched augmentation)"
    )
    parser.add_argument("--cache_dir", default="data/cached_latents")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_variants", action="store_true",
                        help="Skip variant re-encoding (faster)")
    parser.add_argument("--skip_whitening", action="store_true",
                        help="Skip ZCA whitening (useful for debugging)")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=level)

    logger.info("=" * 60)
    logger.info("CLOP-DiT v6.3 Cache Rebuild")
    logger.info("Template anchor + enriched augmentation")
    logger.info("=" * 60)

    # Step 1: Rebuild primary text embeddings from cleaned templates
    logger.info("\n" + "─" * 40)
    logger.info("Step 1: Rebuild primary texts (templates)")
    logger.info("─" * 40)
    rebuild_primary_texts(args.cache_dir, args.device)

    # Step 2: Rebuild variant embeddings from cleaned enriched texts
    if not args.skip_variants:
        logger.info("\n" + "─" * 40)
        logger.info("Step 2: Rebuild variant texts (enriched)")
        logger.info("─" * 40)
        rebuild_variant_texts(args.cache_dir, args.device)

    # Step 3: Apply ZCA whitening to all text embeddings
    if not args.skip_whitening:
        logger.info("\n" + "─" * 40)
        logger.info("Step 3: ZCA whitening")
        logger.info("─" * 40)
        apply_whitening(args.cache_dir)

    # Step 4: Update manifest
    logger.info("\n" + "─" * 40)
    logger.info("Step 4: Update manifest")
    logger.info("─" * 40)
    update_manifest(args.cache_dir)

    logger.info("\n" + "=" * 60)
    logger.info("v6.3 CACHE REBUILD COMPLETE")
    logger.info("Next: python scripts/04a_train_clop.py --config configs/clop.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
