#!/usr/bin/env python3
"""Per-field prompt ablation — disentangle individual field contributions.

Extends the conditioning ablation to test each prompt field INDIVIDUALLY:
  1. full           — all 5 fields (cell type + tissue + organism + markers + disease)
  2. markers_only   — only marker genes
  3. celltype_only  — only cell type name
  4. tissue_only    — only tissue of origin
  5. organism_only  — only organism
  6. disease_only   — only disease context
  7. no_markers     — all fields except markers
  8. no_celltype    — all fields except cell type name

For each variant, generates embeddings via the DiT and measures:
  - KNN accuracy against real cells (steering quality)
  - Centroid cosine to real centroids (fidelity)
  - Conditioning fidelity (cosine to text prototype)

Output:
    results/validation/field_ablation_results.json

Usage:
    python scripts/analysis/prompt_field_ablation.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.paths import CACHE_DIR, RESULTS_DIR, CHECKPOINT_DIR
from src.utils.helpers import seed_everything, get_device

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_structured_caption(caption: str) -> dict:
    """Parse a structured caption into individual fields.

    Expected format:
        "Cell_type are found in tissue of organism. They express markers.
         Associated with disease/context."
    """
    fields = {
        "cell_type": "",
        "tissue": "",
        "organism": "",
        "markers": "",
        "disease": "",
    }

    # Cell type: everything before " are "
    if " are " in caption:
        fields["cell_type"] = caption.split(" are ")[0].strip()
        rest = caption.split(" are ", 1)[1]
    else:
        fields["cell_type"] = caption[:50]
        rest = caption

    # Tissue: "found in X of" or "found in X."
    tissue_match = re.search(r"found in ([^.]+?)(?:\s+of\s+|\.\s*)", rest)
    if tissue_match:
        fields["tissue"] = tissue_match.group(1).strip()

    # Organism: "of Homo sapiens" / "of Mus musculus"
    org_match = re.search(r"\bof\s+(Homo\s+sapiens|Mus\s+musculus|[A-Z][a-z]+\s+[a-z]+)", rest)
    if org_match:
        fields["organism"] = org_match.group(1).strip()

    # Markers: "express X, Y, Z" or "express X and Y"
    marker_match = re.search(r"[Tt]hey express\s+([^.]+)", rest)
    if marker_match:
        fields["markers"] = marker_match.group(1).strip()

    # Disease: "Associated with X" or everything after the last period
    disease_match = re.search(r"[Aa]ssociated with\s+([^.]+)", rest)
    if disease_match:
        fields["disease"] = disease_match.group(1).strip()

    return fields


def build_single_field_captions(fields: dict) -> dict:
    """Build single-field and leave-one-out caption variants.

    Returns dict: {variant_name: caption_string}
    """
    variants = {}

    # Full caption reconstruction
    parts = []
    if fields["cell_type"]:
        parts.append(fields["cell_type"])
    if fields["tissue"]:
        parts.append(f"found in {fields['tissue']}")
    if fields["organism"]:
        parts[-1] = parts[-1] + f" of {fields['organism']}" if parts else f"of {fields['organism']}"
    full = " are ".join([fields["cell_type"], " ".join(parts[1:])]) if len(parts) > 1 else fields["cell_type"]
    if fields["markers"]:
        full += f". They express {fields['markers']}"
    if fields["disease"]:
        full += f". Associated with {fields['disease']}"
    variants["full"] = full

    # Single-field variants
    if fields["markers"]:
        variants["markers_only"] = f"Cells that express {fields['markers']}."
    if fields["cell_type"]:
        variants["celltype_only"] = f"{fields['cell_type']}."
    if fields["tissue"]:
        variants["tissue_only"] = f"Cells found in {fields['tissue']}."
    if fields["organism"]:
        variants["organism_only"] = f"Cells of {fields['organism']}."
    if fields["disease"]:
        variants["disease_only"] = f"Cells associated with {fields['disease']}."

    # Leave-one-out variants
    loo_parts = []
    if fields["cell_type"]:
        loo_parts.append(fields["cell_type"])
    if fields["tissue"] or fields["organism"]:
        tissue_org = ""
        if fields["tissue"]:
            tissue_org = f"found in {fields['tissue']}"
        if fields["organism"]:
            tissue_org += f" of {fields['organism']}" if tissue_org else f"of {fields['organism']}"
        loo_parts.append(tissue_org)

    # No markers
    no_markers = " are ".join(loo_parts[:2]) if len(loo_parts) > 1 else (loo_parts[0] if loo_parts else "")
    if fields["disease"]:
        no_markers += f". Associated with {fields['disease']}"
    variants["no_markers"] = no_markers

    # No cell type
    no_ct_parts = []
    if fields["tissue"]:
        no_ct_parts.append(f"Cells found in {fields['tissue']}")
    if fields["organism"]:
        if no_ct_parts:
            no_ct_parts[-1] += f" of {fields['organism']}"
        else:
            no_ct_parts.append(f"Cells of {fields['organism']}")
    if fields["markers"]:
        no_ct_parts.append(f"They express {fields['markers']}")
    if fields["disease"]:
        no_ct_parts.append(f"Associated with {fields['disease']}")
    variants["no_celltype"] = ". ".join(no_ct_parts) if no_ct_parts else "Unknown cells."

    return variants


def load_models(device):
    """Load CLOP and DiT for inference."""
    from src.architecture.dit import DiT1D
    from src.architecture.clop import CLOPAligner

    # CLOP — try subdirectory first, then flat
    clop_path = CHECKPOINT_DIR / "CLOP" / "best" / "clop_best.pth"
    if not clop_path.exists():
        clop_path = CHECKPOINT_DIR / "clop_best.pth"
    ckpt = torch.load(str(clop_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    clop = CLOPAligner(
        text_dim=cfg.get("text_dim", 1024),
        cell_dim=cfg.get("cell_dim", 512),
        proj_dim=cfg.get("proj_dim", 512),
        text_layers=cfg.get("text_layers", 3),
        cell_layers=cfg.get("cell_layers", 3),
        dropout=cfg.get("dropout", 0.2),
        use_batch_norm=cfg.get("use_batch_norm", False),
        use_ema=cfg.get("use_ema", False),
        use_whitening=cfg.get("use_whitening", False),
        loss_type=cfg.get("loss_type", "prototype_siglip"),
    )
    clop.load_state_dict(ckpt["model_state_dict"])
    clop.to(device).eval()

    # DiT — try subdirectory first, then flat
    dit_path = CHECKPOINT_DIR / "DiT" / "best" / "dit_best.pth"
    if not dit_path.exists():
        dit_path = CHECKPOINT_DIR / "dit_best.pth"
    ckpt_d = torch.load(str(dit_path), map_location=device, weights_only=False)
    cfg_d = ckpt_d.get("config", {})
    sd = ckpt_d.get("ema_state_dict", ckpt_d.get("model_state_dict", {}))
    null_cond = sd.get("c_embedder.null_cond")
    cond_dim = null_cond.shape[-1] if null_cond is not None else cfg_d.get("cond_dim", 512)

    dit = DiT1D(
        latent_dim=cfg_d.get("latent_dim", 512),
        hidden_dim=cfg_d.get("hidden_dim", 512),
        cond_dim=cond_dim,
        num_tokens=cfg_d.get("num_tokens", 16),
        num_blocks=8, num_heads=8, cond_drop_prob=0.0,
    )
    if "ema_state_dict" in ckpt_d:
        dit.load_state_dict(ckpt_d["ema_state_dict"])
    else:
        dit.load_state_dict(ckpt_d["model_state_dict"])
    dit.to(device).eval()

    return clop, dit


def encode_text_cached(text: str, clop, device):
    """Encode text through BiomedBERT -> ZCA -> CLOP projection."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        return None

    # Load ZCA params
    npz_path = CACHE_DIR / "text_preprocessor_preprocessed.npz"
    zca_data = np.load(str(npz_path)) if npz_path.exists() else None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
        )
        model = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
        ).to(device).eval()

        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=512, padding=True).to(device)
        with torch.no_grad():
            out = model(**tokens)
            emb = out.last_hidden_state[:, 0, :]

        if zca_data is not None:
            mean = zca_data.get("mean")
            W = zca_data.get("zca_matrix", zca_data.get("transform_matrix"))
            if mean is not None and W is not None:
                emb_np = emb.cpu().numpy()
                emb_np = (emb_np - mean) @ W.T
                emb = torch.from_numpy(emb_np).float().to(device)

        with torch.no_grad():
            proj = clop.project_text(emb)
            proj = F.normalize(proj, dim=-1)
        return proj.cpu().numpy()
    except Exception as e:
        logger.warning(f"Text encoding failed: {e}")
        return None


def evaluate_variant(gen_embs, gen_labels, real_emb, real_labels, k=5):
    """Evaluate generated embeddings against real via KNN and centroid cosine."""
    le = LabelEncoder()
    le.fit(np.concatenate([real_labels, gen_labels]))

    real_y = le.transform(real_labels)
    gen_y = le.transform(gen_labels)

    # KNN accuracy
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(real_emb, real_y)
    gen_pred = knn.predict(gen_embs)
    knn_acc = float(np.mean(gen_pred == gen_y))

    # Per-type centroid cosine
    cos_values = []
    for gid in np.unique(gen_labels):
        real_mask = real_labels == gid
        gen_mask = gen_labels == gid
        if real_mask.sum() < 3 or gen_mask.sum() < 1:
            continue
        real_cent = real_emb[real_mask].mean(axis=0)
        gen_cent = gen_embs[gen_mask].mean(axis=0)
        cos = float(np.dot(real_cent, gen_cent) /
                    (np.linalg.norm(real_cent) * np.linalg.norm(gen_cent) + 1e-10))
        cos_values.append(cos)

    return {
        "knn_accuracy": knn_acc,
        "centroid_cosine_mean": float(np.mean(cos_values)) if cos_values else 0,
        "centroid_cosine_std": float(np.std(cos_values)) if cos_values else 0,
    }


def main():
    seed_everything(42)
    device = get_device()
    output_dir = RESULTS_DIR / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load captions
    cap_path = CACHE_DIR / "text_captions_deduplicated.json"
    with open(cap_path) as f:
        captions = json.load(f)

    # Load real data
    real_emb = np.load(str(CACHE_DIR / "cell_embeddings_dedup_preprocessed.npy"))
    real_labels = np.load(str(CACHE_DIR / "text_group_ids_dedup.npy"))

    # Load projected text (already through CLOP)
    proj_text = np.load(str(CACHE_DIR / "projected_text.npy"))

    # Parse all captions into fields
    logger.info("Parsing captions into structured fields...")
    all_fields = {}
    all_variants = {}
    for gid_str, caption in captions.items():
        gid = int(gid_str)
        fields = parse_structured_caption(caption)
        all_fields[gid] = fields
        all_variants[gid] = build_single_field_captions(fields)

    # Print example
    example_gid = list(all_variants.keys())[0]
    logger.info(f"Example variants for type {example_gid}:")
    for vname, vcap in all_variants[example_gid].items():
        logger.info(f"  {vname:20s}: {vcap[:80]}")

    # Load models
    logger.info("Loading CLOP + DiT models...")
    clop, dit = load_models(device)

    # For each variant, encode the text, generate, and evaluate
    variant_names = ["full", "markers_only", "celltype_only", "tissue_only",
                     "organism_only", "disease_only", "no_markers", "no_celltype"]

    results = {}
    n_samples_per_type = 50

    for variant_name in variant_names:
        logger.info(f"\nProcessing variant: {variant_name}")

        all_gen_embs = []
        all_gen_labels = []
        n_types_processed = 0

        for gid_str, caption in captions.items():
            gid = int(gid_str)
            variants = all_variants.get(gid, {})

            if variant_name not in variants:
                # Use original CLOP-projected text as fallback
                cond = torch.from_numpy(proj_text[gid:gid+1]).float().to(device)
            else:
                # Encode the variant text through model pipeline
                variant_text = variants[variant_name]
                cond_np = encode_text_cached(variant_text, clop, device)
                if cond_np is None:
                    # If encoding fails, use original projected text
                    cond = torch.from_numpy(proj_text[gid:gid+1]).float().to(device)
                else:
                    cond = torch.from_numpy(cond_np).float().to(device)

            # Generate
            cond_batch = cond.expand(n_samples_per_type, -1)
            with torch.no_grad():
                gen = dit.sample(cond_batch, num_steps=30, cfg_scale=2.0)

            all_gen_embs.append(gen.cpu().numpy())
            all_gen_labels.extend([gid] * n_samples_per_type)
            n_types_processed += 1

        gen_embs = np.concatenate(all_gen_embs, axis=0)
        gen_labels = np.array(all_gen_labels)

        metrics = evaluate_variant(gen_embs, gen_labels, real_emb, real_labels)
        metrics["n_types"] = n_types_processed
        metrics["n_samples"] = len(gen_labels)
        results[variant_name] = metrics

        logger.info(f"  {variant_name}: KNN={metrics['knn_accuracy']:.4f}, "
                    f"cos={metrics['centroid_cosine_mean']:.4f}")

    # Compute relative contributions
    if "full" in results:
        full_knn = results["full"]["knn_accuracy"]
        for vname in results:
            results[vname]["knn_delta_vs_full"] = float(
                results[vname]["knn_accuracy"] - full_knn)
            results[vname]["knn_retention_pct"] = float(
                100 * results[vname]["knn_accuracy"] / full_knn) if full_knn > 0 else 0

    # Save
    with open(output_dir / "field_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("  Prompt Field Ablation Results")
    print(f"{'='*70}")
    print(f"  {'Variant':20s} {'KNN Acc':>10s} {'Cos':>10s} {'Retention':>12s}")
    print(f"  {'-'*52}")
    for vname in variant_names:
        if vname in results:
            r = results[vname]
            ret = f"{r.get('knn_retention_pct', 0):.1f}%"
            print(f"  {vname:20s} {r['knn_accuracy']:10.4f} "
                  f"{r['centroid_cosine_mean']:10.4f} {ret:>12s}")
    print(f"\n  Saved to: {output_dir}/field_ablation_results.json")


if __name__ == "__main__":
    main()
