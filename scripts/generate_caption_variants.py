#!/usr/bin/env python3
"""
Generate caption variants for CLOP text augmentation.

Creates paraphrased/augmented versions of each of the 1088 sub-cluster captions
by applying controlled text transformations:
  1. Sentence reordering (swap definition and detail sentences)
  2. Synonym substitution (cell biology terms)
  3. Detail truncation (shorter versions = regularization)
  4. Tissue/organism context variation
  5. Marker gene subset sampling

Outputs:
  - text_variant_embeddings.npy: (N_variants, 1024) BiomedBERT embeddings
  - text_variant_map.json: [[group_id, variant_idx], ...] mapping
  - text_variants.json: {group_id: [variant_str_1, ...]} for inspection

Usage:
    python scripts/generate_caption_variants.py
    python scripts/generate_caption_variants.py --n_variants 5 --cache_dir data/cached_latents_v5.2
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ── Synonym dictionaries for controlled augmentation ──
CELL_SYNONYMS = {
    "cells": ["cell populations", "cellular subsets", "cell subpopulations"],
    "are": ["represent", "constitute", "comprise"],
    "found": ["observed", "detected", "identified", "present"],
    "including": ["such as", "encompassing", "comprising"],
    "critical for": ["essential for", "crucial for", "important for", "key to"],
    "responsible for": ["involved in", "mediating", "driving"],
    "producing": ["secreting", "generating", "synthesizing"],
    "regulation": ["modulation", "control", "governance"],
    "differentiated": ["mature", "specialized", "committed"],
    "progenitor": ["precursor", "ancestor", "forerunner"],
    "expressing": ["displaying", "showing expression of", "positive for"],
    "provide": ["supply", "deliver", "offer"],
    "support": ["sustain", "maintain", "bolster"],
    "function": ["activity", "role", "action"],
    "immune": ["immunological", "immunity-related"],
    "inflammatory": ["pro-inflammatory", "inflammation-associated"],
}

ORGANISM_VARIANTS = {
    "human": ["Homo sapiens", "human tissue", "human-derived"],
    "mouse": ["Mus musculus", "murine", "mouse-derived"],
}

def extract_sentences(caption: str) -> List[str]:
    """Split caption into sentences, handling abbreviations."""
    # Split on period followed by space and capital letter, but not abbreviations
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', caption.strip())
    return [p.strip() for p in parts if p.strip()]


def synonym_replace(text: str, n_replacements: int = 2) -> str:
    """Replace up to n random terms with synonyms."""
    result = text
    keys = list(CELL_SYNONYMS.keys())
    random.shuffle(keys)
    replaced = 0
    for key in keys:
        if replaced >= n_replacements:
            break
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, result, re.IGNORECASE):
            synonym = random.choice(CELL_SYNONYMS[key])
            result = re.sub(pattern, synonym, result, count=1, flags=re.IGNORECASE)
            replaced += 1
    return result


def truncate_detail(caption: str) -> str:
    """Remove the last 1-2 sentences to create a shorter version."""
    sentences = extract_sentences(caption)
    if len(sentences) <= 1:
        return caption
    # Keep first 1-2 sentences (definition part)
    keep = max(1, len(sentences) - random.randint(1, min(2, len(sentences) - 1)))
    return " ".join(sentences[:keep])


def reorder_sentences(caption: str) -> str:
    """Reorder non-first sentences."""
    sentences = extract_sentences(caption)
    if len(sentences) <= 2:
        return caption
    # Keep first sentence, shuffle the rest
    rest = sentences[1:]
    random.shuffle(rest)
    return " ".join([sentences[0]] + rest)


def swap_organism_mention(caption: str) -> str:
    """Replace organism mentions with variants."""
    result = caption
    for org, variants in ORGANISM_VARIANTS.items():
        if org in result.lower():
            variant = random.choice(variants)
            result = re.sub(r'\b' + org + r'\b', variant, result, count=1, flags=re.IGNORECASE)
            break
    return result


def sample_marker_genes(caption: str) -> str:
    """If caption mentions marker genes, randomly drop some."""
    # Pattern: "expression of GENE1, GENE2, GENE3, ..."
    match = re.search(r'(expression of |markers? (?:include |are )?)([\w,\s/]+?)(?:\.|$)', caption)
    if not match:
        return caption
    prefix = match.group(1)
    genes_str = match.group(2)
    genes = [g.strip() for g in re.split(r'[,\s]+', genes_str) if g.strip() and len(g.strip()) > 1]
    if len(genes) <= 2:
        return caption
    # Keep random subset (at least half)
    keep = random.sample(genes, max(2, len(genes) // 2))
    new_genes = ", ".join(keep)
    return caption[:match.start()] + prefix + new_genes + "." + caption[match.end():]


def generate_variants(caption: str, n_variants: int = 5) -> List[str]:
    """Generate n_variants augmented versions of caption."""
    variants = []
    transforms = [
        lambda c: synonym_replace(c, n_replacements=2),
        lambda c: synonym_replace(c, n_replacements=3),
        lambda c: truncate_detail(c),
        lambda c: reorder_sentences(c),
        lambda c: swap_organism_mention(c),
        lambda c: sample_marker_genes(c),
        lambda c: synonym_replace(truncate_detail(c), 2),  # combined
        lambda c: swap_organism_mention(synonym_replace(c, 1)),  # combined
    ]
    
    for i in range(n_variants):
        transform = transforms[i % len(transforms)]
        variant = transform(caption)
        # Only add if meaningfully different
        if variant != caption and len(variant) > 20:
            variants.append(variant)
    
    # Deduplicate
    variants = list(dict.fromkeys(variants))
    return variants[:n_variants]


def embed_texts(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Embed texts using BiomedBERT."""
    from transformers import AutoTokenizer, AutoModel
    
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
        
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Embedded {i + len(batch)}/{len(texts)} texts")
    
    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Generate caption variants for CLOP augmentation")
    parser.add_argument("--cache_dir", type=str, default="data/cached_latents_v5.2")
    parser.add_argument("--n_variants", type=int, default=5, help="Variants per caption")
    parser.add_argument("--text_encoder", type=str,
                        default="microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    setup_logging()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    cache_dir = Path(args.cache_dir)
    
    # Load original captions (1088 sub-cluster level)
    text_strings_path = cache_dir / "text_strings_polished_pre_v3.json"
    if not text_strings_path.exists():
        logger.error(f"Text strings not found: {text_strings_path}")
        sys.exit(1)
    
    with open(text_strings_path) as f:
        text_strings = json.load(f)
    
    logger.info(f"Loaded {len(text_strings)} original captions")
    
    # Generate variants for each caption
    all_variants = {}  # group_id -> [variant_str, ...]
    variant_map = []   # [[group_id, variant_idx], ...]
    variant_texts = [] # flat list of variant strings for embedding
    
    for gid in sorted(text_strings.keys(), key=int):
        caption = text_strings[gid]
        variants = generate_variants(caption, args.n_variants)
        all_variants[gid] = variants
        for vi, vtxt in enumerate(variants):
            variant_map.append([int(gid), vi])
            variant_texts.append(vtxt)
    
    logger.info(f"Generated {len(variant_texts)} caption variants across {len(all_variants)} groups")
    logger.info(f"Avg variants/group: {len(variant_texts)/len(all_variants):.1f}")
    
    # Save variant texts
    with open(cache_dir / "text_variant_map.json", "w") as f:
        json.dump(variant_map, f)
    with open(cache_dir / "text_variants.json", "w") as f:
        json.dump(all_variants, f, indent=2)
    logger.info(f"Saved variant map and texts")
    
    # Embed variants
    logger.info(f"Embedding {len(variant_texts)} variant texts...")
    embeddings = embed_texts(variant_texts, args.text_encoder)
    np.save(cache_dir / "text_variant_embeddings.npy", embeddings)
    logger.info(f"Saved variant embeddings: {embeddings.shape}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Caption Variant Generation Summary")
    logger.info(f"  Original captions: {len(text_strings)}")
    logger.info(f"  Total variants: {len(variant_texts)}")
    logger.info(f"  Embedding shape: {embeddings.shape}")
    logger.info(f"  Output files:")
    logger.info(f"    text_variant_embeddings.npy")
    logger.info(f"    text_variant_map.json")
    logger.info(f"    text_variants.json")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
