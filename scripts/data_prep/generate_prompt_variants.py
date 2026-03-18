#!/usr/bin/env python3
"""
Generate prompt variants for CLOP robustness evaluation.

Takes base prompts (configs/prompts/default_prompts.json) and generates
expanded variants for each cell type using template-based transformations:
  1. Context shuffling (reorder sentences)
  2. Marker dropping (remove tissue/organ context sentence)
  3. Granularity levels (concise, standard, detailed)
  4. Tissue context variation (replace tissue/organ context)
  5. Synonym substitution (biological term variants)

Each cell type gets up to N variants (default 5). Variants are suitable
for CLOP robustness testing — measuring how sensitive generation is to
natural prompt variation.

Output: configs/prompts/expanded_variants.json

Usage:
    python scripts/data_prep/generate_prompt_variants.py
    python scripts/data_prep/generate_prompt_variants.py --n_variants 10
    python scripts/data_prep/generate_prompt_variants.py --input configs/prompts/default_prompts.json
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ── Biological synonym dictionaries ──────────────────────────────────────

CELL_TERM_SYNONYMS = {
    "T lymphocytes": ["T cells", "T-cell populations"],
    "T cells": ["T lymphocytes", "T-cell subsets"],
    "macrophages": ["mononuclear phagocytes", "tissue macrophages"],
    "natural killer cells": ["NK cells", "innate lymphoid killer cells"],
    "NK cells": ["natural killer cells", "cytotoxic innate lymphocytes"],
    "epithelial cells": ["epithelial populations", "epithelium-derived cells"],
    "fibroblasts": ["stromal fibroblasts", "mesenchymal cells"],
    "dendritic cells": ["antigen-presenting dendritic cells", "DCs"],
    "B cells": ["B lymphocytes", "B-cell populations"],
    "monocytes": ["circulating monocytes", "mononuclear cells"],
    "endothelial cells": ["vascular endothelial cells", "endothelium"],
    "tumor-infiltrating": ["cancer-infiltrating", "intratumoural"],
    "tumor microenvironment": ["tumour niche", "cancer microenvironment"],
    "cytotoxic": ["cell-killing", "cytolytic"],
    "expressing": ["positive for", "displaying"],
}

TISSUE_CONTEXTS = [
    "human lung adenocarcinoma",
    "human breast carcinoma",
    "human pancreatic tissue",
    "human liver tissue",
    "human colorectal carcinoma",
    "human kidney tissue",
    "mouse lung tissue",
    "human peripheral blood",
    "human brain cortex",
    "human skin tissue",
]


# ── Variant generation strategies ────────────────────────────────────────

def shuffle_sentences(text: str) -> str:
    """Reorder sentences while keeping the first sentence fixed."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    if len(parts) <= 2:
        return " ".join(reversed(parts))
    first = parts[0]
    rest = parts[1:]
    random.shuffle(rest)
    return " ".join([first] + rest)


def drop_context(text: str) -> str:
    """Remove the tissue/organ context sentence (usually first sentence)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    if len(parts) <= 1:
        return text
    # Drop the sentence that mentions tissue context (usually first)
    return " ".join(parts[1:])


def make_concise(text: str) -> str:
    """Create a shorter version keeping only the first sentence."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return parts[0]


def synonym_replace(text: str, n_max: int = 2) -> str:
    """Replace up to n_max biological terms with synonyms."""
    result = text
    keys = list(CELL_TERM_SYNONYMS.keys())
    random.shuffle(keys)
    replaced = 0
    for key in keys:
        if replaced >= n_max:
            break
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, result, re.IGNORECASE):
            synonym = random.choice(CELL_TERM_SYNONYMS[key])
            result = re.sub(pattern, synonym, result, count=1, flags=re.IGNORECASE)
            replaced += 1
    return result


def swap_tissue_context(text: str) -> str:
    """Replace the tissue/organ context with a different one."""
    result = text
    for tissue in TISSUE_CONTEXTS:
        if tissue.lower() in result.lower():
            replacement = random.choice(
                [t for t in TISSUE_CONTEXTS if t.lower() != tissue.lower()]
            )
            result = re.sub(re.escape(tissue), replacement, result, flags=re.IGNORECASE)
            break
    return result


# ── Main expansion logic ─────────────────────────────────────────────────

STRATEGIES = [
    ("context_shuffled", shuffle_sentences),
    ("marker_dropped", drop_context),
    ("concise", make_concise),
    ("synonym_replaced", synonym_replace),
    ("tissue_swapped", swap_tissue_context),
]


def expand_prompts(
    base_prompts: Dict[str, str], n_variants: int = 5, seed: int = 42
) -> Dict[str, List[Dict[str, str]]]:
    """Expand each prompt to multiple variants.

    Returns dict: {cell_type: [{variant_type, text}, ...]}
    """
    random.seed(seed)
    expanded = {}

    for cell_type, base_text in base_prompts.items():
        variants = []
        # Always include original
        variants.append({"variant_type": "original", "text": base_text})

        # Apply each strategy
        for strategy_name, strategy_fn in STRATEGIES:
            if len(variants) >= n_variants + 1:  # +1 for original
                break
            variant_text = strategy_fn(base_text)
            # Only add if meaningfully different from original
            if variant_text.strip() != base_text.strip():
                variants.append({"variant_type": strategy_name, "text": variant_text})

        expanded[cell_type] = variants
        logger.info(
            f"  {cell_type}: {len(variants)} variants "
            f"(strategies: {[v['variant_type'] for v in variants]})"
        )

    return expanded


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Generate prompt variants for robustness evaluation")
    parser.add_argument(
        "--input",
        type=str,
        default="configs/prompts/default_prompts.json",
        help="Input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/prompts/expanded_variants.json",
        help="Output expanded variants JSON file",
    )
    parser.add_argument("--n_variants", type=int, default=5, help="Target variants per cell type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        base_prompts = json.load(f)

    logger.info(f"Loaded {len(base_prompts)} base prompts from {input_path}")
    expanded = expand_prompts(base_prompts, n_variants=args.n_variants, seed=args.seed)

    total_variants = sum(len(v) for v in expanded.values())
    logger.info(f"Generated {total_variants} total variants for {len(expanded)} cell types")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
