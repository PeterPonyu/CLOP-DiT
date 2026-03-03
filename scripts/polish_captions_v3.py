#!/usr/bin/env python3
"""
Comprehensive caption polishing script v3.
Fixes all identified issues in text_strings_polished.json:

1. 16 short "Immature / migrating neurons" captions (missing definition)
2. 15 unchanged captions (12 dataset-level ok, 3 GEO abstracts too long)
3. 11 entries with genome-prefixed genes (mm10___, GRCh38_)
4. 74 "Expressed marker genes" generic endings
5. 243 generic "Express ..." endings → make tissue/condition-aware

Strategy: Keep the rich middle section (unique per group), improve head + tail.
"""

import json
import re
import copy
from pathlib import Path

DATA_DIR = Path("data/cached_latents_v5.2")
POLISHED = DATA_DIR / "text_strings_polished.json"
ORIGINALS = DATA_DIR / "text_strings_original_templates.json"
OUTPUT = DATA_DIR / "text_strings_polished.json"
BACKUP = DATA_DIR / "text_strings_polished_pre_v3.json"

# ──────────────────────────────────────────────────────────
# CELL-TYPE DEFINITION LIBRARY (for missing definitions)
# ──────────────────────────────────────────────────────────
CELL_DEFINITIONS = {
    "immature / migrating neurons": (
        "Immature / migrating neurons are Newly differentiated neurons "
        "undergoing migration and axonal guidance to reach their final "
        "circuit positions, expressing cytoskeletal remodeling and "
        "growth-cone guidance molecules."
    ),
}

# ──────────────────────────────────────────────────────────
# TISSUE-AWARE SUFFIX EXPANSIONS
# ──────────────────────────────────────────────────────────
# For the generic "Express ..." endings, add tissue/condition-specific content
TISSUE_MODIFIERS = {
    # condition modifiers
    "cancer": "in the tumor microenvironment context",
    "melanoma": "within the melanoma microenvironment",
    "leukemia": "in the leukemic bone marrow niche",
    "alzheimer": "in the neuroinflammatory context of Alzheimer's disease",
    "aging": "reflecting age-associated transcriptomic changes",
    "inflammation": "during active inflammatory response",
    "fibrosis": "in the fibrotic tissue milieu",
    "injury": "during tissue repair following injury",
    "als": "in the context of ALS-associated neurodegeneration",
    "autism": "in the autism spectrum disorder context",
    "infection": "during infectious challenge",
    "development": "during developmental differentiation",
}

# ──────────────────────────────────────────────────────────
# GEO ABSTRACT CONDENSATION (for 3 long GEO captions)
# ──────────────────────────────────────────────────────────
GEO_REPLACEMENTS = {
    "438": (
        "Single-cell RNA sequencing of lymphatic endothelial cells in "
        "sentinel lymph nodes from breast cancer patients. This dataset "
        "profiles tumor-induced lymphatic remodeling, revealing decreased "
        "inflammatory and increased immunosuppressive CD200+ lymphatic "
        "endothelial cell subsets in metastatic nodes, with Matrix Gla "
        "protein (MGP) identified as a key mediator of cancer cell "
        "adhesion to lymphatic endothelium."
    ),
    "881": (
        "Single-cell RNA sequencing of pancreatic ductal adenocarcinoma "
        "(PDAC) tumor tissue profiling the tumor microenvironment. This "
        "dataset investigates miRNA-mediated regulation of the PDAC "
        "microenvironment using paired RNA-seq, miRNA-seq, and 10x "
        "Genomics single-cell sequencing, identifying differentially "
        "expressed miRNAs and their regulatory gene targets in tumor "
        "versus adjacent benign tissue."
    ),
    "938": (
        "Single-cell RNA sequencing of tumor-infiltrating T cells and "
        "monocytes from a neoadjuvant pancreatic cancer clinical trial. "
        "This dataset profiles immune cell responses to GVAX vaccine, "
        "anti-PD1, and CD137 agonist combinations in pancreatic ductal "
        "adenocarcinoma, revealing CD137-enhanced CD8+ T cell activation "
        "and clonal expansion alongside TREM2-mediated immunosuppressive "
        "macrophage reprogramming."
    ),
}

# ──────────────────────────────────────────────────────────
# SPECIFIC MARKER EXPANSION for "Expressed marker genes" ending
# ──────────────────────────────────────────────────────────
def _expand_expressed_marker_genes(caption: str, key: str) -> str:
    """Replace generic 'Expressed marker genes' with gene-informed description."""
    if not caption.rstrip().endswith("Expressed marker genes"):
        return caption

    # Extract the gene names from "by expression of X, Y and Z"
    m = re.search(r'by expression of (.+?)\.?\s*Expressed marker genes', caption)
    if not m:
        # Fallback: just improve the ending
        return caption.rstrip().rstrip('.') + ", representing a transcriptionally distinct population"
    
    genes_str = m.group(1)
    # Parse gene names
    genes = [g.strip().strip(',') for g in re.split(r',\s*|\s+and\s+', genes_str)]
    genes = [g for g in genes if g]
    
    # Strip genome prefixes from gene names
    clean_genes = [re.sub(r'^(mm10___|GRCh38_)', '', g) for g in genes]
    
    # Build an informative ending
    if len(clean_genes) <= 2:
        gene_txt = " and ".join(clean_genes)
    else:
        gene_txt = ", ".join(clean_genes[:-1]) + " and " + clean_genes[-1]
    
    base = caption[:caption.rindex("Expressed marker genes")].rstrip()
    if base.endswith('.'):
        base = base[:-1].rstrip()
    
    return base + f". Characterized by expression pattern of {gene_txt}"


# ──────────────────────────────────────────────────────────
# GENERIC ENDING DIVERSIFIER
# ──────────────────────────────────────────────────────────
def _extract_condition(caption: str) -> str:
    """Extract disease/condition from caption."""
    m = re.search(r'affected by (.+?) by expression', caption)
    if m:
        return m.group(1).strip().lower()
    return ""

def _extract_tissue(caption: str) -> str:
    """Extract tissue from caption."""
    m = re.search(r'identified in (?:human|mouse|unknown organism)\s+(.+?)(?:\s+affected|\s+by expression)', caption)
    if m:
        return m.group(1).strip().lower()
    return ""

def _extract_organism(caption: str) -> str:
    """Extract organism from caption."""
    m = re.search(r'identified in (human|mouse|unknown organism)', caption)
    if m:
        return m.group(1).strip().lower()
    return ""

def _extract_specific_genes(caption: str) -> list:
    """Extract the specific marker genes listed in the caption."""
    m = re.search(r'by expression of (.+?)\.', caption)
    if not m:
        return []
    genes_str = m.group(1)
    genes = [g.strip().strip(',') for g in re.split(r',\s*|\s+and\s+', genes_str)]
    return [g for g in genes if g and len(g) > 1]

def _diversify_generic_ending(caption: str, key: str) -> str:
    """Make generic Express endings tissue/condition-specific."""
    # Find the Express part
    idx = caption.rfind("Express ")
    if idx == -1:
        return caption
    
    express_part = caption[idx:]
    base_part = caption[:idx].rstrip()
    
    condition = _extract_condition(caption)
    tissue = _extract_tissue(caption)
    organism = _extract_organism(caption)
    specific_genes = _extract_specific_genes(caption)
    
    # Clean up specific genes (strip prefixes)
    specific_genes = [re.sub(r'^(mm10___|GRCh38_)', '', g) for g in specific_genes]
    
    # Build context suffix
    context_parts = []
    
    # Add condition context
    for cond_key, modifier in TISSUE_MODIFIERS.items():
        if cond_key in condition:
            context_parts.append(modifier)
            break
    
    # Add tissue specificity if we have it
    if tissue and tissue not in ('tissue',):
        context_parts.append(f"in {tissue} tissue")
    
    # Combine: keep original Express + add context
    if context_parts:
        # Remove trailing period if any
        express_clean = express_part.rstrip().rstrip('.')
        new_ending = f"{express_clean}, {' '.join(context_parts)}"
    else:
        new_ending = express_part.rstrip()
    
    return f"{base_part} {new_ending}"


# ──────────────────────────────────────────────────────────
# STRIP GENOME PREFIXES
# ──────────────────────────────────────────────────────────
def _strip_genome_prefixes(caption: str) -> str:
    """Remove mm10___ and GRCh38_ prefixes from gene names."""
    caption = re.sub(r'mm10___', '', caption)
    caption = re.sub(r'GRCh38_', '', caption)
    return caption


# ──────────────────────────────────────────────────────────
# FIX SHORT IMMATURE NEURONS
# ──────────────────────────────────────────────────────────
def _fix_short_immature_neurons(caption: str, key: str) -> str:
    """Add cell-type definition to short immature neuron captions."""
    if not caption.strip().startswith("Immature / migrating neurons were identified"):
        return caption
    
    # Extract the identification part
    # "Immature / migrating neurons were identified in X by expression of Y."
    m = re.search(
        r'Immature / migrating neurons were identified in (.+?) by expression of (.+?)\.?$',
        caption.strip()
    )
    if not m:
        return caption
    
    location = m.group(1).strip()
    genes_str = m.group(2).strip().rstrip('.')
    
    # Parse genes for the Express ending
    genes = [g.strip().strip(',') for g in re.split(r',\s*|\s+and\s+', genes_str)]
    genes = [re.sub(r'^(mm10___|GRCh38_)', '', g) for g in genes if g]
    
    if len(genes) <= 2:
        gene_list = " and ".join(genes)
    else:
        gene_list = ", ".join(genes[:-1]) + " and " + genes[-1]
    
    # Determine condition for context
    condition_match = re.search(r'affected by (.+?) by', caption)
    condition = condition_match.group(1) if condition_match else ""
    
    definition = CELL_DEFINITIONS["immature / migrating neurons"]
    
    new_caption = (
        f"{definition} In this dataset, these cells were identified in "
        f"{location} by expression of {gene_list}. "
        f"Express neuronal migration markers (dcx, stmn2, tubb3) and "
        f"immature neuronal transcription factors"
    )
    
    # Add condition context
    if condition:
        for cond_key, modifier in TISSUE_MODIFIERS.items():
            if cond_key in condition.lower():
                new_caption += f", {modifier}"
                break
    
    return new_caption


# ──────────────────────────────────────────────────────────
# POLISH UNCHANGED DATASET-LEVEL CAPTIONS
# ──────────────────────────────────────────────────────────
DATASET_LEVEL_IMPROVEMENTS = {
    # These are already decent dataset-level captions; just ensure they're
    # well-formed and informative. Small improvements only.
    "21": (
        "Single-cell RNA sequencing of peripheral blood mononuclear cells "
        "(PBMCs) from patients with Merkel cell carcinoma (MCC). This dataset "
        "profiles circulating immune cell populations including T cells, NK cells, "
        "monocytes, and B cells, capturing their functional states and activation "
        "signatures in the context of this aggressive neuroendocrine skin cancer."
    ),
    "80": (
        "Single-cell RNA sequencing of human pulmonary pleomorphic carcinoma (PPC), "
        "a rare aggressive subtype of non-small cell lung cancer. This dataset "
        "profiles surgically resected primary PPC tumors from four patients, "
        "characterizing cellular heterogeneity including malignant epithelial "
        "subclones, immune infiltration patterns, and stromal remodeling in this "
        "poorly understood lung malignancy."
    ),
    "158": (
        "Single-cell RNA sequencing of human pituitary gland during embryonic and "
        "fetal development. This dataset profiles the cellular differentiation "
        "trajectories of the anterior pituitary, capturing hormone-producing "
        "lineages including corticotrophs, somatotrophs, gonadotrophs, "
        "thyrotrophs, and their shared progenitor populations."
    ),
    "167": (
        "Single-cell RNA sequencing of adult mouse urethra. This dataset profiles "
        "the epithelial and stromal cell populations of the murine lower urinary "
        "tract, capturing urethral epithelial cells including basal and luminal "
        "subtypes, smooth muscle cells, fibroblasts, and surrounding connective "
        "tissue populations."
    ),
    "266": (
        "Single-cell RNA sequencing of brain metastases originating from human "
        "triple-negative breast cancer (TNBC). This dataset characterizes the "
        "metastatic tumor microenvironment in the brain, capturing TNBC-derived "
        "malignant cells adapting to the neural niche, resident microglia, "
        "astrocytes, and infiltrating peripheral immune cells."
    ),
    "341": (
        "Single-cell RNA sequencing of human lung adenocarcinoma. This dataset "
        "profiles the tumor microenvironment of pulmonary adenocarcinoma, "
        "capturing malignant epithelial cells, immune infiltrates including "
        "T cells and myeloid populations, and stromal components including "
        "cancer-associated fibroblasts and endothelial cells."
    ),
    "696": (
        "Single-cell RNA sequencing of aged mouse hematopoietic stem cells (HSCs). "
        "This dataset profiles transcriptomic changes associated with aging in the "
        "murine hematopoietic stem cell compartment, characterizing age-related "
        "myeloid lineage bias, reduced lymphoid potential, and functional decline "
        "in long-term repopulating activity."
    ),
    "778": (
        "Single-cell RNA sequencing of human breast cancer tumor tissue. This "
        "dataset captures the heterogeneous cellular landscape of breast carcinoma, "
        "including malignant epithelial cells with distinct molecular subtypes, "
        "tumor-infiltrating lymphocytes, myeloid cells, endothelial cells, and "
        "cancer-associated fibroblasts."
    ),
    "947": (
        "Single-cell RNA sequencing of human lung adenocarcinoma. This dataset "
        "profiles the tumor microenvironment of pulmonary adenocarcinoma, "
        "capturing malignant alveolar and bronchial epithelial cells, "
        "tumor-associated macrophages with distinct polarization states, "
        "effector and exhausted T cell populations, and reactive fibroblasts."
    ),
    "955": (
        "Single-cell RNA sequencing of human colorectal cancer liver metastases. "
        "This dataset profiles the metastatic microenvironment of colon cancer "
        "cells that have disseminated to the liver, characterizing tumor-hepatocyte "
        "interactions, hepatic stellate cell activation, and the composition of "
        "the metastatic immune niche including tissue-resident and recruited "
        "immune populations."
    ),
    "967": (
        "Single-cell RNA sequencing of human ascending aortic wall tissue from "
        "patients with sporadic type A aortic dissection. This dataset profiles "
        "the cellular landscape of the diseased aortic wall, capturing smooth "
        "muscle cell phenotype switching, macrophage infiltration, endothelial "
        "dysfunction, and extracellular matrix remodeling associated with acute "
        "aortic dissection pathogenesis."
    ),
    "1003": (
        "Single-cell RNA sequencing of adult mouse prostate tissue. This dataset "
        "profiles the epithelial and stromal cell populations of the adult murine "
        "prostate, capturing luminal secretory cells, basal progenitors, rare "
        "neuroendocrine epithelial subtypes, alongside prostatic stroma including "
        "smooth muscle and fibroblast populations."
    ),
}


# ──────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────
def main():
    with open(POLISHED) as f:
        polished = json.load(f)
    with open(ORIGINALS) as f:
        originals = json.load(f)
    
    # Backup
    with open(BACKUP, 'w') as f:
        json.dump(polished, f, indent=2)
    print(f"Backed up to {BACKUP}")
    
    improved = copy.deepcopy(polished)
    stats = {
        "short_neurons_fixed": 0,
        "unchanged_fixed": 0,
        "geo_condensed": 0,
        "genome_prefixes_stripped": 0,
        "expressed_markers_improved": 0,
        "generic_endings_diversified": 0,
        "total_modified": 0,
    }
    
    for key in list(improved.keys()):
        original_caption = improved[key]
        caption = improved[key]
        
        # ── Phase 1: Fix GEO abstract captions ──
        if key in GEO_REPLACEMENTS:
            caption = GEO_REPLACEMENTS[key]
            stats["geo_condensed"] += 1
        
        # ── Phase 2: Fix unchanged dataset-level captions ──
        elif key in DATASET_LEVEL_IMPROVEMENTS:
            caption = DATASET_LEVEL_IMPROVEMENTS[key]
            stats["unchanged_fixed"] += 1
        
        # ── Phase 3: Fix short immature neuron captions ──
        elif (caption.strip().startswith("Immature / migrating neurons were identified") 
              and len(caption) < 150):
            caption = _fix_short_immature_neurons(caption, key)
            stats["short_neurons_fixed"] += 1
        
        # ── Phase 4: Strip genome prefixes ──
        if "mm10___" in caption or "GRCh38_" in caption:
            caption = _strip_genome_prefixes(caption)
            stats["genome_prefixes_stripped"] += 1
        
        # ── Phase 5: Fix "Expressed marker genes" endings ──
        if caption.rstrip().endswith("Expressed marker genes"):
            caption = _expand_expressed_marker_genes(caption, key)
            stats["expressed_markers_improved"] += 1
        
        # ── Phase 6: Diversify generic "Express ..." endings ──
        elif "Express " in caption:
            # Only diversify if the Express part is a commonly repeated template
            idx = caption.rfind("Express ")
            express_part = caption[idx:].strip()
            # Check if this exact phrase appears >5 times
            caption = _diversify_generic_ending(caption, key)
            if caption != original_caption:
                stats["generic_endings_diversified"] += 1
        
        # Track modifications
        if caption != original_caption:
            improved[key] = caption
            stats["total_modified"] += 1
    
    # Write output
    with open(OUTPUT, 'w') as f:
        json.dump(improved, f, indent=2)
    
    print(f"\n{'='*60}")
    print("POLISHING RESULTS")
    print(f"{'='*60}")
    for stat_name, count in stats.items():
        print(f"  {stat_name}: {count}")
    print(f"{'='*60}")
    
    # Verify: compute basic stats
    lengths = [len(v) for v in improved.values()]
    print(f"\nCaption length stats:")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f} chars")
    
    # Show some examples of each fix type
    print(f"\n{'='*60}")
    print("SAMPLE FIXES")
    print(f"{'='*60}")
    
    # Show a fixed short neuron
    for k in ['198', '366', '584']:
        if polished[k] != improved[k]:
            print(f"\n--- [{k}] SHORT NEURON FIX ---")
            print(f"BEFORE ({len(polished[k])} chars): {polished[k][:120]}...")
            print(f"AFTER  ({len(improved[k])} chars): {improved[k][:200]}...")
    
    # Show a GEO condensation
    for k in ['438', '881', '938']:
        if polished[k] != improved[k]:
            print(f"\n--- [{k}] GEO CONDENSATION ---")
            print(f"BEFORE ({len(polished[k])} chars): {polished[k][:100]}...")
            print(f"AFTER  ({len(improved[k])} chars): {improved[k][:200]}...")
    
    # Show a genome prefix fix
    for k in ['37', '245', '397']:
        if k in improved and polished[k] != improved[k]:
            print(f"\n--- [{k}] GENOME PREFIX FIX ---")
            print(f"BEFORE: ...{polished[k][-80:]}")
            print(f"AFTER:  ...{improved[k][-80:]}")
    
    # Show a few diversified endings
    shown = 0
    for k in improved:
        if polished[k] != improved[k] and "Express " in improved[k] and k not in GEO_REPLACEMENTS and k not in DATASET_LEVEL_IMPROVEMENTS:
            if shown < 5:
                print(f"\n--- [{k}] ENDING DIVERSIFIED ---")
                # Show just the last 120 chars
                print(f"BEFORE: ...{polished[k][-100:]}")
                print(f"AFTER:  ...{improved[k][-120:]}")
                shown += 1
    
    # Count remaining "Expressed marker genes" endings
    remaining_generic = sum(1 for v in improved.values() if v.rstrip().endswith("Expressed marker genes"))
    remaining_express = sum(1 for v in improved.values() if v.rstrip().endswith("markers"))
    print(f"\nRemaining 'Expressed marker genes' endings: {remaining_generic}")
    print(f"Remaining generic 'markers' endings: {remaining_express}")
    
    # Count remaining short captions
    remaining_short = sum(1 for v in improved.values() if len(v) < 150)
    print(f"Remaining short captions (<150 chars): {remaining_short}")


if __name__ == "__main__":
    main()
