#!/usr/bin/env python3
# 02b_enrich_descriptions.py — Evidence-based text description enrichment
"""
CLOP-DiT v6.2: Enrich sub-cluster text descriptions with biological evidence.

Problem:
    Current template texts have LOW ENTROPY — 42 "Monocyte" clusters map to
    nearly identical BiomedBERT embeddings despite biological differences.
    Text descriptions average 284 chars with only a few marker genes.

Solution:
    Extract rich biological evidence per cluster and generate information-dense
    natural language descriptions that maximize embedding discriminability.

Evidence sources:
    1. Marker genes (existing) — top DE genes + expression statistics
    2. Anti-markers — genes significantly downregulated vs rest
    3. MSigDB Hallmark pathways — 50 cancer/immune/metabolic hallmarks
    4. KEGG pathway enrichment — metabolic and signaling pathways
    5. GO Biological Process — functional annotations
    6. Transcription factors — regulatory context
    7. Cell Ontology mapping — standardized cell type terms

Output:
    data/processed_h5ad/subcluster_metadata_enriched.json
    — Same structure as subcluster_metadata.json but with:
      • enriched 'text' field (500-800 chars, evidence-dense)
      • 'text_variants' list (3-5 caption variants per cluster)
      • 'evidence' dict (structured biological evidence)

Usage:
    python scripts/02b_enrich_descriptions.py \
        --h5ad_dir data/processed_h5ad \
        --subcluster_meta data/processed_h5ad/subcluster_metadata.json \
        --msigdb_dir ../datasets/msigdb \
        --output data/processed_h5ad/subcluster_metadata_enriched.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import scanpy as sc
import scipy.stats as stats
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ============================================================================
#  Gene Set Database Loader
# ============================================================================

class GeneSetDB:
    """Load and query MSigDB GMT gene sets."""

    def __init__(self, msigdb_dir: str = "../datasets/msigdb"):
        self.msigdb_dir = Path(msigdb_dir)
        self.hallmark_hs = {}    # Hallmark human
        self.hallmark_mm = {}    # Hallmark mouse
        self.kegg_hs = {}        # KEGG human
        self.canonical_hs = {}   # Canonical pathways human
        self.canonical_mm = {}   # Canonical pathways mouse
        self.go_bp_hs = {}       # GO BP human
        self.go_bp_mm = {}       # GO BP mouse
        self.tfs_mm = set()      # Mouse TFs
        self._load_all()

    def _parse_gmt(self, path: Path) -> Dict[str, Set[str]]:
        """Parse a GMT file into {pathway_name: set(genes)}."""
        gene_sets = {}
        if not path.exists():
            logger.warning(f"GMT file not found: {path}")
            return gene_sets
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0]
                genes = set(parts[2:])  # skip URL in parts[1]
                gene_sets[name] = genes
        return gene_sets

    def _load_all(self):
        """Load all available gene set databases."""
        d = self.msigdb_dir

        self.hallmark_hs = self._parse_gmt(d / "h.all.v2024.1.Hs.symbols.gmt")
        self.hallmark_mm = self._parse_gmt(d / "mh.all.v2024.1.Mm.symbols.gmt")
        self.kegg_hs = self._parse_gmt(d / "c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt")
        self.canonical_hs = self._parse_gmt(d / "c2.cp.v2024.1.Hs.symbols.gmt")
        self.canonical_mm = self._parse_gmt(d / "m2.cp.v2024.1.Mm.symbols.gmt")
        self.go_bp_hs = self._parse_gmt(d / "c5.go.bp.v2024.1.Hs.symbols.gmt")
        self.go_bp_mm = self._parse_gmt(d / "m5.go.bp.v2024.1.Mm.symbols.gmt")

        # Load TF lists
        tf_file = d / "allTFs_mm.txt"
        if tf_file.exists():
            with open(tf_file) as f:
                self.tfs_mm = set(line.strip() for line in f if line.strip())

        n_total = sum(len(x) for x in [
            self.hallmark_hs, self.hallmark_mm, self.kegg_hs,
            self.canonical_hs, self.canonical_mm,
            self.go_bp_hs, self.go_bp_mm,
        ])
        logger.info(f"Loaded {n_total} gene sets from MSigDB, {len(self.tfs_mm)} TFs")

    def enrich(
        self,
        marker_genes: List[str],
        organism: str = "human",
        background_size: int = 2000,
        top_k: int = 5,
        pval_threshold: float = 0.05,
    ) -> Dict[str, List[Tuple[str, float, int]]]:
        """Run hypergeometric enrichment for marker genes against multiple databases.

        Parameters
        ----------
        marker_genes : list of str
            DE marker genes for a cluster (typically top 20-50).
        organism : str
            'human' or 'mouse'.
        background_size : int
            Total genes tested (HVGs in the dataset, typically 2000).
        top_k : int
            Return top-k enriched pathways per database.
        pval_threshold : float
            Significance threshold.

        Returns
        -------
        results : dict
            {'hallmark': [(name, pval, overlap_count), ...],
             'kegg': [...], 'go_bp': [...], 'canonical': [...]}
        """
        marker_set = set(g.upper() for g in marker_genes)

        if organism == "mouse":
            # Mouse gene symbols: capitalize first letter only (e.g., Cd8a)
            # MSigDB mouse GMT uses this format, but we uppercase for matching
            dbs = {
                "hallmark": self.hallmark_mm,
                "canonical": self.canonical_mm,
                "go_bp": self.go_bp_mm,
            }
        else:
            dbs = {
                "hallmark": self.hallmark_hs,
                "kegg": self.kegg_hs,
                "canonical": self.canonical_hs,
                "go_bp": self.go_bp_hs,
            }

        results = {}
        for db_name, gene_sets in dbs.items():
            hits = []
            for pathway_name, pathway_genes in gene_sets.items():
                pathway_upper = set(g.upper() for g in pathway_genes)
                overlap = marker_set & pathway_upper
                if len(overlap) < 2:
                    continue

                # Hypergeometric test
                # P(X >= k) where X ~ Hypergeometric(N, K, n)
                # N = background, K = pathway size, n = marker list size, k = overlap
                N = background_size
                K = min(len(pathway_upper), N)
                n = len(marker_set)
                k = len(overlap)

                pval = stats.hypergeom.sf(k - 1, N, K, n)
                if pval < pval_threshold:
                    hits.append((pathway_name, pval, k, sorted(overlap)))

            # Sort by p-value and take top_k
            hits.sort(key=lambda x: x[1])
            results[db_name] = [
                (name, float(pval), count, genes)
                for name, pval, count, genes in hits[:top_k]
            ]

        return results

    def get_tfs(self, marker_genes: List[str], organism: str = "mouse") -> List[str]:
        """Identify transcription factors among marker genes."""
        if organism == "mouse":
            tf_set = self.tfs_mm
        else:
            # Use mouse TFs but uppercase for matching
            tf_set = set(g.upper() for g in self.tfs_mm)

        tfs = []
        for g in marker_genes:
            g_check = g if organism == "mouse" else g.upper()
            if g_check in tf_set:
                tfs.append(g)
        return tfs


# ============================================================================
#  Anti-marker Extraction
# ============================================================================

def get_anti_markers(
    adata: sc.AnnData,
    cluster_cells: List[int],
    n_anti: int = 10,
) -> List[Tuple[str, float]]:
    """Find genes significantly DOWNregulated in cluster vs rest.

    Returns list of (gene_name, log_fold_change) tuples, sorted by strongest
    downregulation (most negative logFC first).
    """
    n_cells = adata.shape[0]
    if len(cluster_cells) < 5 or len(cluster_cells) > n_cells - 5:
        return []

    # Create cluster labels
    labels = np.zeros(n_cells, dtype=int)
    labels[cluster_cells] = 1

    # Get expression matrix
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # Compute mean expression in cluster vs rest
    cluster_mask = labels == 1
    mean_cluster = X[cluster_mask].mean(axis=0)
    mean_rest = X[~cluster_mask].mean(axis=0)

    # Log fold change (add pseudocount)
    eps = 1e-4
    logfc = np.log2((mean_cluster + eps) / (mean_rest + eps))

    # Get most downregulated genes
    sorted_idx = np.argsort(logfc)
    anti_markers = []
    for idx in sorted_idx[:n_anti]:
        gene = adata.var_names[idx]
        lfc = float(logfc[idx])
        if lfc < -0.5:  # Only include clearly downregulated genes
            anti_markers.append((gene, lfc))

    return anti_markers


# ============================================================================
#  Expression Statistics
# ============================================================================

def get_expression_stats(
    adata: sc.AnnData,
    cluster_cells: List[int],
    marker_genes: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute expression statistics for marker genes in a cluster.

    Returns dict of {gene: {mean_expr, pct_expressed, specificity, logfc}}.
    """
    n_cells = adata.shape[0]
    cluster_mask = np.zeros(n_cells, dtype=bool)
    cluster_mask[cluster_cells] = True

    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    stats_dict = {}
    for gene in marker_genes:
        idx = gene_to_idx.get(gene)
        if idx is None:
            continue

        expr_cluster = X[cluster_mask, idx]
        expr_rest = X[~cluster_mask, idx]

        mean_c = float(expr_cluster.mean())
        mean_r = float(expr_rest.mean())
        pct_expr = float((expr_cluster > 0).mean())
        pct_rest = float((expr_rest > 0).mean())

        eps = 1e-4
        logfc = float(np.log2((mean_c + eps) / (mean_r + eps)))
        # Specificity: fraction of total expression coming from this cluster
        total_expr = float(X[:, idx].sum())
        if total_expr > 0:
            specificity = float(expr_cluster.sum() / total_expr)
        else:
            specificity = 0.0

        stats_dict[gene] = {
            "mean_expr": round(mean_c, 3),
            "logfc": round(logfc, 2),
            "pct_expressed": round(pct_expr, 3),
            "pct_rest": round(pct_rest, 3),
            "specificity": round(specificity, 3),
        }

    return stats_dict


# ============================================================================
#  Enriched Caption Generator
# ============================================================================

# Cleaned pathway names (remove MSigDB prefixes)
def _clean_pathway_name(name: str) -> str:
    """Convert MSigDB pathway name to readable form."""
    # Remove common prefixes
    for prefix in [
        "HALLMARK_", "KEGG_", "REACTOME_", "BIOCARTA_", "PID_", "WP_",
        "GOBP_", "GOCC_", "GOMF_", "HP_",
        "KEGG_MEDICUS_", "NABA_",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Replace underscores with spaces and title-case
    return name.replace("_", " ").lower()


def generate_enriched_caption(
    cluster_info: Dict,
    evidence: Dict,
    organism: str,
    tissue: str,
    disease: str,
    dataset_context: str,
) -> str:
    """Generate an information-dense primary caption from biological evidence.

    Target: 500-800 characters with specific marker genes, pathways, expression
    patterns, and biological context. Maximizes BiomedBERT embedding discriminability.

    Returns a single enriched description string.
    """
    cell_type = cluster_info["cell_type"]
    confidence = cluster_info["confidence"]
    n_cells = cluster_info["n_cells"]
    cluster_id = cluster_info.get("cluster_id", "?")
    top_markers = cluster_info.get("top_markers", [])

    # Get evidence components
    expr_stats = evidence.get("expression_stats", {})
    anti_markers = evidence.get("anti_markers", [])
    pathways = evidence.get("pathways", {})
    tfs = evidence.get("transcription_factors", [])

    # ── Build caption components ──

    # 1. Opening: organism + tissue + disease + cell type
    if cell_type == "Unknown":
        cell_desc = "uncharacterized cell population"
    elif confidence >= 0.50:
        cell_desc = f"{cell_type}"
    elif confidence >= 0.30:
        cell_desc = f"putative {cell_type}"
    else:
        cell_desc = f"candidate {cell_type}"

    disease_str = f" in {disease} context" if disease not in ("healthy", "normal", "unknown") else ""

    # 2. Marker genes with expression stats
    marker_parts = []
    for gene in top_markers[:6]:
        st = expr_stats.get(gene, {})
        if st:
            logfc_str = f"logFC={st['logfc']}" if st.get('logfc') else ""
            pct_str = f"{st['pct_expressed']*100:.0f}%" if st.get('pct_expressed') else ""
            specificity_str = f"spec={st['specificity']:.2f}" if st.get('specificity') else ""
            parts = [logfc_str, pct_str, specificity_str]
            parts = [p for p in parts if p]
            if parts:
                marker_parts.append(f"{gene} ({', '.join(parts[:2])})")
            else:
                marker_parts.append(gene)
        else:
            marker_parts.append(gene)
    marker_str = ", ".join(marker_parts)

    # 3. Anti-markers (top 3)
    anti_str = ""
    if anti_markers:
        anti_genes = [f"{g} (logFC={lfc:.1f})" for g, lfc in anti_markers[:3]]
        anti_str = f" Downregulated: {', '.join(anti_genes)}."

    # 4. Pathway enrichment
    pathway_parts = []
    for db_name in ["hallmark", "kegg", "canonical"]:
        for pw_name, pval, count, genes in pathways.get(db_name, [])[:2]:
            clean = _clean_pathway_name(pw_name)
            pathway_parts.append(f"{clean} (p={pval:.1e}, {count} genes)")
    pathway_str = ""
    if pathway_parts:
        pathway_str = f" Enriched pathways: {'; '.join(pathway_parts[:3])}."

    # 5. GO terms (top 2, shorter format)
    go_parts = []
    for pw_name, pval, count, genes in pathways.get("go_bp", [])[:2]:
        clean = _clean_pathway_name(pw_name)
        # Truncate long GO terms
        if len(clean) > 50:
            clean = clean[:47] + "..."
        go_parts.append(clean)
    go_str = ""
    if go_parts:
        go_str = f" GO processes: {'; '.join(go_parts)}."

    # 6. Transcription factors
    tf_str = ""
    if tfs:
        tf_str = f" Active TFs: {', '.join(tfs[:3])}."

    # ── Assemble full caption ──
    caption = (
        f"{organism.capitalize()} {tissue}{disease_str}: "
        f"{cell_desc} (n={n_cells}, cluster {cluster_id}) "
        f"expressing {marker_str}."
        f"{anti_str}"
        f"{pathway_str}"
        f"{go_str}"
        f"{tf_str}"
        f" Source: {dataset_context}"
    )

    return caption


def generate_caption_variants(
    cluster_info: Dict,
    evidence: Dict,
    organism: str,
    tissue: str,
    disease: str,
    dataset_context: str,
) -> List[str]:
    """Generate 3-5 caption variants for a cluster to increase text diversity.

    Each variant emphasizes different aspects of the same cluster:
    1. Primary: Full evidence-based caption (from generate_enriched_caption)
    2. Marker-centric: Focus on gene expression patterns
    3. Pathway-centric: Focus on functional annotations
    4. Ontology-centric: Focus on cell identity and context
    5. Comparative: Focus on what distinguishes this cluster (anti-markers)
    """
    cell_type = cluster_info["cell_type"]
    confidence = cluster_info["confidence"]
    n_cells = cluster_info["n_cells"]
    cluster_id = cluster_info.get("cluster_id", "?")
    top_markers = cluster_info.get("top_markers", [])

    expr_stats = evidence.get("expression_stats", {})
    anti_markers = evidence.get("anti_markers", [])
    pathways = evidence.get("pathways", {})
    tfs = evidence.get("transcription_factors", [])

    variants = []

    # ── Variant 1: Primary (full evidence) ──
    variants.append(generate_enriched_caption(
        cluster_info, evidence, organism, tissue, disease, dataset_context
    ))

    # ── Variant 2: Marker-centric ──
    if cell_type == "Unknown":
        cell_desc = "uncharacterized cells"
    else:
        cell_desc = cell_type

    marker_details = []
    for gene in top_markers[:8]:
        st = expr_stats.get(gene, {})
        if st and st.get("logfc"):
            pct = st.get("pct_expressed", 0) * 100
            marker_details.append(f"{gene} (logFC={st['logfc']}, {pct:.0f}% cells)")
        else:
            marker_details.append(gene)

    v2 = (
        f"Gene expression profile of {cell_desc} from {organism} {tissue}: "
        f"Upregulated markers include {', '.join(marker_details)}."
    )
    if anti_markers:
        anti_genes = [f"{g}" for g, _ in anti_markers[:5]]
        v2 += f" Notably absent: {', '.join(anti_genes)}."
    v2 += f" Cluster of {n_cells} cells from {dataset_context}"
    variants.append(v2)

    # ── Variant 3: Pathway-centric ──
    pw_details = []
    for db_name in ["hallmark", "kegg", "canonical", "go_bp"]:
        for pw_name, pval, count, genes in pathways.get(db_name, [])[:2]:
            clean = _clean_pathway_name(pw_name)
            gene_str = ", ".join(genes[:3])
            pw_details.append(f"{clean} ({gene_str}; p={pval:.1e})")

    if pw_details:
        v3 = (
            f"Functional characterization of {cell_desc} subpopulation "
            f"in {organism} {tissue}: "
            f"Pathway analysis reveals enrichment for {'; '.join(pw_details[:4])}."
        )
        if tfs:
            v3 += f" Key transcription factors: {', '.join(tfs[:3])}."
        v3 += f" n={n_cells} cells from {dataset_context}"
        variants.append(v3)

    # ── Variant 4: Ontology + context ──
    disease_ctx = f" with {disease}" if disease not in ("healthy", "normal", "unknown") else ""
    v4 = (
        f"Single-cell transcriptomic cluster from {organism} {tissue} tissue{disease_ctx}. "
        f"Cell identity: {cell_desc} (confidence={confidence:.2f}). "
        f"Defined by {', '.join(top_markers[:5])} expression."
    )
    if pathways.get("hallmark"):
        hall = [_clean_pathway_name(n) for n, _, _, _ in pathways["hallmark"][:2]]
        v4 += f" Hallmark programs: {', '.join(hall)}."
    v4 += f" Dataset: {dataset_context}"
    variants.append(v4)

    # ── Variant 5: Comparative (anti-markers + specificity) ──
    if anti_markers and len(anti_markers) >= 2:
        specific_genes = []
        for gene in top_markers[:5]:
            st = expr_stats.get(gene, {})
            if st.get("specificity", 0) > 0.3:
                specific_genes.append(f"{gene} (spec={st['specificity']:.2f})")

        anti_genes = [f"{g}" for g, lfc in anti_markers[:4]]
        v5 = (
            f"Distinguishing features of {cell_desc} cluster in {organism} {tissue}: "
        )
        if specific_genes:
            v5 += f"Highly specific markers: {', '.join(specific_genes)}. "
        v5 += f"Negative markers (absent): {', '.join(anti_genes)}. "
        v5 += f"n={n_cells} cells, cluster {cluster_id}. "
        v5 += f"Source: {dataset_context}"
        variants.append(v5)

    return variants


# ============================================================================
#  Context Parsing (from dataset text/ID)
# ============================================================================

def parse_organism(dataset_text: str, dataset_id: str) -> str:
    """Detect organism from text and ID."""
    text_lower = dataset_text.lower()
    id_lower = dataset_id.lower()

    if any(x in text_lower for x in ["human", "homo sapiens", "patient", "donor"]):
        return "human"
    if "hm" in id_lower and "mm" not in id_lower:
        return "human"
    if any(x in text_lower for x in ["mouse", "murine", "mus musculus"]):
        return "mouse"
    if "mm" in id_lower:
        return "mouse"
    if any(x in text_lower for x in ["patients", "clinical", "therapy"]):
        return "human"
    if any(x in text_lower for x in ["c57bl", "balb", "transgenic mice"]):
        return "mouse"
    return "unknown"


def parse_tissue(dataset_text: str, dataset_id: str) -> str:
    """Detect tissue from text and ID."""
    text_lower = dataset_text.lower()
    tissue_map = [
        (["lung", "pulmonary", "alveolar", "airway"], "lung"),
        (["brain", "cortex", "cerebr", "hippocampus", "dentate", "forebrain"], "brain"),
        (["liver", "hepat"], "liver"),
        (["kidney", "renal"], "kidney"),
        (["pancrea", "islet"], "pancreas"),
        (["bone marrow", "bm "], "bone marrow"),
        (["blood", "pbmc", "peripheral"], "peripheral blood"),
        (["skin", "cutaneous", "derma"], "skin"),
        (["intestin", "colon", "gut", "ileum"], "intestine"),
        (["breast", "mammary"], "breast"),
        (["heart", "cardiac"], "heart"),
        (["retina", "eye", "ocular"], "retina"),
        (["pituitary"], "pituitary"),
        (["thymus"], "thymus"),
        (["spleen", "splenic"], "spleen"),
        (["tumor", "cancer", "carcinoma"], "tumor"),
        (["prostate"], "prostate"),
        (["ovary", "ovarian"], "ovary"),
        (["uterus", "endometr"], "uterus"),
        (["muscle", "skeletal muscle"], "muscle"),
        (["adipose", "fat tissue"], "adipose tissue"),
        (["urethra", "bladder", "urinary"], "urethra/bladder"),
    ]
    for keywords, tissue in tissue_map:
        if any(kw in text_lower for kw in keywords):
            return tissue
    return "tissue"


def parse_disease(dataset_text: str, dataset_id: str) -> str:
    """Detect disease context from text and ID."""
    text_lower = dataset_text.lower()
    id_lower = dataset_id.lower()
    if any(x in id_lower for x in ["cancer", "tumor"]) or \
       any(x in text_lower for x in ["cancer", "carcinoma", "tumor", "malignant", "leukemia", "lymphoma", "melanoma"]):
        return "cancer"
    if any(x in text_lower for x in ["fibrosis", "fibrotic"]):
        return "fibrosis"
    if any(x in text_lower for x in ["inflamm", "lps", "infection", "sepsis"]):
        return "inflammation"
    if any(x in text_lower for x in ["aging", "aged"]):
        return "aging"
    if any(x in id_lower for x in ["dev"]) or \
       any(x in text_lower for x in ["development", "embryo", "fetal"]):
        return "development"
    return "healthy"


def get_dataset_context_summary(dataset_text: str) -> str:
    """Extract a full context summary from the dataset description.

    Returns the complete dataset description (cleaned of boilerplate prefixes)
    without any truncation — BiomedBERT handles up to 512 tokens (~2000 chars).
    """
    ctx = dataset_text.strip()

    # Strip boilerplate prefixes (keep the informative content)
    for prefix in [
        "Single-cell RNA sequencing of ",
        "Single-cell RNA-seq of ",
        "scRNA-seq of ",
        "Single-cell RNA sequencing analysis of ",
    ]:
        if ctx.lower().startswith(prefix.lower()):
            ctx = ctx[len(prefix):]
            break

    # Ensure the text ends with a period
    if ctx and not ctx.endswith("."):
        ctx += "."

    return ctx


# ============================================================================
#  Main Enrichment Pipeline
# ============================================================================

def enrich_all_clusters(
    h5ad_dir: str,
    subcluster_meta_file: str,
    msigdb_dir: str,
    output_file: str,
    n_anti_markers: int = 10,
    n_de_genes: int = 30,
):
    """Run the full evidence extraction and caption enrichment pipeline.

    For each cluster:
    1. Load the h5ad file
    2. Run Leiden clustering to get the same cluster assignments
    3. Extract anti-markers, expression stats, pathway enrichment
    4. Generate enriched captions + variants
    """
    h5ad_dir = Path(h5ad_dir)
    gsdb = GeneSetDB(msigdb_dir)

    with open(subcluster_meta_file) as f:
        sub_meta = json.load(f)

    logger.info(f"Enriching {len(sub_meta)} datasets with biological evidence...")

    enriched = {}
    total_clusters = 0
    total_variants = 0

    for ds_idx, (ds_id, ds_info) in enumerate(sub_meta.items()):
        logger.info(f"[{ds_idx+1}/{len(sub_meta)}] {ds_id} ({ds_info['n_clusters']} clusters)")

        # Find h5ad file
        h5ad_path = h5ad_dir / f"{ds_id}_processed.h5ad"
        if not h5ad_path.exists():
            # Try alternative naming
            candidates = list(h5ad_dir.glob(f"*{ds_id}*processed*.h5ad"))
            if candidates:
                h5ad_path = candidates[0]
            else:
                logger.warning(f"  h5ad not found for {ds_id}, using text-only enrichment")
                h5ad_path = None

        # Load h5ad (needed for anti-markers and expression stats)
        adata = None
        if h5ad_path and h5ad_path.exists():
            try:
                adata = sc.read_h5ad(h5ad_path)
            except Exception as e:
                logger.warning(f"  Failed to load {h5ad_path}: {e}")

        # Parse context
        dataset_text = ds_info["dataset_text"]
        organism = parse_organism(dataset_text, ds_id)
        tissue = parse_tissue(dataset_text, ds_id)
        disease = parse_disease(dataset_text, ds_id)
        context_summary = get_dataset_context_summary(dataset_text)

        enriched_ds = {
            "dataset_text": dataset_text,
            "n_cells_total": ds_info["n_cells_total"],
            "n_clusters": ds_info["n_clusters"],
            "resolution": ds_info["resolution"],
            "resolution_mode": ds_info.get("resolution_mode", "fixed"),
            "organism": organism,
            "tissue": tissue,
            "disease": disease,
            "clusters": {},
        }

        for cid, cinfo in ds_info["clusters"].items():
            cell_indices = cinfo.get("cell_indices", [])
            top_markers = cinfo.get("top_markers", [])

            # ── Evidence extraction ──
            evidence = {
                "expression_stats": {},
                "anti_markers": [],
                "pathways": {},
                "transcription_factors": [],
            }

            # Expression stats + anti-markers (need h5ad)
            if adata is not None and cell_indices:
                # Filter valid indices
                valid_idx = [i for i in cell_indices if i < adata.shape[0]]
                if valid_idx:
                    evidence["expression_stats"] = get_expression_stats(
                        adata, valid_idx, top_markers[:n_de_genes]
                    )
                    evidence["anti_markers"] = get_anti_markers(
                        adata, valid_idx, n_anti=n_anti_markers
                    )

            # Pathway enrichment (marker genes only — no h5ad needed)
            if top_markers:
                evidence["pathways"] = gsdb.enrich(
                    top_markers[:n_de_genes],
                    organism=organism,
                    background_size=2000,  # our HVG count
                    top_k=5,
                    pval_threshold=0.05,
                )

            # TFs
            if top_markers:
                evidence["transcription_factors"] = gsdb.get_tfs(
                    top_markers[:n_de_genes], organism=organism
                )

            # ── Generate enriched captions ──
            cluster_info_ext = {
                "cell_type": cinfo["cell_type"],
                "confidence": cinfo["confidence"],
                "n_cells": cinfo["n_cells"],
                "cluster_id": cid,
                "top_markers": top_markers,
                "label_source": cinfo.get("label_source", "unknown"),
            }

            enriched_text = generate_enriched_caption(
                cluster_info_ext, evidence,
                organism, tissue, disease, context_summary,
            )

            text_variants = generate_caption_variants(
                cluster_info_ext, evidence,
                organism, tissue, disease, context_summary,
            )

            # Store enriched cluster info
            enriched_ds["clusters"][cid] = {
                "cell_type": cinfo["cell_type"],
                "confidence": cinfo["confidence"],
                "n_cells": cinfo["n_cells"],
                "top_markers": top_markers,
                "cell_indices": cell_indices,
                "label_source": cinfo.get("label_source", "unknown"),
                # Original text for comparison
                "text_original": cinfo["text"],
                # Enriched primary text
                "text": enriched_text,
                # Caption variants
                "text_variants": text_variants,
                # Structured evidence
                "evidence": evidence,
            }

            total_clusters += 1
            total_variants += len(text_variants)

        enriched[ds_id] = enriched_ds

    # Save
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    logger.info(
        f"Enrichment complete: {total_clusters} clusters, "
        f"{total_variants} total caption variants. "
        f"Saved to {output_path}"
    )

    # Print comparison stats
    logger.info("=== Caption Quality Comparison ===")
    orig_lengths = []
    new_lengths = []
    variant_counts = []
    for ds_id, ds_info in enriched.items():
        for cid, cinfo in ds_info["clusters"].items():
            orig_lengths.append(len(cinfo["text_original"]))
            new_lengths.append(len(cinfo["text"]))
            variant_counts.append(len(cinfo["text_variants"]))

    logger.info(f"  Original text length: mean={np.mean(orig_lengths):.0f} chars")
    logger.info(f"  Enriched text length: mean={np.mean(new_lengths):.0f} chars")
    logger.info(f"  Caption variants/cluster: mean={np.mean(variant_counts):.1f}")
    logger.info(f"  Total unique captions: {total_variants}")

    return enriched


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enrich sub-cluster descriptions with biological evidence")
    parser.add_argument("--h5ad_dir", default="data/processed_h5ad",
                       help="Directory containing processed h5ad files")
    parser.add_argument("--subcluster_meta", default="data/processed_h5ad/subcluster_metadata.json",
                       help="Sub-cluster metadata from 02_subcluster_descriptions.py")
    parser.add_argument("--msigdb_dir", default="../datasets/msigdb",
                       help="MSigDB gene set directory")
    parser.add_argument("--output", default="data/processed_h5ad/subcluster_metadata_enriched.json",
                       help="Output enriched metadata file")
    parser.add_argument("--n_anti_markers", type=int, default=10,
                       help="Number of anti-markers to extract per cluster")
    parser.add_argument("--n_de_genes", type=int, default=30,
                       help="Number of DE genes to use for enrichment")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=level)

    enrich_all_clusters(
        h5ad_dir=args.h5ad_dir,
        subcluster_meta_file=args.subcluster_meta,
        msigdb_dir=args.msigdb_dir,
        output_file=args.output,
        n_anti_markers=args.n_anti_markers,
        n_de_genes=args.n_de_genes,
    )


if __name__ == "__main__":
    main()
