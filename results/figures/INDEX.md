# Figure Panels

> Publication-ready figure panels (PDF format).

**Location**: `results/figures/`
**Panels**: 19 (A-S)
**Format**: PDF (publication quality)
**Last Updated**: 2025-03-08

---

## Quick Reference

| Panel | File | Description |
|-------|------|-------------|
| **A** | `panel_a_clop_training.pdf` | CLOP training curves |
| **B** | `panel_b_clop_embedding_umap.pdf` | CLOP embedding UMAP |
| **C** | `panel_c_umap_quality.pdf` | UMAP quality metrics |
| **D** | `panel_d_markers_per_type.pdf` | Markers per cell type |
| **E** | `panel_e_clop_metrics.pdf` | CLOP alignment metrics |
| **F** | `panel_f_type_composition.pdf` | Cell type composition |
| **G** | `panel_g_cell_generation.pdf` | Cell generation example |
| **H** | `panel_h_marker_heatmap.pdf` | Marker gene heatmap |
| **I** | `panel_i_umap_cell_types.pdf` | UMAP by cell type |
| **J** | `panel_j_diversity_metrics.pdf` | Diversity metrics |
| **K** | `panel_k_conditioning.pdf` | Text conditioning analysis |
| **L** | `panel_l_clustering.pdf` | Clustering analysis |
| **M** | `panel_m_classifier_performance.pdf` | Classifier performance |
| **N** | `panel_n_de_concordance.pdf` | DE concordance analysis |
| **O** | `panel_o_diversity_umap.pdf` | Diversity on UMAP |
| **P** | `panel_p_marker_genes.pdf` | Marker gene analysis |
| **Q** | `panel_q_downstream_tasks.pdf` | Downstream tasks |
| **R** | `panel_r_baseline_comparison.pdf` | Baseline comparison |
| **S** | `panel_s_cell_cell_transfer.pdf` | Cell-to-cell transfer |

---

## PNG Generation

PNG files are **generated on demand** from PDFs:

```bash
# Generate all PNGs
python results/figures/generate_pngs.py --all

# Generate specific panel
python results/figures/generate_pngs.py --panel A

# Generate with custom DPI
python results/figures/generate_pngs.py --panel A --dpi 300
```

**Generated PNGs**: Stored in `generated_pngs/` (gitignored)

---

## Figure Merged Report

| File | Size | Description |
|------|------|-------------|
| `clop_dit_full_report.pdf` | ~77MB | Complete 19-panel report |

---

## Regeneration

To regenerate all figures:

```bash
bash scripts/pipeline/regenerate_report.sh
```

See [operational/QUICK_START.md](../../docs/operational/QUICK_START.md) for full workflow.

---

## Storage Policy

- **PDFs**: Stored in git (publication quality, ~400KB total)
- **PNGs**: Generated on demand (~7.6MB if all generated)
- **generated_pngs/**: Gitignored runtime directory

---

## See Also

- [Figure Roadmap](../../docs/roadmaps/FIGURE_ENHANCEMENT_ROADMAP.md)
- [Figure Organization](../../docs/roadmaps/FIGURE_ORGANIZATION.md)
- [Quick Start](../../docs/operational/QUICK_START.md)
