# CLOP-DiT Figure Generation Workflow

**For future Claude Code sessions working on this repository.**
**Last updated: 2026-03-07**

---

## Overview

All figures are generated via Python/matplotlib. There is NO frontend/browser-based figure generation in the active pipeline. The legacy `arch-figure/` directory (Nuxt.js + Puppeteer) is dead code.

---

## Pipeline Steps

### 1. Evaluation (produces metrics JSONs)

```bash
python scripts/06_evaluate.py          # → results/archive/v5_final/final_evaluation.json
python scripts/compute_bootstrap_cis.py # → results/bootstrap_cis.json
```

### 2. Figure Generation (produces PNGs + PDFs)

**All article figures (except Fig 1):**
```bash
python -m src.visualization.results_visualizer --no-umap --dpi 300
```
- Produces: `results/figures/panel_*.{png,pdf}` and `results/figures/fig_*.{png,pdf}`
- VCD runs automatically via `save_with_vcd()` in `src/visualization/style.py`
- Uses `--no-umap` to skip slow UMAP recomputation (uses cached coordinates)

**Architecture figure (Fig 1) only:**
```bash
python scripts/generate_architecture_figure.py
```
- Produces: `results/figures/fig_architecture.{png,pdf}`

### 3. Article Delivery (symlinks to articles/figures/)

```bash
python -m src.visualization.article_delivery
```
- Creates/updates symlinks in `articles/figures/` pointing to `results/figures/`

### 4. LaTeX Compilation

```bash
cd articles && latexmk -pdf clop_dit_biology.tex
```

### 5. Full regeneration script

```bash
bash scripts/pipeline/regenerate_report.sh
```
- Runs steps 2 + 3 + 4 in sequence

---

## Key Files

| File | Purpose |
|------|---------|
| `src/visualization/style.py` | Central style: colors, fonts, `add_panel_label()`, `save_with_vcd()` |
| `src/visualization/results_visualizer.py` | Main orchestrator for all panels |
| `src/visualization/panels_*.py` | Individual panel generation functions |
| `src/visualization/panels_merged.py` | Merged/composite figure functions |
| `src/visualization/article_delivery.py` | Symlink management for articles/figures/ |
| `scripts/generate_architecture_figure.py` | Fig 1 architecture diagram |
| `scripts/vcd/` | Visual Conflict Detection modules (30 passes) |

## Style Constants (from style.py)

| Constant | Value | Usage |
|----------|-------|-------|
| `FONT_TITLE` | 11 | Subplot titles |
| `FONT_LABEL` | 10 | Axis labels |
| `FONT_TICK` | 10 | Tick labels |
| `FONT_TICK_DENSE` | 8 | Dense heatmap ticks |
| `FONT_LEGEND` | 10 | Standard legend |
| `FONT_LEGEND_DENSE` | 8 | Dense/multi-panel legend |
| `FONT_ANNOTATION` | 8 | Annotations |
| `FONT_SMALL` | 7 | Minimum readable text |
| Panel labels | 12 bold | Via `add_panel_label()` |

## VCD (Visual Conflict Detection)

- Runs automatically in `save_with_vcd()` when any figure is saved
- Manual check: `from src.visualization.style import run_vcd_check; run_vcd_check(fig)`
- Detects: text overlap, truncation, contrast issues, overplotting, font-size violations
- Config: `scripts/vcd/vcd_config.py`

## Figure-to-Article Mapping (15 figures)

| Fig | Content | Source Panel |
|-----|---------|-------------|
| 1 | Architecture diagram | fig_architecture |
| 2 | Training dynamics (2x4 grid) | fig_training_dynamics |
| 3 | Embedding space (2x3 grid) | fig_embedding_space |
| 4 | Metrics summary dashboard | panel_d |
| 5 | Fidelity + alignment (merged G+F) | fig_fidelity_alignment |
| 6 | Marker gene comparison | panel_n |
| 7 | Expression correlation | panel_h |
| 8 | Expression analysis | panel_i |
| 9 | Conditioning UMAP | panel_m |
| 10 | Diversity diagnostics | panel_j |
| 11 | Noise-scale tradeoff | panel_l |
| 12 | Baseline comparison | panel_o |
| 13 | Benchmark | panel_s |
| 14 | Downstream clustering+classifier (2x3) | fig_downstream_pq |
| 15 | DE concordance | panel_r |

## GPU

- NVIDIA GeForce RTX 5090 Laptop GPU, 24 GB VRAM
- CUDA 13.0
- PyTorch 2.1
