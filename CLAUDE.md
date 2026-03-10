# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start — Installation and Common Commands

### Installation
```bash
# Create environment and install
conda create -n clopdit python=3.10
conda activate clopdit
pip install -e .

# Optional: additional dependencies for specific features
pip install wandb tensorboard  # experiment tracking
pip install unsloth bitsandbytes  # scGPT fine-tuning (CUDA required)
```

### Most Common Commands

**Regenerate all article figures** (Figs 1–20)
```bash
cd /home/zeyufu/Desktop/CLOP-DiT
python scripts/pipeline/run_regeneration.py      # Generate PNG + PDF with VCD checks
```

**Build the LaTeX article**
```bash
bash scripts/pipeline/build_article.sh            # Verify figures and build PDF
bash scripts/pipeline/build_article.sh --check    # Verify figures only
```

**Run tests**
```bash
pytest tests/                                      # Run all tests
pytest tests/test_visualization.py                 # Run visualization tests
pytest tests/ -v                                   # Verbose output
```

**Generate figures from source scripts** (a la carte)
```bash
python scripts/analysis/generate_architecture_figure.py    # Fig 1
python scripts/analysis/evaluation_pipeline_figure.py       # Fig 2
python -m src.visualization.results_visualizer             # Figs 3–18
python scripts/analysis/variance_matching_pilot.py         # Fig 19
python scripts/analysis/gene_gene_correlation.py           # Fig 20
```

**Full pipeline orchestration**
```bash
python scripts/pipeline/run_pipeline.py --stage all                  # All stages
python scripts/pipeline/run_pipeline.py --stage figures --build-article
python scripts/pipeline/run_pipeline.py --from generate              # Generate through article_delivery
```

---

## Architecture Overview

### High-Level Design

CLOP-DiT has a modular three-stage architecture that generates realistic gene expression profiles from natural language text. The pipeline is implemented as:

1. **Stage 1: CLOP Aligner**
   BiomedBERT text embeddings (1024-d) → ZCA whitening → CLOP aligner → shared 512-d space

2. **Stage 2: DiT (Diffusion Transformer)**
   - Learns conditional distribution of cell embeddings
   - Uses 1D flow matching with classifier-free guidance
   - Output: synthetic 512-d embeddings

3. **Stage 3: scGPT Decoder**
   - Maps embeddings back to per-gene expression
   - Uses native scGPT `generate()` pathway

### Figure Generation Architecture

Publication figures follow a **centralized manifest-based delivery system**:

```
Figure Source Scripts (generate PNG/PDF)
    ↓
results/figures/fig_*.pdf
    ↓
src/visualization/article_delivery.py (single source of truth)
    ↓
articles/figures/fig{NN}_*.pdf (symlinks)
    ↓
articles/clop_dit_biology.tex (\includegraphics)
```

**Key files:**
- **`src/visualization/article_delivery.py`** — Canonical 20-figure manifest (single source of truth)
  - Maps source filenames → article figure numbers
  - Verifies PDFs exist and creates symlinks to `articles/figures/`
- **`src/visualization/style.py`** — Global style management
  - rcParams (Nature/Cell conventions, fonts, colors)
  - `apply_style()` called by all figure scripts
  - `save_with_vcd()` saves PNG + PDF with visual conflict detection
  - Helper functions: `add_panel_label()`, `add_colorbar_safe()`, `style_axes()`
- **`src/visualization/results_visualizer.py`** — Orchestrator for Figs 3–18
  - Loads cached results and calls individual `fig0X_*.py` modules
- **Individual figure modules** — `fig0X_*.py` for Figs 3–18
- **External scripts** — Fig 1 (`generate_architecture_figure.py`), Fig 2 (`evaluation_pipeline_figure.py`), Figs 19–20 (`variance_matching_pilot.py`, `gene_gene_correlation.py`)

### Path Configuration

**Centralized via `src/utils/paths.py`:**
- Resolves paths in order: environment variables → `configs/pipeline.yaml` → defaults
- All scripts import from `paths.py`, not hardcoded paths
- Override with env vars: `CLOPDIT_FIG_DIR`, `CLOPDIT_RESULTS_DIR`, `CLOPDIT_CACHE_DIR`, etc.

**Key path constants:**
```python
from src.utils.paths import (
    FIG_DIR,                    # results/figures/
    RESULTS_DIR,                # results/
    ARTICLE_FIGURES_DIR,        # articles/figures/
    CACHE_DIR,                  # data/cache/
    CHECKPOINT_DIR,             # models/checkpoints/
    PROCESSED_H5AD_DIR,         # data/processed_h5ad/
)
```

### Data Flow

```
configs/pipeline.yaml (centralized paths)
    ↓
src/utils/paths.py (resolution logic)
    ↓
Data prep (00-03): data/ → cache/
    ↓
Training (04a-b): cache/ → models/checkpoints/
    ↓
Generation (generate_embeddings): embeddings → JSON
    ↓
Analysis (decode, diversity, conditioning): JSON → results/
    ↓
Figures (individual scripts + ResultsVisualizer): results/ → figures/
    ↓
Article delivery (article_delivery.py): figures/ → articles/figures/ (symlinks)
    ↓
LaTeX build: articles/clop_dit_biology.tex → PDF
```

---

## Figure Generation Workflow

### When to regenerate figures

- After modifying figure source code (`fig0X_*.py`, `generate_*.py`, `variance_*.py`, `gene_*.py`)
- After modifying global style settings in `src/visualization/style.py`
- When rebuilding the article PDF

### Full regeneration process

1. **Regenerate all 20 figures with VCD checks:**
   ```bash
   python scripts/pipeline/run_regeneration.py  # Auto-runs VCD on every figure
   ```

2. **Check VCD report for text overlaps:**
   ```bash
   cat results/vcd_report_summary.md  # Human-readable summary
   cat results/vcd_report.json        # Detailed per-figure data
   ```

3. **Fix any VCD violations** by modifying source scripts, then regenerate.

4. **Build article PDF:**
   ```bash
   bash scripts/pipeline/build_article.sh
   ```

5. **Verify LaTeX compilation** produced `articles/clop_dit_biology.pdf`

### Single figure debugging

To regenerate and inspect one figure without running the full pipeline:

```bash
# Edit the figure script, then run it directly:
python src/visualization/fig07_alignment.py  # For Fig 7

# Or via the ResultsVisualizer with a single figure:
python -c "
from src.visualization.results_visualizer import ResultsVisualizer
rv = ResultsVisualizer()
rv.fig07_text_cell_alignment()
"
```

---

## Common Development Tasks

### Adding a new figure

1. Create `src/visualization/figXX_name.py` with a plot function
2. Import and call in `src/visualization/results_visualizer.py` (method `figXX_*()`)
3. Add entry to `article_delivery.py` manifest
4. Export PNG + PDF using `save_with_vcd(fig, path)` from `style.py`

### Modifying global style

Edit `src/visualization/style.py`:
- **rcParams:** `VIS_STYLE` dict (font sizes, colors, grid, etc.)
- **Colors:** `COLORS` dict (semantic names for real/generated/baseline)
- **Font constants:** `FONT_*` (e.g., `FONT_LEGEND`, `FONT_SMALL`)

Then regenerate all figures to apply globally.

### Fixing text overlap issues

Use **Visual Conflict Detection (VCD)** in `scripts/vcd/`:
- `run_regeneration.py` auto-runs VCD on every generated figure
- Output: `results/vcd_report_summary.md` (human-readable) and `results/vcd_report.json` (detailed)
- Common fixes: reduce annotation count, adjust font sizes, reposition legend, change `frameon=False`

### Running the full pipeline

```bash
python scripts/pipeline/run_pipeline.py --stage all              # Full pipeline
python scripts/pipeline/run_pipeline.py --from generate --stage figures  # From generation onward
python scripts/pipeline/run_pipeline.py --stage figures --build-article   # Figures + build article
```

See `PIPELINE.md` for detailed stage descriptions.

---

## Key Files and Responsibilities

| File/Dir | Role |
|----------|------|
| `src/visualization/article_delivery.py` | **Canonical 20-figure manifest** — single source of truth for figure numbering and symlinks |
| `src/visualization/style.py` | Global style settings, VCD, and figure-saving helpers |
| `src/visualization/results_visualizer.py` | Orchestrator that generates Figs 3–18 |
| `src/visualization/fig0X_*.py` | Individual figure modules (Figs 3–18) |
| `src/utils/paths.py` | Centralized path resolution (env > YAML > defaults) |
| `configs/pipeline.yaml` | Path configuration (fallback to `paths.py` defaults) |
| `scripts/analysis/generate_architecture_figure.py` | Fig 1 source |
| `scripts/analysis/evaluation_pipeline_figure.py` | Fig 2 source |
| `scripts/analysis/variance_matching_pilot.py` | Fig 19 source |
| `scripts/analysis/gene_gene_correlation.py` | Fig 20 source |
| `scripts/pipeline/run_regeneration.py` | Regenerate all figures with VCD checks |
| `scripts/pipeline/build_article.sh` | Build LaTeX article PDF |
| `articles/clop_dit_biology.tex` | Main LaTeX manuscript |

---

## Testing

All tests use pytest and are located in `tests/`:

```bash
pytest tests/                    # Run all
pytest tests/test_visualization.py -v  # Run with verbose output
```

Common test patterns:
- **Import tests:** verify all modules load without error
- **Integration tests:** test article_delivery symlink creation
- **Config tests:** verify paths resolve correctly

---

## Important Conventions

1. **Never hardcode paths** — Always import from `src.utils.paths`
2. **Global style first** — Call `apply_style()` before creating figures
3. **Save with VCD** — Use `save_with_vcd(fig, path)` instead of `plt.savefig()`
4. **Symlinks only** — Don't copy figures into `articles/figures/`; use symlinks via `article_delivery.py`
5. **Centralized colors** — Use `COLORS` dict from `style.py` for semantic consistency
6. **Panel labels** — Use `add_panel_label(ax, 'a')` helper from `style.py` for consistency

---

## Troubleshooting

**ModuleNotFoundError when running scripts:**
Make sure you've installed the package in development mode: `pip install -e .`

**Figures not updating after edits:**
1. Clear old figures: `rm results/figures/*`
2. Regenerate: `python scripts/pipeline/run_regeneration.py`
3. Rebuild article: `bash scripts/pipeline/build_article.sh`

**LaTeX compilation fails:**
1. Check that all figures exist: `bash scripts/pipeline/build_article.sh --check`
2. Verify symlinks: `ls -la articles/figures/`
3. Check LaTeX log: `articles/clop_dit_biology.log`

**Path issues with pipeline:**
- Verify `configs/pipeline.yaml` exists and is readable
- Check environment variables: `env | grep CLOPDIT`
- Test path resolution: `python -c "from src.utils.paths import FIG_DIR; print(FIG_DIR)"`

---

## References

- `README.md` — Project overview and quick start
- `PIPELINE.md` — Detailed pipeline stage documentation
- `REPRODUCIBILITY.md` — Reproduction guide and environment setup
- `VERSIONS.md` — Version history and model checkpoints
