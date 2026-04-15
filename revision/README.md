# Revision Workspace — Major Revision

**Branch:** `revision/major`
**Started:** 2026-04-15
**Target submission:** 2026-05-04
**Pre-revision anchor:** tag `pre-revision-2026-04-15`

This directory is the single workspace for all major-revision activity. It is deliberately separate from `articles/`, `results/`, `data/`, and `models/` so that the revision work never overwrites the submitted-version artifacts.

---

## Directory map

```
revision/
├── README.md                         ← this file
├── reviewer_response_draft.md        ← evolving rebuttal letter (English)
│
├── prerevision_baseline/             🔒 FROZEN — DO NOT MODIFY
│   ├── README.md                     what's in here + invariants
│   ├── git_state.txt                 commit / tag / date
│   ├── artifact_hashes.txt           SHA256 of cached_latents + checkpoints
│   ├── baseline_snapshot_2026-04-15.md   headline pre-revision metrics
│   └── metrics_frozen/               JSON snapshots of key results
│
├── manuscripts/                      📝 LaTeX diff workflow (local-only; gitignored)
│   ├── README.md
│   ├── v1_prerevision/               frozen .tex/.bib/figures for diff base
│   ├── v2_revision/                  active revision working tree
│   └── diff/                         latexdiff outputs (v1 vs v2)
│
├── performance_tracking/
│   └── comparison_template.md        per-experiment comparison template
│
└── experiments/                      🔬 Revision experiment lanes
    ├── lane_a_analysis/              A1–A5 — pure post-hoc analysis, no retrain
    │   ├── a1_knn_confusion/
    │   ├── a2_organism_split/
    │   ├── a3_abundance_fidelity/
    │   ├── a4_hvg_variance/
    │   └── a5_rare_failure_mechanism/
    ├── lane_b_retrain/               B1–B4 — partial retrain / baseline comp.
    │   ├── b1_gaussian_rvar_baseline/
    │   ├── b2_zca_ablation/
    │   ├── b3_mixing_sweep/
    │   └── b4_clop_to_full_bridge/
    ├── lane_c_data/                  targeted data expansion (OOD, species, rare)
    └── lane_d_encoder/               alternative text-encoder comparison
```

## Lane-to-reviewer mapping

| Lane | Experiments | Primary reviewer comments | Retraining? |
|---|---|---|---|
| **A** | A1–A5 | 2.6, 2.7, 2.8, 2.11, 3.2 | No — derived from existing artifacts |
| **B** | B1–B4 | 2.9, 2.10, 3.3, 3.4 | Partial — stage-specific |
| **C** | data expansion | 3.1, 3.2, 2.7 | Yes — full pipeline |
| **D** | encoder comparison | 2.2, 2.3, 2.5 | Yes — CLOP + DiT |

See `reviewer_response_draft.md` for the full comment-by-comment rebuttal.

## Working rules

1. **Never modify `prerevision_baseline/`.** It is the only fixed reference for all comparisons.
2. **Never commit changes to `data/`, `models/`, or `results/`.** Those directories stay `.gitignore`'d; artifact provenance is tracked via `artifact_hashes.txt`.
3. **Manuscript sources are local-only.** `manuscripts/v1_prerevision/`, `manuscripts/v2_revision/`, and `manuscripts/diff/` are gitignored. The pre-revision tex, figures, and compiled PDF live on disk for LaTeX diff purposes but are never tracked, so the public repository stays free of venue-specific templating.
4. **Each experiment lives in its own subdir under `experiments/lane_X/`.** Use the template in `performance_tracking/comparison_template.md` for the results note.
5. **Branch discipline:** all revision work commits to `revision/major`.

## Verification

```bash
# confirm you are on the revision branch
git branch --show-current              # → revision/major

# confirm the baseline anchor still exists
git tag -l | grep pre-revision         # → pre-revision-2026-04-15

# confirm artifacts have not drifted from the baseline
cd /home/zeyufu/Desktop/CLOP-DiT
sha256sum -c revision/prerevision_baseline/artifact_hashes.txt | grep -v OK
# (should print nothing if baseline artifacts are intact)
```
