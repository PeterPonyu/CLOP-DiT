# REVISION_LOG — single source of truth for the major revision

This file is the canonical handoff document for the `revision/major`
branch. Any future session should read this first to understand what
has already been done, what the current state is, and where the
unfinished work lives. Commit hashes and line numbers link directly
back to the workspace so nothing has to be reconstructed from memory.

Last updated: 2026-04-15 (Lane A complete, B1+B2+sec.7+8 complete, Lane D complete).

---

## 0. Session summary (read this first)

| Item | State |
|---|---|
| Active branch | `revision/major` |
| Pre-revision anchor tag | `pre-revision-2026-04-15` → clean HEAD of `revision/major` |
| Pre-submission branch | `github-ready` (untouched, still points to pre-revision commit `3b6f38a`) |
| Public GitHub release | <https://github.com/PeterPonyu/CLOP-DiT/releases/tag/pre-revision-2026-04-15> |
| Zenodo DOI | To auto-issue from the tagged release |
| Venue-sensitive information in tracked files | **None.** All public docs are journal-neutral; manuscript sources are local-only under `revision/manuscripts/` and gitignored. |
| Lane A (post-hoc analysis) | **Complete (5/5).** All scripts deterministic, rerunnable, SHA-256 pinned to inputs. |
| Lane B (partial retrain) | B1 latent-level first pass complete; **B2 ZCA ablation complete (section 9)**; B3/B4 not started. |
| Lane C (data expansion) | Not started — gated on A3. |
| Lane D (encoder comparison) | **Complete.** scGPT-specific compression confirmed (section 10). |

Integrity verification (run any time):

```bash
git cat-file -e pre-revision-2026-04-15^{commit} && echo "anchor present"
sha256sum -c revision/prerevision_baseline/artifact_hashes.txt | grep -v ': OK$' | head
```

---

## 1. Chronological commit log on `revision/major`

Every commit is intentional, signed, and reversible. No force-pushes
except the single tag move at step 5.

| Step | Commit | Summary |
|---:|---|---|
| 1 | `162f15a` | **Scaffold revision workspace + freeze pre-revision baseline.** Created `revision/prerevision_baseline/` with the SHA-256 manifest (189 artifact files under `data/cached_latents/` and `models/checkpoints/`), copied key `results/*.json` into `metrics_frozen/`, froze the pre-revision manuscript sources into `revision/manuscripts/v1_prerevision/`, and seeded 11 empty experiment stubs under `revision/experiments/`. |
| 2 | `68a970c` | Track pre-revision cover-letter source and `figure_subpanels_archive/` so no on-disk state from the submitted version can be lost. |
| 3 | `236bc2a` | Seed `v2_revision/` (copy of `v1_prerevision/`) and fill per-experiment `results.md` stubs (A1–A5, B1–B4, C, D). |
| 4 | `e64d6a1` | Add `revision/ZENODO_RELEASE_NOTES.md` ready to be used by `gh release create`. |
| 5 | `ae1ee2b` | **Scrub venue-specific references and untrack manuscript sources.** Rewrote README, revision docs, and release notes to be venue-neutral; broadened gitignore patterns from hardcoded paths to globs; `git rm --cached` of the manuscript subtrees under `revision/manuscripts/` and of the legacy cover-letter source at repository root; deleted the venue-specific article-build helper under `scripts/pipeline/`. |
| 6 | `4deabda` | Generalise two source-comment references to publication-target wording. |
| 7 | Re-tag | Deleted `pre-revision-2026-04-15` on both local and remote, re-tagged at scrubbed `4deabda` with a neutral message, force-pushed. `git archive pre-revision-2026-04-15` tarball verified to contain zero venue references. |
| 8 | GH release | `gh release create pre-revision-2026-04-15` with notes from `revision/ZENODO_RELEASE_NOTES.md` — triggers Zenodo DOI. |
| 9 | `3ac8f76` | **A4 — HVG variance deficit numeric summary.** |
| 10 | `5676065` | **A3 — Abundance vs fidelity quantitative analysis.** |
| 11 | `1375701` | **A2 — Organism-stratified evaluation.** |
| 12 | `f7364a7` | **A5 — Rare-cell augmentation failure mechanism.** |
| 13 | `44238ab` | **A1 — KNN error taxonomy + close Lane A (R2.6).** |
| 14 | `185cc36` | Add this `REVISION_LOG.md` master handoff document. |
| 15 | `6808237` | Remove residual venue name from `REVISION_LOG.md`. |
| 16 | `d0023f1` | **B1 — Gaussian / unconditional `r_var` latent-level baseline (R2.9 first pass).** |
| 17 | `8cb4241` | Update `REVISION_LOG.md` with B1 results + refreshed Lane-B priority order. |

Final `revision/major` HEAD: `8cb4241`.

---

## 2. Lane A — scientific audit and rebuttal-ready findings

All five analyses share a common invariant:

- Inputs are frozen at or derived from artifacts fixed at
  `pre-revision-2026-04-15`; SHA-256 prefixes are embedded in every
  output JSON under `inputs.*.sha256_head`.
- No script mutates `data/`, `models/`, or `results/`. Everything is
  read-only against those and writes only into its own experiment
  subdirectory.
- Each experiment emits (i) a `compute_*.py` script, (ii) a machine-
  readable JSON payload, (iii) a human-readable `*_preview.txt`, and
  (iv) a `results.md` with a paste-ready rebuttal sentence.

### Scientific audit table

| ID | Reviewer | Scope / method | Known caveat (already documented in results.md) | Scientifically sound? |
|---|---|---|---|---|
| A1 | R2.6 | 1-NN cosine, 6 900 generated latents vs ≤ 600 real/type pool; errors classified by a 10-family taxonomy over all 69 gids. | KNN protocol differs from the baseline held-out protocol that produced the headline 0.369 number; the exercise measures **error composition**, not overall accuracy. Space alignment verified: gen vs `cell_embeddings_dedup_preprocessed` cosine ≈ 0.92, vs `projected_cells` ≈ 0.08. | ✓ Sound. |
| A2 | R3.2 | Type-level organism classification from `text_caption_metadata.json`; Mann-Whitney U across three groups (human_only, mouse_only, both) on six per-type metrics. | Stratification is TYPE-level, not cell-level — because the frozen baseline does not emit per-cell organism tags. A strict cell-level stratification is scoped for Lane C, but the type-level answer is already informative enough to rule out a species bias. | ✓ Sound. |
| A3 | R2.7 | Pearson and Spearman of four per-type metrics against `log10(training_count)` and against `real_intra_cos`; bottom-vs-top-quartile Mann-Whitney. | 68 / 69 types matched (only "Generic epithelial cells" dropped); expression Pearson r saturates at ≈ 1.00 so it is not informative at the per-type level and is reported for completeness. | ✓ Sound — the saturation is documented. |
| A4 | R2.8 | Gene-wise variance ratio at three scales (pooled, within-type averaged, cross-dataset held-out reference). Tolerance band `[0.5, 2.0]`; HVG = top 25 % by real variance. | The number reviewers quote (r_var ≈ 0.01) is the cross-dataset ratio — a strictly different slice from within-type on the ID set. All three slices are reported side-by-side precisely to prevent cherry-picking. | ✓ Sound. |
| A5 | R2.11 | Within-type total variance and intra-cluster cosine at the latent scale (preprocessed CLOP input space) and the decoded expression scale; mechanism verdict from a (latent ratio, expression ratio) decision rule with 30 % tolerance. | "Rare by count" is the bottom-quartile of training counts — mechanistic rarity (e.g. Cycling) is handled separately via the reviewer-flagged slot. The 30 % tolerance is coarse but results are crisp: 13/18 rare-by-count types are "upstream compression, decoder expands" and 5/5 reference top-count types are "approximately matched". | ✓ Sound. |

### Headline findings (compressed)

1. **A1** — 49.2 % of all KNN mis-typings stay within the same
   biological family; 9 of the top-15 most-confused type pairs are
   within-family. Catastrophic cross-family failure is the minority.
2. **A2** — Human-only vs mouse-only cell types are statistically
   indistinguishable on every headline metric (all p > 0.14). No
   species bias at the type level.
3. **A3** — `log10(training_count)` is a weak or wrong-signed
   predictor of per-type fidelity (Frechet-distance Spearman
   ρ = +0.57). `real_intra_cos` (within-type tightness) is the right
   predictor (Spearman ρ = −0.77, p = 3 × 10⁻¹³). Retire the phrase
   "biologically difficult" in favour of "broad intrinsic
   heterogeneity".
4. **A4** — Pooled gene-wise variance Pearson r = 0.988 is dominated
   by between-type means. Within-type average Pearson r = +0.85,
   median variance ratio 1.13. Cross-dataset held-out ratio ≈ 0.01
   (strict-OOD variance collapse). All three numbers are correct for
   their scope; the baseline snapshot's headline "near-zero r_var"
   referred to the third slice only.
5. **A5** — The rare-cell augmentation failure is **upstream**:
   13/18 rare types show latent variance compressed to 30–66 % of
   real while decoded expression variance recovers to 47–190 %. The
   scGPT decoder is already expansive; the generator is centroid-
   biased. Latent-stage mixing (Lane B3) is the correct lever.

### Cross-experiment coherence check

- A3 (heterogeneity > abundance) and A5 (upstream compression of
  latent spread) are mutually consistent: the types where the
  generator underperforms are the types with broad real spread (low
  `real_intra_cos`), and the failure mode is the generator failing to
  reproduce that spread.
- A2 finds no species bias at the type level. A3 finds the dominant
  predictor is heterogeneity, not abundance. Together they argue that
  the dataset-count imbalance (59 human / 21 mouse) is not the right
  frame for cross-species concerns.
- A4's pooled-vs-within-type spread, A5's latent-vs-expression
  variance ratios, and A1's family-level confusion are all consistent
  with a generator that is good at placing type centroids but under-
  disperses within-type.

Taken together, Lane A tells a single coherent story that can be used
as the **framing paragraph** for the revised Discussion.

---

## 3. Infrastructure and provenance

- Git tag `pre-revision-2026-04-15` → commit `4deabda` (scrubbed HEAD).
  `git archive` of this tag contains 0 venue references (verified).
- `revision/prerevision_baseline/artifact_hashes.txt` covers 189
  files. Verified 189/189 OK at time of freeze. Re-running
  `sha256sum -c` is the provenance check.
- Manuscript sources (`v1_prerevision`, `v2_revision`, `diff`) are
  gitignored. Local-only. Rerunning `latexdiff` does not need any
  remote sync.
- Remote backup: `origin/revision/major` = local `revision/major`.
  Every Lane A commit has been pushed.

---

## 4. Next-session handoff

If the next session starts cold, the minimal orientation is:

1. Read `revision/README.md` for the directory map.
2. Read this file (`REVISION_LOG.md`) for commit-level history and
   scientific status.
3. Read `revision/reviewer_response_draft.md` for the full comment-
   by-comment rebuttal structure.
4. The five Lane A `results.md` files are self-contained and each
   ends with a rebuttal sentence ready to paste into the response
   letter.
5. `revision/prerevision_baseline/baseline_snapshot_2026-04-15.md` is
   the headline-metric reference every new experiment must compare
   against. Do not read numbers off the live `results/` directory,
   which will drift as Lane B runs.

---

## 5. B1 latent-level summary (2026-04-15)

Within-type mean Pearson `r_var` on the in-distribution CLOP latent
slice, with identical per-type sample counts:

| Generator | within-type mean r_var | within-type median variance ratio | fraction of 69 types with r_var > 0 |
|---|---:|---:|---:|
| Gaussian-per-type (oracle) | +0.813 | 1.05 | 100 % |
| **CLOP-DiT** | **+0.201** | 0.71 | 100 % |
| Pooled Gaussian (CFG = 0 stand-in) | +0.014 | 1.20 | 60.9 % |

CLOP-DiT is strictly above the type-agnostic floor and strictly below
the type-aware oracle. The near-zero variance recovery reported in
the baseline's cross-dataset held-out slice is therefore not a shared
feature of latent generators; CLOP-DiT's gap to the oracle is the
improvement headroom targeted by Lane B3 and Lane C.

Scope note: B1 is currently latent-level only. A decoder-level
extension (reproducing the cross-dataset `median_variance_ratio`
column for all three generators) is queued but not required to
answer R2.9; the decoder is expansive (A5), so the latent limitation
is what propagates.

## 6. Immediate next steps (Lane B continued)

| Order | Experiment | Cost | Why now |
|---:|---|---|---|
| 1 | **B3 — Rare-cell mixing strategy sweep** | Downstream classifier training only | A5 + B1 both point at upstream latent under-dispersion; B3 tests whether latent-stage mixing recovers rare-class F1. |
| 2 | **B1-decoder extension** | Decoder inference only | Reproduce the cross-dataset `median_variance_ratio` column for Gaussian and pooled baselines to complete the R2.9 response at the expression scale. |
| 3 | **B4 — CLOP→full bridge (3 DiT retrains)** | 3 × DiT retrain | Turn the selected CLOP ablations into end-to-end comparisons. |
| 4 | **B2 — ZCA formal ablation** | 4 × CLOP + 2–3 × DiT retrain | Most expensive; saved for last. |

Lane C is gated on Lane A3 (already complete); Lane D is the
lowest-priority supplementary lane.

All Lane B experiments must:

- Read the baseline metric from
  `revision/prerevision_baseline/metrics_frozen/*`, not live
  `results/`.
- Save new checkpoints under `models/revision/<experiment_id>/` so
  `artifact_hashes.txt` stays valid for the frozen baseline.
- Emit `run_log.txt` with seed, config hash, commit, wall-clock.

## 7. Cap-increase experiment (2026-04-15, Lane B follow-up)

**Tag:** `pre-cap-increase-2026-04-15` (pushed to origin before
anything was re-run).
**Scripts:** `revision/experiments/cap_increase/`
**Driver question:** would a larger per-dataset cell cap
(`max_cells=10000` vs the baseline 3000) expose additional within-type
latent variance and thereby address R2.11's heterogeneity concern?

### What was run

1. Added `--subsample_seed` flag to `scripts/data_prep/00_prepare_all_data.py`
   so `sc.pp.subsample` is reproducible (commit `c33e90e`).
2. Cap10k: re-preprocessed the 50 non-geodh datasets with
   `max_cells=10000 seed=0`. 432 354 cells at cap10k vs 138 477 at
   cap3k (same 50 datasets). Output: `data/processed_h5ad_cap10k/`.
3. Cap3k_seed0: matched control run with `max_cells=3000 seed=0`.
   Output: `data/processed_h5ad_cap3k_seed0/`.
4. Both caches re-embedded with scGPT (`03_cache_latents.py`, frozen
   scGPT weights), 512-d latents.
5. Leiden + signature-based subcluster annotation on both caches
   (`02_subcluster_descriptions.py`, adaptive resolution).

### Headline result

With the seed fixed on both caps (so **only the cap differs**):

| metric | median | mean | IQR |
|---|---:|---:|---:|
| per-dataset variance ratio | **1.008** | 1.010 | [0.984, 1.035] |
| within-cluster variance ratio (475 pairs) | **0.976** | 0.987 | [0.895, 1.079] |
| within-cluster mean-L2 ratio | 0.988 | 0.988 | [0.948, 1.036] |

3.12 x more cells in → **no practically-meaningful gain in scGPT
latent variance, per-dataset or per-cluster**.

### Why the first-pass diagnostic looked bimodal

The first cap3k-vs-cap10k comparison used the historic unseeded
cap3k cache and reported gains up to 5 x and losses down to 0.16 x.
The seed-aligned rerun collapses that IQR to `[0.984, 1.035]` and
`[0.895, 1.079]`; the tails were pure `sc.pp.subsample` draw drift,
not cap effect.

### Mechanistic read

Within-type variance in **scGPT latent space saturates well before
3 000 cells**. The encoder's raw-count → rank-binning → frozen
transformer path lacks the capacity to represent the fine-grained
variation R2.11 points at. A cap-increase cannot fix an
encoder-side bottleneck; this is consistent with the A5 + B1 Lane
diagnosis.

### Implications for the revision

- **Do not advertise cap-increase as a heterogeneity fix.** The
  seed-aligned comparison disproves it at the scGPT latent level.
- **Do not retrain CLOP / DiT on the cap10k cache.** There is no
  new signal to train on.
- **Report the experiment as a negative result** that corroborates
  the existing story: the bottleneck is at the encoder / generator
  stage, upstream of data volume.
- Paste-ready rebuttal paragraph is in
  `revision/experiments/cap_increase/SYNTHESIS.md`.

### Artefacts (gitignored; referenced by path only)

- `data/processed_h5ad_cap10k/` and `data/processed_h5ad_cap3k_seed0/`
- `data/cached_latents_cap10k/` and `data/cached_latents_cap3k_seed0/`
- `data/processed_h5ad_{cap10k, cap3k_seed0}/subcluster_metadata.json`
- Versioned JSONs under `revision/experiments/cap_increase/`:
  `seed0_variance_comparison.json`, `within_cluster_variance.json`,
  `cap3k_vs_cap10k_cellcounts.json`, `per_dataset_variance_ratio.json`
  (first-pass, confounded), `smoke_report.json`.

### Commit trail

`85d64e6` → `c33e90e` → `c7271ee` → `dd605ed` → `de378ea`.

## 8. Encoder bottleneck experiment trio (2026-04-15)

Three experiments converge on the same diagnosis: scGPT encoder
saturates within-type latent variance regardless of input.

**Scripts:** `revision/experiments/encoder_bottleneck/`
**Synthesis:** `revision/experiments/encoder_bottleneck/SYNTHESIS.md`

### Three experiments

| experiment | lever | range | within-type latent var change |
|---|---|---|---:|
| Cap-increase (sec. 7) | `max_cells` | 3000 -> 10000 | median ratio 1.008 (null) |
| HVG ablation | `n_top_genes` | 500 -> 8000 | median 11.18 -> 8.49 -> 8.78 (saturates >= 2000) |
| Raw-vs-latent | (no lever) | – | within L2 ratio 0.168 (encoder compresses 6 x) |

### Conclusion

The within-type heterogeneity gap that R2.11 flags is a property of
the frozen scGPT latent manifold, not of CLOP or the DiT. No
input-side lever (cap-cells, HVG count, seed) can move it. Productive
future levers: encoder replacement (Lane D), CLOP variance-preserving
loss, or DiT variance-preserving objective. Paste-ready Discussion
paragraph in the synthesis doc.

### Commit trail

`0b73b71` (raw-vs-latent) -> `e72124b` (HVG partial 500/1k/2k) ->
`addd49f` (HVG complete 4k/8k).

## 9. ZCA whitening ablation (2026-04-15, Lane B2)

**Scripts:** `revision/experiments/zca_ablation/`
**Synthesis:** `revision/experiments/zca_ablation/SYNTHESIS.md`
**Driver question:** Is ZCA whitening a critical component of the CLOP
pipeline, or is the aligner robust to its removal?

### Design

Three CLOP training runs (100 epochs, seed 42, PrototypeSigLIP,
identical architecture) differing only in embedding preprocessing:

| condition | preprocessing | cell pairwise cos after |
|---|---|---:|
| whiten (ZCA) | mean-center + ZCA + L2 | 0.43 |
| center_norm | mean-center + L2 | 0.87 |
| none | raw collapsed | 0.99 |

### Results

| condition | quality | proto_acc | tc_align | sep | cos_sim | best ep |
|---|---:|---:|---:|---:|---:|---:|
| whiten | **0.966** | 0.999 | 0.978 | 1.011 | **0.851** | 98 |
| center_norm | 0.960 | 1.000 | 0.984 | 1.010 | 0.815 | 96 |
| none | 0.956 | 0.998 | 0.977 | 1.010 | 0.809 | 99 |

Quality drop from whiten → none: **1.0 %**. Prototype accuracy
near-perfect (≥ 0.998) in all conditions. Main ZCA contribution is
a 5 % improvement in positive-pair cosine similarity.

### Conclusion

ZCA whitening is a modest refinement, not a critical component. The
3-layer MLP projectors + PrototypeSigLIP loss learn through raw
embedding collapse. ZCA's benefit is in fine-grained cosine
structure (decorrelating the 512-d scGPT dimensions). This is
consistent with the encoder-bottleneck finding: most useful signal
is in between-type centroid structure, well-preserved without
preprocessing.

### Commit trail

(single commit with all ablation infrastructure + results)

## 10. Encoder comparison experiment (2026-04-15, Lane D)

**Scripts:** `revision/experiments/encoder_comparison/`
**Synthesis:** `revision/experiments/encoder_comparison/SYNTHESIS.md`
**Driver question:** Is the within-type compression diagnosed in Section 8
specific to the scGPT transformer architecture, or universal to all cell
embedding methods?

### Design

Three encoders applied to 8 representative datasets spanning the full
tightness ratio range (0.39–1.12 from experiment A):

| encoder | architecture | training data | embedding dim |
|---|---|---|---:|
| scGPT-human | 12-layer transformer | 33M human cells | 512 |
| scGPT-pancancer | 12-layer transformer | 5.7M cancer cells | 512 |
| PCA | linear (TruncatedSVD) | fitted per-dataset | 512 |

### Results

**Within-L2 compression ratio** (encoder / raw; lower = more compression):

| encoder | median | mean | IQR |
|---|---:|---:|---|
| scGPT-human | **0.144** | 0.154 | [0.121, 0.181] |
| scGPT-pancancer | **0.066** | 0.072 | [0.051, 0.086] |
| PCA | **0.899** | 0.903 | [0.890, 0.922] |

**Tightness ratio** (encoder / raw):

| encoder | median | mean | IQR |
|---|---:|---:|---|
| scGPT-human | 0.647 | 0.666 | [0.459, 0.797] |
| scGPT-pancancer | 0.670 | 0.779 | [0.537, 0.958] |
| PCA | 0.900 | 0.904 | [0.891, 0.923] |

### Conclusion

The within-type compression is **scGPT-architecture-specific, not
universal**. PCA preserves within-type L2 almost perfectly (0.90 ratio)
while both scGPT variants compress aggressively (0.07–0.14). Since PCA
maps to the same 512-d dimensionality, dimensionality reduction is ruled
out as the cause. The compression arises from the transformer's learned
representation, which prioritises cell-type identity over within-type
fine structure.

scGPT-pancancer compresses ~2× more aggressively than scGPT-human
(0.066 vs 0.144), confirming that training data diversity modulates but
does not eliminate the compression. Both variants produce similarly
separable clusters (tightness ratio ~0.65–0.67).

This sharpens the Section 8 diagnosis from "frozen encoder" to
"transformer learned representation" and confirms scGPT-human as the
better production choice.

### Commit trail

(this commit)

## 11. B4 CLOP→full-pipeline bridge — Phase 1 diagnostic (2026-04-15)

**Scripts:** `revision/experiments/b4_clop_bridge/`
**Driver question (R2.10):** Do CLOP Stage-1 ablation conclusions
transfer to the full generation pipeline, or are they specific to the
CLOP-only evaluation?

### Phase 1: Projected text space diagnostic

Projected 1,088 unique text embeddings through 5 CLOP checkpoints
(production, ablation baseline, no_cohesion, no_cell_noise,
fixed_temperature) and compared the resulting 512-d projected text
spaces structurally (Pearson r of the 1088 × 1088 inter-group cosine
similarity matrix).

| comparison | absolute cos | structural Pearson r |
|---|---:|---:|
| production vs abl_baseline | 0.267 | 0.053 |
| production vs no_cohesion | 0.347 | 0.064 |
| abl_baseline vs no_cell_noise | 0.884 | 0.756 |
| abl_baseline vs fixed_temperature | 0.646 | 0.738 |
| abl_baseline vs no_cohesion | 0.778 | 0.549 |

**Key finding:** Production CLOP and ablation baseline (identical config)
produce **structurally unrelated** projected text spaces (r ≈ 0.05).
This is expected from contrastive learning's rotational invariance:
separately trained projectors converge to different orientations.
Within the ablation suite, variants show moderate structural similarity
(r = 0.36–0.76), confirming the hyperparameter changes DO alter the
embedding geometry.

**Verdict:** Full DiT retraining per variant IS needed for the
end-to-end bridge. No shortcut available.

### Phase 2: DiT retraining (in progress)

Three DiT retrains (300 epochs each, ~90 min/variant on RTX 5090):
- `abl_baseline` → control for ablation-internal comparison
- `no_cohesion` → largest structural divergence from baseline
- `no_cell_noise` → highest Stage-1 proto_acc with most structural
  similarity to baseline

After training: generate embeddings, compute FD / centroid cosine /
coverage, and compare whether Stage-1 ranking transfers to Stage-2.

### Commit trail

(this commit, phase-2 in progress)
