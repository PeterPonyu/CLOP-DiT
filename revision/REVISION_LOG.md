# REVISION_LOG — single source of truth for the major revision

This file is the canonical handoff document for the `revision/major`
branch. Any future session should read this first to understand what
has already been done, what the current state is, and where the
unfinished work lives. Commit hashes and line numbers link directly
back to the workspace so nothing has to be reconstructed from memory.

Last updated: 2026-04-15 (end of Lane A).

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
| Lane B (partial retrain) | Not started. |
| Lane C (data expansion) | Not started — gated on A3. |
| Lane D (encoder comparison) | Not started — supplementary. |

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
| 5 | `ae1ee2b` | **Scrub venue-specific references and untrack manuscript sources.** Rewrote README, revision docs, and Zenodo notes to be journal-neutral; broadened gitignore patterns from hardcoded paths to globs; `git rm --cached` of `manuscripts/{v1_prerevision,v2_revision}/` and root `cover_letter_peerj_cs.tex`; deleted `scripts/pipeline/build_article_elsevier.sh`. |
| 6 | `4deabda` | Generalise two source-comment references to publication-target wording. |
| 7 | Re-tag | Deleted `pre-revision-2026-04-15` on both local and remote, re-tagged at scrubbed `4deabda` with a neutral message, force-pushed. `git archive pre-revision-2026-04-15` tarball verified to contain zero venue references. |
| 8 | GH release | `gh release create pre-revision-2026-04-15` with notes from `revision/ZENODO_RELEASE_NOTES.md` — triggers Zenodo DOI. |
| 9 | `3ac8f76` | **A4 — HVG variance deficit numeric summary.** |
| 10 | `5676065` | **A3 — Abundance vs fidelity quantitative analysis.** |
| 11 | `1375701` | **A2 — Organism-stratified evaluation.** |
| 12 | `f7364a7` | **A5 — Rare-cell augmentation failure mechanism.** |
| 13 | `44238ab` | **A1 — KNN error taxonomy + close Lane A.** |

Final `revision/major` HEAD: `44238ab`.

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

## 5. Immediate next steps (Lane B)

| Order | Experiment | Cost | Why now |
|---:|---|---|---|
| 1 | **B1 — Gaussian / unconditional `r_var` baseline** | Inference only (hours) | Answers R2.9 directly and reconciles A4's three-slice variance story against a statistical floor. |
| 2 | **B3 — Rare-cell mixing strategy sweep** | Downstream classifier training only | A5 already points the diagnosis at the upstream generator, so B3 can target latent-stage mixing strategies (random oversampling / SMOTE / hybrid). |
| 3 | **B4 — CLOP→full bridge (3 DiT retrains)** | 3 × DiT retrain | Only once A3 has identified the dominant predictor (heterogeneity) is the right set of CLOP ablations clear. |
| 4 | **B2 — ZCA formal ablation** | 4 × CLOP + 2–3 × DiT retrain | Most expensive; saved for when the reviewer-direct experiments are done. |

Lane C is gated on Lane A3 (already complete); Lane D is the
lowest-priority supplementary lane.

All Lane B experiments must:

- Read the baseline metric from
  `revision/prerevision_baseline/metrics_frozen/*`, not live
  `results/`.
- Save new checkpoints under `models/revision/<experiment_id>/` so
  `artifact_hashes.txt` stays valid for the frozen baseline.
- Emit `run_log.txt` with seed, config hash, commit, wall-clock.
