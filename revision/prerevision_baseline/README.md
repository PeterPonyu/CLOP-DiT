# Pre-revision Baseline — FROZEN REFERENCE

**DO NOT MODIFY THIS DIRECTORY.**

Every file in here is a frozen snapshot of the pre-revision state as of
**2026-04-15**, git tag `pre-revision-2026-04-15`, commit `3b6f38a`.

All revision experiments must be compared against these files, not against
their live counterparts in `results/`, `articles/`, or `models/`, because
those locations will be mutated during the revision.

## Contents

| File / dir | What it fixes |
|---|---|
| `git_state.txt` | Exact commit, tag, and date at freeze time |
| `artifact_hashes.txt` | SHA256 of every file in `data/cached_latents/` and `models/checkpoints/` — the full reproducibility root |
| `baseline_snapshot_2026-04-15.md` | Headline metrics: KNN, steering, DivR, r_var, field ablation, OOD, rare-cell, tissue-level fidelity |
| `metrics_frozen/` | JSON copies of every `results/*.json` that a headline metric in the baseline snapshot is derived from |

## Provenance

The `metrics_frozen/` files were copied (not moved) from `results/` at the
moment of the freeze. The originals remain in `results/` and are free to be
overwritten during revision experiments — the frozen copies here are what
reviewer-facing comparisons should reference.

## How to verify the baseline is still reachable

```bash
# from repo root
git cat-file -e pre-revision-2026-04-15^{commit} && echo "tag present"
sha256sum -c revision/prerevision_baseline/artifact_hashes.txt | grep -v ': OK$' | head
# the second command prints nothing if all baseline artifacts are unchanged
```

## If you need to update this directory

You almost certainly do not. The only legitimate reasons are:

1. A file was accidentally omitted during freeze — add it, update
   `git_state.txt` with a note, and commit in a separate commit.
2. A typo fix in a `.md` or `.txt` here that does not change any numeric
   value.

If you need a **new** baseline (e.g., after a major revision is itself
archived), create a sibling directory `prerevision_baseline_<new_date>/`
and tag a new commit. Do **not** overwrite this one.
