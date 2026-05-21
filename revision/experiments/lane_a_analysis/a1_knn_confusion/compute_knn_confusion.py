"""A1 — KNN error taxonomy.

For every generated latent, find its 1-nearest-neighbour (cosine)
among the real latents, call that neighbour's group the predicted
label, and classify the error against a 10-family biological
taxonomy. The point of the exercise is to distinguish graceful
errors (within the same cell lineage) from catastrophic errors
(across unrelated lineages).

Inputs (all read-only, live paths):
  results/generated_embeddings.npy              (6900, 512)
  results/generated_labels.npy                  (6900,)
  data/cached_latents/cell_embeddings_dedup_preprocessed.npy  (167245, 512)
  data/cached_latents/text_group_ids_dedup.npy  (167245,)
  data/cached_latents/text_captions_deduplicated.json
  revision/experiments/lane_a_analysis/a1_knn_confusion/family_taxonomy.yaml

Outputs (written next to this script):
  knn_confusion.json         (full stats + per-family + top pairs)
  knn_confusion_preview.txt  (human-readable summary)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

GEN_LATENT = REPO_ROOT / "results" / "generated_embeddings.npy"
GEN_LABELS = REPO_ROOT / "results" / "generated_labels.npy"
REAL_LATENT = REPO_ROOT / "data" / "cached_latents" / "cell_embeddings_dedup_preprocessed.npy"
REAL_LABELS = REPO_ROOT / "data" / "cached_latents" / "text_group_ids_dedup.npy"
CAPTIONS = REPO_ROOT / "data" / "cached_latents" / "text_captions_deduplicated.json"
TAXONOMY = HERE / "family_taxonomy.yaml"


def parse_taxonomy(text: str) -> dict[int, str]:
    """Minimal YAML parser tuned for our family_taxonomy.yaml layout."""
    mapping: dict[int, str] = {}
    current_family: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_family = line[:-1].strip()
            continue
        if current_family and line.lstrip().startswith("-"):
            m = re.match(r"\s*-\s*(\d+)", line)
            if m:
                mapping[int(m.group(1))] = current_family
    return mapping


def short_name(full: str) -> str:
    for m in [" are ", " is ", " have ", " represent ", " form ", " characterize"]:
        if m in full:
            return full.split(m)[0].strip()
    return full.split(".")[0].strip()


def sha256_head(path: Path, limit: int = 2**26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def knn1_cosine(query: np.ndarray, db: np.ndarray,
                db_labels: np.ndarray, batch: int = 256) -> np.ndarray:
    """For each query row, return the label of the nearest db row by cosine."""
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    d = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-12)
    pred = np.empty(q.shape[0], dtype=db_labels.dtype)
    for i in range(0, q.shape[0], batch):
        sims = q[i:i+batch] @ d.T                    # (B, N_db)
        pred[i:i+batch] = db_labels[np.argmax(sims, axis=1)]
    return pred


def main() -> None:
    captions = json.loads(CAPTIONS.read_text())
    id_to_name = {int(k): short_name(v) for k, v in captions.items()}
    gid_to_family = parse_taxonomy(TAXONOMY.read_text())

    assert len(gid_to_family) == 69, (
        f"taxonomy covers {len(gid_to_family)} types, expected 69")

    real = np.load(REAL_LATENT)                 # (167245, 512)
    real_lbl = np.load(REAL_LABELS)
    gen = np.load(GEN_LATENT)                   # (6900, 512)
    gen_lbl = np.load(GEN_LABELS)

    # Optional speedup: subsample real to at most 600 cells per type to
    # cap the neighbour-search cost. KNN-1 is dominated by the nearest
    # neighbour and subsampling preserves type-representative neighbours.
    PER_TYPE_CAP = 600
    rng = np.random.default_rng(0)
    keep_idx = []
    for t in np.unique(real_lbl):
        idx = np.where(real_lbl == t)[0]
        if idx.size > PER_TYPE_CAP:
            idx = rng.choice(idx, PER_TYPE_CAP, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    real_sub = real[keep_idx]
    real_sub_lbl = real_lbl[keep_idx]

    pred = knn1_cosine(gen, real_sub, real_sub_lbl)

    gen_family = np.array([gid_to_family[int(t)] for t in gen_lbl])
    pred_family = np.array([gid_to_family[int(t)] for t in pred])

    # --- top-level metrics -----------------------------------------
    type_correct = (pred == gen_lbl)
    family_correct = (pred_family == gen_family)

    overall = {
        "n_samples": int(gen.shape[0]),
        "type_accuracy": float(type_correct.mean()),
        "family_accuracy": float(family_correct.mean()),
        "type_error_rate": float(1.0 - type_correct.mean()),
        "family_error_rate": float(1.0 - family_correct.mean()),
        "graceful_error_rate": float(
            ((~type_correct) & family_correct).mean()
        ),
        "catastrophic_error_rate": float(
            ((~type_correct) & ~family_correct).mean()
        ),
    }

    # Of the mis-typed cells, what fraction stayed within the family?
    mistyped = ~type_correct
    within_given_wrong = (
        float(family_correct[mistyped].mean())
        if mistyped.sum() else float("nan")
    )

    # --- per-family confusion --------------------------------------
    families = sorted(set(gid_to_family.values()))
    fam_to_idx = {f: i for i, f in enumerate(families)}
    K = len(families)
    confusion_family = np.zeros((K, K), dtype=np.int64)
    for gf, pf in zip(gen_family, pred_family):
        confusion_family[fam_to_idx[gf], fam_to_idx[pf]] += 1

    family_rows = []
    for i, f in enumerate(families):
        row = confusion_family[i]
        total = int(row.sum())
        diag = int(row[i])
        family_rows.append({
            "family": f,
            "n_queries": total,
            "within_family_correct": diag,
            "within_family_accuracy": float(diag / total) if total else None,
            "top_offdiagonal": [
                {"to_family": families[j], "count": int(row[j])}
                for j in np.argsort(row)[::-1] if j != i and row[j] > 0
            ][:3],
        })

    # --- top confused type pairs -----------------------------------
    pair_counts: Counter = Counter()
    for t_true, t_pred in zip(gen_lbl.tolist(), pred.tolist()):
        if t_true != t_pred:
            pair_counts[(int(t_true), int(t_pred))] += 1
    top_pairs = []
    for (t_true, t_pred), c in pair_counts.most_common(15):
        top_pairs.append({
            "true_gid": t_true,
            "true_name": id_to_name.get(t_true, str(t_true)),
            "true_family": gid_to_family[t_true],
            "pred_gid": t_pred,
            "pred_name": id_to_name.get(t_pred, str(t_pred)),
            "pred_family": gid_to_family[t_pred],
            "count": c,
            "within_family": gid_to_family[t_true] == gid_to_family[t_pred],
        })

    within_family_top_pairs = sum(1 for p in top_pairs if p["within_family"])

    summary = {
        "inputs": {
            "generated_embeddings": {
                "path": str(GEN_LATENT.relative_to(REPO_ROOT)),
                "shape": list(gen.shape),
                "sha256_head": sha256_head(GEN_LATENT),
            },
            "real_embeddings": {
                "path": str(REAL_LATENT.relative_to(REPO_ROOT)),
                "shape": list(real.shape),
                "subsampled_per_type_cap": PER_TYPE_CAP,
            },
            "taxonomy": str(TAXONOMY.relative_to(REPO_ROOT)),
        },
        "family_list": families,
        "overall": overall,
        "within_family_given_wrong_type": within_given_wrong,
        "per_family": family_rows,
        "confusion_matrix_family": {
            "row_labels": families,
            "col_labels": families,
            "matrix": confusion_family.tolist(),
        },
        "top_confused_type_pairs": top_pairs,
        "top_pairs_within_family_count":
            f"{within_family_top_pairs}/{len(top_pairs)}",
    }

    (HERE / "knn_confusion.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # --- preview ----------------------------------------------------
    lines = [
        "A1 KNN-1 error taxonomy",
        "=======================",
        f"n samples         : {overall['n_samples']}",
        f"type accuracy     : {overall['type_accuracy']:.3f}",
        f"family accuracy   : {overall['family_accuracy']:.3f}",
        f"type error rate   : {overall['type_error_rate']:.3f}",
        f"graceful errors   : {overall['graceful_error_rate']:.3f}  "
        f"(wrong type, right family)",
        f"catastrophic err. : {overall['catastrophic_error_rate']:.3f}  "
        f"(wrong type and wrong family)",
        f"among mis-typed, within-family = {within_given_wrong:.3f}",
        "",
        "Per-family within-family accuracy",
        "---------------------------------",
    ]
    for row in sorted(family_rows, key=lambda r: -(r["within_family_accuracy"] or 0)):
        lines.append(
            f"  {row['family']:35s} n={row['n_queries']:>4d}  "
            f"within-family acc = {row['within_family_accuracy']:.3f}  "
            + ("top off-diag: "
               + ", ".join(f"{e['to_family']}({e['count']})"
                           for e in row['top_offdiagonal'])
               if row['top_offdiagonal'] else "")
        )

    lines += ["",
              f"Top confused type pairs (within-family "
              f"= {within_family_top_pairs}/{len(top_pairs)})",
              "-" * 60]
    for p in top_pairs:
        tag = "within" if p["within_family"] else "CROSS "
        lines.append(
            f"  [{tag}] {p['count']:4d}  "
            f"{p['true_name'][:34]:34s} -> {p['pred_name'][:34]:34s}"
        )
    preview = "\n".join(lines) + "\n"
    (HERE / "knn_confusion_preview.txt").write_text(preview)
    print(preview)


if __name__ == "__main__":
    main()
