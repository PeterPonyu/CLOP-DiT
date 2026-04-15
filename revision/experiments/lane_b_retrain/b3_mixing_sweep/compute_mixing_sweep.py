"""B3 — Rare-cell mixing strategy sweep.

Tests whether CLOP-DiT-generated synthetic samples combined with
classical imbalance strategies (random oversampling, SMOTE) can
recover rare-class classification performance that naive CLOP-DiT-
only augmentation loses. Operates at the latent (CLOP) scale because
A5 + B1 both pinpoint the bottleneck as upstream latent under-
dispersion; latent-scale mixing is therefore the correct lever.

Strategies swept:

  - baseline       : no augmentation
  - oversampling   : duplicate real rare-class cells to the ratio
  - smote          : KNN-interpolation synthesis within real rare cells
  - clop           : append CLOP-DiT generated rare-class samples
  - clop+oversamp  : clop + duplicate real rare cells
  - clop+smote     : clop + SMOTE-synthesised rare cells

Ratios: synthetic_count / real_rare_train_count = 1x, 2x, 5x, 10x.

Rare classes tested (both natural scarcity, no artificial rarefaction):

  - Megakaryocytes (group 51, 967 training cells)
  - Ameloblasts    (group 64, 384 training cells)

Majority cap: 300 cells per non-rare type (keeps runs fast but gives
the classifier a real 69-way discrimination task).

Inputs (read-only):
  data/cached_latents/cell_embeddings_dedup_preprocessed.npy
  data/cached_latents/text_group_ids_dedup.npy
  results/generated_embeddings.npy
  results/generated_labels.npy
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

REAL_LATENT = REPO_ROOT / "data" / "cached_latents" / "cell_embeddings_dedup_preprocessed.npy"
REAL_LABELS = REPO_ROOT / "data" / "cached_latents" / "text_group_ids_dedup.npy"
GEN_LATENT = REPO_ROOT / "results" / "generated_embeddings.npy"
GEN_LABELS = REPO_ROOT / "results" / "generated_labels.npy"

RARE_GIDS = [51, 64]
MAJORITY_CAP = 300
SEED = 0
TEST_FRAC = 0.30
RATIOS = [1, 2, 5, 10]


def sha256_head(path: Path, limit: int = 2**26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def smote(minority: np.ndarray, n_synth: int,
          k: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n_synth synthetic rare-class points by KNN-interpolation."""
    n, d = minority.shape
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(minority)
    _, idx = nn.kneighbors(minority)
    idx = idx[:, 1:]  # drop self
    anchors = rng.integers(0, n, n_synth)
    neighbors = idx[anchors, rng.integers(0, k_eff, n_synth)]
    alpha = rng.uniform(0, 1, (n_synth, 1)).astype(minority.dtype)
    return minority[anchors] + alpha * (minority[neighbors] - minority[anchors])


def assemble_majority(real: np.ndarray, real_l: np.ndarray,
                      rare_gid: int, test_frac: float,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    """Build majority train/test slices capped at MAJORITY_CAP per type."""
    X_train, y_train, X_test, y_test = [], [], [], []
    for gid in np.unique(real_l):
        if gid == rare_gid:
            continue
        idx_all = np.where(real_l == gid)[0]
        n_keep = min(MAJORITY_CAP, idx_all.size)
        idx = rng.choice(idx_all, n_keep, replace=False)
        # stratified split
        tr, te = train_test_split(idx, test_size=test_frac,
                                  random_state=int(rng.integers(0, 2**31)))
        X_train.append(real[tr]); y_train.extend([gid] * tr.size)
        X_test.append(real[te]); y_test.extend([gid] * te.size)
    return (np.concatenate(X_train, axis=0),
            np.asarray(y_train, dtype=real_l.dtype),
            np.concatenate(X_test, axis=0),
            np.asarray(y_test, dtype=real_l.dtype))


def evaluate_strategy(X_tr: np.ndarray, y_tr: np.ndarray,
                      X_te: np.ndarray, y_te: np.ndarray,
                      rare_gid: int) -> dict:
    clf = LogisticRegression(solver="lbfgs", max_iter=120,
                             n_jobs=-1, multi_class="auto")
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "rare_f1": float(f1_score(y_te, pred,
                                  labels=[rare_gid], average=None)[0]),
        "macro_f1": float(f1_score(y_te, pred, average="macro",
                                    zero_division=0)),
        "accuracy": float((pred == y_te).mean()),
        "train_n": int(X_tr.shape[0]),
        "fit_seconds": round(time.time() - t0, 2),
    }


def run_one_rare(real: np.ndarray, real_l: np.ndarray,
                 gen: np.ndarray, gen_l: np.ndarray,
                 rare_gid: int) -> dict:
    rng = np.random.default_rng(SEED + rare_gid)
    X_maj_tr, y_maj_tr, X_te_maj, y_te_maj = assemble_majority(
        real, real_l, rare_gid, TEST_FRAC, rng
    )

    rare_real = real[real_l == rare_gid]
    rare_gen = gen[gen_l == rare_gid]

    rare_tr_idx, rare_te_idx = train_test_split(
        np.arange(rare_real.shape[0]), test_size=TEST_FRAC,
        random_state=SEED
    )
    X_rare_tr_real = rare_real[rare_tr_idx]
    X_rare_te = rare_real[rare_te_idx]

    X_te = np.concatenate([X_te_maj, X_rare_te])
    y_te = np.concatenate([y_te_maj, np.full(X_rare_te.shape[0], rare_gid,
                                              dtype=y_maj_tr.dtype)])

    n_rare_tr = X_rare_tr_real.shape[0]

    # --- baseline (no aug) -----------------------------------------
    X_tr = np.concatenate([X_maj_tr, X_rare_tr_real])
    y_tr = np.concatenate([y_maj_tr, np.full(n_rare_tr, rare_gid,
                                              dtype=y_maj_tr.dtype)])
    base = evaluate_strategy(X_tr, y_tr, X_te, y_te, rare_gid)

    out = {
        "rare_gid": int(rare_gid),
        "n_real_rare_total": int(rare_real.shape[0]),
        "n_real_rare_train": int(n_rare_tr),
        "n_real_rare_test": int(X_rare_te.shape[0]),
        "n_clop_rare_available": int(rare_gen.shape[0]),
        "majority_train_size": int(X_maj_tr.shape[0]),
        "test_total_size": int(X_te.shape[0]),
        "baseline": base,
        "sweep": {},
    }

    strategies = ("oversampling", "smote", "clop",
                  "clop+oversamp", "clop+smote")

    for strategy in strategies:
        out["sweep"][strategy] = {}
        for ratio in RATIOS:
            n_synth = n_rare_tr * ratio
            rng_local = np.random.default_rng(
                SEED + rare_gid * 100 + ratio
            )

            if strategy == "oversampling":
                synth_idx = rng_local.choice(
                    n_rare_tr, n_synth, replace=True
                )
                synth = X_rare_tr_real[synth_idx]

            elif strategy == "smote":
                synth = smote(X_rare_tr_real, n_synth, k=5, rng=rng_local)

            elif strategy == "clop":
                n_avail = rare_gen.shape[0]
                if n_avail == 0:
                    continue
                synth_idx = rng_local.choice(
                    n_avail, n_synth, replace=(n_synth > n_avail)
                )
                synth = rare_gen[synth_idx]

            elif strategy == "clop+oversamp":
                half = n_synth // 2
                n_avail = rare_gen.shape[0]
                if n_avail == 0:
                    continue
                a_idx = rng_local.choice(
                    n_avail, half, replace=(half > n_avail)
                )
                b_idx = rng_local.choice(
                    n_rare_tr, n_synth - half, replace=True
                )
                synth = np.concatenate(
                    [rare_gen[a_idx], X_rare_tr_real[b_idx]], axis=0
                )

            elif strategy == "clop+smote":
                half = n_synth // 2
                n_avail = rare_gen.shape[0]
                if n_avail == 0:
                    continue
                a_idx = rng_local.choice(
                    n_avail, half, replace=(half > n_avail)
                )
                synth = np.concatenate([
                    rare_gen[a_idx],
                    smote(X_rare_tr_real, n_synth - half, k=5,
                          rng=rng_local),
                ], axis=0)
            else:
                continue

            X_tr = np.concatenate([X_maj_tr, X_rare_tr_real, synth])
            y_tr = np.concatenate([
                y_maj_tr,
                np.full(n_rare_tr + synth.shape[0], rare_gid,
                        dtype=y_maj_tr.dtype),
            ])
            res = evaluate_strategy(X_tr, y_tr, X_te, y_te, rare_gid)
            res["n_synth"] = int(synth.shape[0])
            out["sweep"][strategy][f"{ratio}x"] = res

    return out


def main() -> None:
    real = np.load(REAL_LATENT).astype(np.float32)
    real_l = np.load(REAL_LABELS)
    gen = np.load(GEN_LATENT).astype(np.float32)
    gen_l = np.load(GEN_LABELS)

    per_rare = {}
    for rare_gid in RARE_GIDS:
        print(f"\n=== rare type gid={rare_gid} ===")
        out = run_one_rare(real, real_l, gen, gen_l, rare_gid)
        per_rare[int(rare_gid)] = out
        # print a short per-rare summary
        print(f"  baseline rare-F1 = {out['baseline']['rare_f1']:.3f}, "
              f"macro-F1 = {out['baseline']['macro_f1']:.3f}")
        for s, ratios in out["sweep"].items():
            line = f"  {s:18s}"
            for r, res in ratios.items():
                line += f"  {r}:{res['rare_f1']:.3f}"
            print(line)

    summary = {
        "seed": SEED,
        "majority_cap": MAJORITY_CAP,
        "test_frac": TEST_FRAC,
        "ratios": RATIOS,
        "rare_gids": RARE_GIDS,
        "inputs": {
            "real_latent": {
                "path": str(REAL_LATENT.relative_to(REPO_ROOT)),
                "sha256_head": sha256_head(REAL_LATENT),
            },
            "generated_latent": {
                "path": str(GEN_LATENT.relative_to(REPO_ROOT)),
                "sha256_head": sha256_head(GEN_LATENT),
            },
        },
        "per_rare_type": per_rare,
    }
    (HERE / "mixing_sweep.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: float(o)
                   if isinstance(o, np.floating) else None) + "\n"
    )

    # preview
    lines = ["B3 — rare-cell mixing sweep (latent scale)",
             "==========================================",
             f"seed={SEED}, majority_cap={MAJORITY_CAP}, "
             f"test_frac={TEST_FRAC}, ratios={RATIOS}",
             ""]
    for rare_gid, out in per_rare.items():
        lines.append(f"=== gid {rare_gid}  "
                     f"(n_real={out['n_real_rare_total']}, "
                     f"train={out['n_real_rare_train']}, "
                     f"test={out['n_real_rare_test']}, "
                     f"clop_rare_available={out['n_clop_rare_available']}) ===")
        b = out["baseline"]
        lines.append(f"  baseline                  rare-F1={b['rare_f1']:.3f}  "
                     f"macro-F1={b['macro_f1']:.3f}")
        header = (f"  {'strategy':18s} {'1x':>11s} {'2x':>11s} "
                  f"{'5x':>11s} {'10x':>11s}")
        lines.append(header)
        for s, ratios in out["sweep"].items():
            row = f"  {s:18s}"
            for r in (f"{rr}x" for rr in RATIOS):
                if r in ratios:
                    row += f"  {ratios[r]['rare_f1']:>9.3f}"
                else:
                    row += "  " + " " * 9 + "-"
            lines.append(row)
        lines.append("")
    (HERE / "mixing_sweep_preview.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
