#!/usr/bin/env python3
"""Audit CLOP data pairing quality and split geometry."""

import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/zeyufu/Desktop/CLOP-DiT')
CACHE = ROOT / 'data' / 'cached_latents_v5.2'
OUT = ROOT / 'results' / 'clop_data_audit.json'


def main():
    cell = np.load(CACHE / 'cell_embeddings.npy', mmap_mode='r')
    text_u = np.load(CACHE / 'text_embeddings_unique.npy', mmap_mode='r')
    text_pp = np.load(CACHE / 'text_embeddings_unique_preprocessed.npy', mmap_mode='r')
    text_gid = np.load(CACHE / 'text_group_ids.npy')
    sample = np.load(CACHE / 'sample_ids.npy')
    text_full_pp = np.load(CACHE / 'text_embeddings_preprocessed.npy', mmap_mode='r')

    # mapping consistency: full preprocessed text vs unique_preprocessed[group_id]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(text_gid), size=10000)
    ref = np.asarray(text_pp[text_gid[idx]])
    cur = np.asarray(text_full_pp[idx])

    g_counts = np.bincount(text_gid, minlength=text_u.shape[0])

    # recreate split logic from create_dataloaders (seed=42, val_split=0.1)
    sids = np.unique(sample)
    rng = np.random.default_rng(42)
    shuffled = sids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.1))
    val_ids = set(shuffled[:n_val].tolist())

    train_mask = np.array([sid not in val_ids for sid in sample])
    val_mask = ~train_mask

    train_groups = np.unique(text_gid[train_mask])
    val_groups = np.unique(text_gid[val_mask])

    train_set = set(train_groups.tolist())
    val_set = set(val_groups.tolist())

    # semantic bridge: val→train nearest similarity
    def nearest_stats(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        mx = (a @ b.T).max(axis=1)
        return {
            'mean': float(mx.mean()),
            'p25': float(np.percentile(mx, 25)),
            'median': float(np.median(mx)),
            'p75': float(np.percentile(mx, 75)),
            'min': float(mx.min()),
            'max': float(mx.max()),
        }

    val_to_train_raw = nearest_stats(text_u[val_groups], text_u[train_groups])
    val_to_train_pp = nearest_stats(text_pp[val_groups], text_pp[train_groups])

    text_obj = json.loads((CACHE / 'text_strings.json').read_text())
    if isinstance(text_obj, dict):
        texts = [text_obj[str(i)] for i in range(len(text_obj))]
    else:
        texts = text_obj
    lengths = np.array([len(t) for t in texts])

    report = {
        'shape': {
            'n_cells': int(cell.shape[0]),
            'cell_dim': int(cell.shape[1]),
            'n_text_groups': int(text_u.shape[0]),
            'text_dim': int(text_u.shape[1]),
        },
        'pairing_integrity': {
            'gid_min': int(text_gid.min()),
            'gid_max': int(text_gid.max()),
            'gid_unique': int(len(np.unique(text_gid))),
            'mapping_max_abs_err': float(np.max(np.abs(ref - cur))),
            'mapping_mean_abs_err': float(np.mean(np.abs(ref - cur))),
        },
        'group_distribution': {
            'group_size_min': int(g_counts[g_counts > 0].min()),
            'group_size_median': float(np.median(g_counts[g_counts > 0])),
            'group_size_p75': float(np.percentile(g_counts[g_counts > 0], 75)),
            'group_size_max': int(g_counts.max()),
            'singleton_groups': int((g_counts == 1).sum()),
        },
        'split_geometry': {
            'n_datasets_total': int(len(sids)),
            'n_val_datasets': int(n_val),
            'train_cells': int(train_mask.sum()),
            'val_cells': int(val_mask.sum()),
            'train_groups': int(len(train_set)),
            'val_groups': int(len(val_set)),
            'group_intersection': int(len(train_set & val_set)),
            'val_only_groups': int(len(val_set - train_set)),
            'val_only_ratio': float(len(val_set - train_set) / max(1, len(val_set))),
        },
        'semantic_overlap': {
            'val_to_train_maxcos_raw': val_to_train_raw,
            'val_to_train_maxcos_whitened': val_to_train_pp,
        },
        'text_length': {
            'n_texts': int(len(texts)),
            'mean_chars': float(lengths.mean()),
            'median_chars': float(np.median(lengths)),
            'p95_chars': float(np.percentile(lengths, 95)),
            'max_chars': int(lengths.max()),
        },
        'interpretation': [
            'Pairing integrity is numerically correct if mapping errors are ~1e-6 or lower.',
            'Zero train/val group intersection means this is not class-ID generalization, but semantic transfer.',
            'If whitened nearest-neighbor overlap is much lower than raw overlap, whitening may reduce semantic continuity across groups.',
        ],
    }

    OUT.write_text(json.dumps(report, indent=2))
    print(f'✓ Wrote {OUT}')


if __name__ == '__main__':
    main()
