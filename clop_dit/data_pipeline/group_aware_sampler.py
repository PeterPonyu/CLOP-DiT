import numpy as np
from collections import defaultdict
from typing import Dict, Iterator, List, Optional
from torch.utils.data import Sampler


class GroupAwareBatchSampler(Sampler[List[int]]):
    """Batch sampler that balances text groups and injects hard-negative groups.

    Yields batches of dataset-local indices (0..len(dataset)-1) with:
    - multiple samples per selected group (strong positives)
    - optional nearest-neighbor group sampling (hard negatives)
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        batch_size: int,
        groups_per_batch: int = 128,
        hard_negative_ratio: float = 0.5,
        hard_negative_k: int = 20,
        text_embeddings: Optional[np.ndarray] = None,
        drop_last: bool = True,
        seed: int = 42,
        class_weight_power: float = 0.0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if groups_per_batch <= 0:
            raise ValueError("groups_per_batch must be > 0")

        self.group_ids = np.asarray(group_ids)
        self.batch_size = int(batch_size)
        self.groups_per_batch = int(min(groups_per_batch, batch_size))
        self.hard_negative_ratio = float(np.clip(hard_negative_ratio, 0.0, 1.0))
        self.hard_negative_k = int(max(1, hard_negative_k))
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        self.n_samples = len(self.group_ids)
        self.unique_groups = np.unique(self.group_ids)
        if len(self.unique_groups) == 0:
            raise ValueError("No groups available for GroupAwareBatchSampler")

        self.group_to_local_indices: Dict[int, np.ndarray] = {}
        for gid in self.unique_groups:
            idx = np.where(self.group_ids == gid)[0]
            self.group_to_local_indices[int(gid)] = idx

        # Inverse-frequency weighting for rare-type upsampling.
        # class_weight_power=0 → uniform, =0.5 → sqrt-balanced, =1.0 → fully balanced
        self.class_weight_power = float(class_weight_power)
        if self.class_weight_power > 0:
            counts = np.array([len(self.group_to_local_indices[int(g)])
                               for g in self.unique_groups], dtype=np.float64)
            inv_freq = 1.0 / np.maximum(counts, 1.0)
            weights = inv_freq ** self.class_weight_power
            self.group_sample_probs = weights / weights.sum()
        else:
            self.group_sample_probs = None

        self.group_neighbors: Dict[int, np.ndarray] = {}
        if text_embeddings is not None:
            self._build_neighbors(text_embeddings)

    def _build_neighbors(self, text_embeddings: np.ndarray) -> None:
        emb = np.asarray(text_embeddings, dtype=np.float32)
        if emb.ndim != 2:
            return

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-8)

        valid_groups = [int(g) for g in self.unique_groups if int(g) < emb.shape[0]]
        if len(valid_groups) < 2:
            return

        sub = emb[valid_groups]
        sim = sub @ sub.T

        top_k = min(self.hard_negative_k + 1, sim.shape[1])
        top_idx = np.argpartition(-sim, kth=np.arange(top_k), axis=1)[:, :top_k]

        for row, gid in enumerate(valid_groups):
            cand = [valid_groups[c] for c in top_idx[row] if valid_groups[c] != gid]
            self.group_neighbors[gid] = np.asarray(cand, dtype=np.int64)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size
        return int(np.ceil(self.n_samples / self.batch_size))

    def _sample_groups(self) -> List[int]:
        target = self.groups_per_batch
        selected: List[int] = []
        selected_set = set()

        if self.group_sample_probs is not None:
            seed_gid = int(self.rng.choice(self.unique_groups, p=self.group_sample_probs))
        else:
            seed_gid = int(self.rng.choice(self.unique_groups))
        selected.append(seed_gid)
        selected_set.add(seed_gid)

        while len(selected) < target:
            use_hard = self.rng.random() < self.hard_negative_ratio

            if use_hard and selected:
                anchor_gid = int(self.rng.choice(selected))
                neighbors = self.group_neighbors.get(anchor_gid)
                if neighbors is not None and len(neighbors) > 0:
                    perm = self.rng.permutation(len(neighbors))
                    added = False
                    for p in perm:
                        cand = int(neighbors[p])
                        if cand in self.group_to_local_indices and cand not in selected_set:
                            selected.append(cand)
                            selected_set.add(cand)
                            added = True
                            break
                    if added:
                        continue

            remaining = [int(g) for g in self.unique_groups if int(g) not in selected_set]
            if not remaining:
                break
            if self.group_sample_probs is not None:
                rem_idx = [i for i, g in enumerate(self.unique_groups) if int(g) not in selected_set]
                rem_probs = self.group_sample_probs[rem_idx]
                rem_probs = rem_probs / rem_probs.sum()
                g = int(self.unique_groups[self.rng.choice(rem_idx, p=rem_probs)])
            else:
                g = int(self.rng.choice(remaining))
            selected.append(g)
            selected_set.add(g)

        return selected

    def __iter__(self) -> Iterator[List[int]]:
        n_batches = len(self)
        samples_per_group = max(1, self.batch_size // max(1, self.groups_per_batch))

        for _ in range(n_batches):
            chosen_groups = self._sample_groups()
            batch: List[int] = []

            for gid in chosen_groups:
                candidates = self.group_to_local_indices.get(int(gid))
                if candidates is None or len(candidates) == 0:
                    continue

                if len(candidates) >= samples_per_group:
                    picks = self.rng.choice(candidates, size=samples_per_group, replace=False)
                else:
                    picks = self.rng.choice(candidates, size=samples_per_group, replace=True)

                batch.extend([int(i) for i in picks])
                if len(batch) >= self.batch_size:
                    break

            if len(batch) < self.batch_size:
                need = self.batch_size - len(batch)
                filler = self.rng.choice(np.arange(self.n_samples), size=need, replace=False)
                batch.extend([int(i) for i in filler])

            if len(batch) > self.batch_size:
                batch = batch[: self.batch_size]

            if not self.drop_last or len(batch) == self.batch_size:
                yield batch
