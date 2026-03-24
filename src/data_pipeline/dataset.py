# dataset.py — PyTorch Dataset classes for CLOP and DiT training
"""
Dataset classes for the CLOP-DiT pipeline.

CLOPDataset: (text_emb, cell_emb) pairs for contrastive alignment
DiTDataset:  (cell_emb, text_emb, t) for flow matching training
InferenceDataset: text descriptions → condition vectors for generation
"""

import json
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, List
import logging

from .group_aware_sampler import GroupAwareBatchSampler

logger = logging.getLogger(__name__)


# ============================================================================
#  CLOP Training Dataset
# ============================================================================

class CLOPDataset(Dataset):
    """Dataset for contrastive text-cell alignment training.

    Loads pre-computed embeddings and forms (text, cell) pairs.
    Supports two storage formats:
    - Legacy (v5/v6.0-6.1): Duplicated text_embeddings.npy (N_total, text_dim)
    - Deduplicated (v6.2+): text_embeddings_unique.npy + text_group_ids.npy

    v6.2 features:
    - Deduplicated text storage (99.5% space savings)
    - Caption variant sampling during training (text augmentation)
    - Explicit text_group_ids for efficient group detection in loss

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing cached embeddings.
    noise_std : float
        Standard deviation of Gaussian noise for cell augmentation.
    sample_level : bool
        If True, sample at the dataset level (one text per dataset).
    use_preprocessed : bool
        If True, load *_preprocessed.npy files (whitened embeddings).
    variant_prob : float
        Probability of sampling a caption variant instead of the primary
        text during training. 0.0 = always use primary, 0.5 = 50% chance
        of a randomly chosen variant. Only works with v6.2 deduplicated cache.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cached_latents_v5.2",
        noise_std: float = 0.0,
        sample_level: bool = False,
        use_preprocessed: bool = False,
        variant_prob: float = 0.0,
        preprocess_text: bool = True,
        preprocess_cell: bool = True,
        text_embeddings_path: Optional[str] = None,
        variant_emb_path: Optional[str] = None,
        variant_map_path: Optional[str] = None,
        use_deduplicated: bool = False,
    ):
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.noise_std = noise_std
        self.sample_level = sample_level
        self.variant_prob = variant_prob
        self._deduplicated = False
        self._has_variants = False
        self.use_deduplicated = use_deduplicated

        # Resolve per-modality preprocessing flags
        # If use_preprocessed=True, default both to True (backward compat)
        # Individual flags can override
        use_pp_text = use_preprocessed and preprocess_text
        use_pp_cell = use_preprocessed and preprocess_cell

        # ── Detect storage format ──
        # Priority: user-specified deduplicated > v6.2 deduplicated > v5 legacy
        if use_deduplicated:
            dedup_path = cache_dir / "text_embeddings_dedup.npy"
            group_id_path = cache_dir / "text_group_ids_dedup.npy"
            if dedup_path.exists() and group_id_path.exists():
                self._deduplicated = True
                logger.info("Using USER-DEDUPLICATED text storage (v5.3+)")
            else:
                logger.warning(f"Requested dedup but files not found in {cache_dir}; falling back to v6.2")
                dedup_path = cache_dir / "text_embeddings_unique.npy"
                group_id_path = cache_dir / "text_group_ids.npy"
                if dedup_path.exists() and group_id_path.exists():
                    self._deduplicated = True
                    logger.info("Using DEDUPLICATED text storage (v6.2)")
                else:
                    logger.info("Using legacy duplicated text storage")
        else:
            dedup_path = cache_dir / "text_embeddings_unique.npy"
            group_id_path = cache_dir / "text_group_ids.npy"

            if dedup_path.exists() and group_id_path.exists():
                self._deduplicated = True
                logger.info("Using DEDUPLICATED text storage (v6.2)")
            else:
                logger.info("Using legacy duplicated text storage")

        # ── Load cell embeddings ──
        if use_deduplicated:
            # Prefer preprocessed dedup > raw dedup > preprocessed full > raw full
            dedup_pp_path = cache_dir / "cell_embeddings_dedup_preprocessed.npy"
            dedup_raw_path = cache_dir / "cell_embeddings_dedup.npy"
            if use_pp_cell and dedup_pp_path.exists():
                cell_path = dedup_pp_path
                logger.info("Loading PREPROCESSED DEDUPLICATED cell embeddings (whitened, unchar removed)")
            elif dedup_raw_path.exists():
                cell_path = dedup_raw_path
                logger.warning(
                    "Loading RAW deduplicated cell embeddings (NOT whitened). "
                    "Run preprocessing to create cell_embeddings_dedup_preprocessed.npy"
                )
            elif use_pp_cell and (cache_dir / "cell_embeddings_preprocessed.npy").exists():
                cell_path = cache_dir / "cell_embeddings_preprocessed.npy"
                logger.info("Loading PREPROCESSED cell embeddings (full set, no dedup filtering)")
            else:
                cell_path = cache_dir / "cell_embeddings.npy"
                logger.warning("Loading RAW cell embeddings (no whitening, no dedup)")
        elif use_pp_cell:
            cell_path = cache_dir / "cell_embeddings_preprocessed.npy"
            if not cell_path.exists():
                logger.warning("Preprocessed cell embeddings not found — falling back to raw.")
                cell_path = cache_dir / "cell_embeddings.npy"
            else:
                logger.info("Loading PREPROCESSED (whitened) cell embeddings")
        else:
            cell_path = cache_dir / "cell_embeddings.npy"
            if use_preprocessed and not preprocess_cell:
                logger.info("Loading RAW cell embeddings (cell whitening disabled)")

        self.cell_emb = np.load(cell_path, mmap_mode="r")

        # Load sample IDs (use deduplicated version if available)
        if use_deduplicated and (cache_dir / "sample_ids_dedup.npy").exists():
            self.sample_ids = np.load(cache_dir / "sample_ids_dedup.npy")
        else:
            self.sample_ids = np.load(cache_dir / "sample_ids.npy")

        # ── Load text embeddings ──
        if self._deduplicated:
            # Custom text embeddings path takes priority over default
            if text_embeddings_path is not None:
                custom_path = Path(text_embeddings_path)
                if not custom_path.is_absolute():
                    # Resolve relative to project root (parent of cache_dir ancestor)
                    custom_path = cache_dir.parent.parent / custom_path
                self.text_emb_unique = np.load(str(custom_path), mmap_mode="r")
                logger.info(f"Loading CUSTOM text embeddings from: {custom_path}")
            else:
                # Try preprocessed deduplicated first, then raw deduplicated, then v6.2
                dedup_text_pp_path = cache_dir / "text_embeddings_dedup_preprocessed.npy"
                dedup_text_path = cache_dir / "text_embeddings_dedup.npy"
                if use_pp_text and use_deduplicated and dedup_text_pp_path.exists():
                    self.text_emb_unique = np.load(str(dedup_text_pp_path), mmap_mode="r")
                    logger.info("Loading PREPROCESSED DEDUPLICATED text embeddings (69 unique, whitened)")
                elif use_deduplicated and dedup_text_path.exists():
                    self.text_emb_unique = np.load(str(dedup_text_path), mmap_mode="r")
                    logger.info("Loading DEDUPLICATED text embeddings (69 unique captions)")
                else:
                    self.text_emb_unique = np.load(str(dedup_path), mmap_mode="r")

                # Load preprocessed unique texts if available (v6.2 only).
                # IMPORTANT: Skip when use_deduplicated=True because
                # text_embeddings_unique_preprocessed.npy has 1088 rows (original
                # sub-clusters), while text_group_ids_dedup.npy maps to indices 0–68.
                # Loading the 1088-row file silently aligns cells to wrong text
                # embeddings (arbitrary rows 0–68 of 1088 ≠ the 69 deduplicated
                # cell-type centroids).
                if use_pp_text and not use_deduplicated:
                    pp_unique = cache_dir / "text_embeddings_unique_preprocessed.npy"
                    if pp_unique.exists():
                        self.text_emb_unique = np.load(pp_unique, mmap_mode="r")
                        logger.info("Loading PREPROCESSED (whitened) unique text embeddings")

            self.text_group_ids = np.load(group_id_path)

            # For legacy compatibility: text_emb used by some code paths
            self.text_emb = None  # Will be fetched via __getitem__

            # Load variant embeddings: custom paths first, then default
            _var_emb = Path(variant_emb_path) if variant_emb_path else cache_dir / "text_variant_embeddings.npy"
            _var_map = Path(variant_map_path) if variant_map_path else cache_dir / "text_variant_map.json"
            if _var_emb.exists() and _var_map.exists() and variant_prob > 0:
                self._variant_embs = np.load(str(_var_emb), mmap_mode="r")
                with open(str(_var_map)) as f:
                    variant_map = json.load(f)  # list of [group_id, variant_idx]

                # Build group_id → list of variant embedding indices
                self._group_variant_indices = defaultdict(list)
                for emb_idx, (gid, _) in enumerate(variant_map):
                    self._group_variant_indices[gid].append(emb_idx)
                self._has_variants = len(self._group_variant_indices) > 0
                if self._has_variants:
                    logger.info(
                        f"Loaded {self._variant_embs.shape[0]} caption variant embeddings "
                        f"for {len(self._group_variant_indices)} text groups"
                    )
        else:
            # Legacy format: full duplicated text embeddings
            if use_pp_text:
                text_path = cache_dir / "text_embeddings_preprocessed.npy"
                if not text_path.exists():
                    logger.warning("Preprocessed text embeddings not found — falling back to raw.")
                    text_path = cache_dir / "text_embeddings.npy"
                else:
                    logger.info("Loading PREPROCESSED (whitened) text embeddings")
            else:
                text_path = cache_dir / "text_embeddings.npy"

            self.text_emb = np.load(text_path, mmap_mode="r")
            self.text_group_ids = None

        # ── Sample-level aggregation ──
        if sample_level:
            unique_ids = np.unique(self.sample_ids)
            self._cell_means = []
            self._text_reps = []
            for sid in unique_ids:
                mask = self.sample_ids == sid
                self._cell_means.append(self.cell_emb[mask].mean(axis=0))
                if self._deduplicated:
                    gid = self.text_group_ids[mask][0]
                    self._text_reps.append(self.text_emb_unique[gid])
                else:
                    self._text_reps.append(self.text_emb[mask][0])
            self._cell_means = np.stack(self._cell_means)
            self._text_reps = np.stack(self._text_reps)

        # Validate
        n_cells = len(self.cell_emb)
        if not self._deduplicated:
            assert len(self.text_emb) == n_cells, \
                f"Mismatch: {n_cells} cells vs {len(self.text_emb)} texts"
        else:
            assert len(self.text_group_ids) == n_cells, \
                f"Mismatch: {n_cells} cells vs {len(self.text_group_ids)} group IDs"

    def __len__(self) -> int:
        if self.sample_level:
            return len(self._cell_means)
        return len(self.cell_emb)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.sample_level:
            cell = torch.from_numpy(self._cell_means[idx].copy()).float()
            text = torch.from_numpy(self._text_reps[idx].copy()).float()
            sid = idx
            gid = -1  # no group info in sample_level mode
        else:
            cell = torch.from_numpy(np.array(self.cell_emb[idx])).float()
            sid = int(self.sample_ids[idx])

            if self._deduplicated:
                gid = int(self.text_group_ids[idx])

                # Caption variant sampling (text augmentation)
                if self._has_variants and self.variant_prob > 0 and \
                   np.random.random() < self.variant_prob and \
                   gid in self._group_variant_indices:
                    # Sample a random variant embedding
                    var_indices = self._group_variant_indices[gid]
                    var_idx = var_indices[np.random.randint(len(var_indices))]
                    text = torch.from_numpy(np.array(self._variant_embs[var_idx])).float()
                else:
                    text = torch.from_numpy(np.array(self.text_emb_unique[gid])).float()
            else:
                text = torch.from_numpy(np.array(self.text_emb[idx])).float()
                gid = -1  # no group info in non-deduplicated mode

        # Cell noise augmentation
        if self.noise_std > 0 and self.training_mode:
            cell = cell + torch.randn_like(cell) * self.noise_std

        return text, cell, sid, gid

    @property
    def training_mode(self) -> bool:
        return self.noise_std > 0

    @property
    def cell_dim(self) -> int:
        return self.cell_emb.shape[1]

    @property
    def text_dim(self) -> int:
        if self._deduplicated:
            return self.text_emb_unique.shape[1]
        return self.text_emb.shape[1]

    @property
    def num_text_groups(self) -> int:
        """Number of unique text groups (for PrototypeSigLIP)."""
        if self._deduplicated:
            return self.text_emb_unique.shape[0]
        return -1  # Unknown without dedup


# ============================================================================
#  DiT Training Dataset
# ============================================================================

class DiTDataset(Dataset):
    """Dataset for Flow Matching DiT training.

    Provides (z_1, text_condition) pairs where z_1 is the real cell embedding.
    Noise z_0 and timestep t are sampled on-the-fly during training.

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing cached embeddings.
    clop_aligner : nn.Module, optional
        Trained CLOP aligner for projecting text embeddings.
        If None, raw text embeddings are used as conditions.
    time_sampling : str
        Timestep sampling strategy:
        - 'uniform': t ~ U[0, 1] (standard)
        - 'logit_normal': t ~ sigma(N(mean, std)) (SD3/Flux-style,
          concentrates samples near t=0.5 where the velocity field
          has the highest curvature and is hardest to learn)
    time_sampling_mean : float
        Mean for logit-normal sampling (default 0.0 → centered at t=0.5).
    time_sampling_std : float
        Std for logit-normal sampling (default 1.0).
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cached_latents_v5.2",
        projected_text_path: Optional[Union[str, Path]] = None,
        time_sampling: str = "logit_normal",
        time_sampling_mean: float = 0.0,
        time_sampling_std: float = 1.0,
        use_deduplicated: bool = False,
        use_preprocessed: bool = False,
    ):
        cache_dir = Path(cache_dir)

        # Time sampling config
        self.time_sampling = time_sampling
        self.time_sampling_mean = time_sampling_mean
        self.time_sampling_std = time_sampling_std

        # ── Load cell embeddings (the generation TARGET z_1) ──
        # Priority: dedup+preprocessed > dedup raw > preprocessed > raw
        cell_path = None
        if use_deduplicated:
            dedup_pp = cache_dir / "cell_embeddings_dedup_preprocessed.npy"
            dedup_raw = cache_dir / "cell_embeddings_dedup.npy"
            if use_preprocessed and dedup_pp.exists():
                cell_path = dedup_pp
                logger.info("DiTDataset: loading PREPROCESSED DEDUP cell embeddings (whitened)")
            elif dedup_raw.exists():
                cell_path = dedup_raw
                logger.warning("DiTDataset: loading RAW dedup cell embeddings (not whitened!)")
        if cell_path is None:
            pp_path = cache_dir / "cell_embeddings_preprocessed.npy"
            raw_path = cache_dir / "cell_embeddings.npy"
            if use_preprocessed and pp_path.exists():
                cell_path = pp_path
                logger.info("DiTDataset: loading preprocessed cell embeddings")
            else:
                cell_path = raw_path
                logger.info("DiTDataset: loading raw cell embeddings")

        self.cell_emb = np.load(cell_path)

        # ── Load text conditions (the CLOP-projected condition c) ──
        if projected_text_path and Path(projected_text_path).exists():
            self.text_cond = np.load(projected_text_path)
        else:
            # Try multiple fallback paths (v6.2+ removed text_embeddings.npy)
            for candidate in ["projected_text.npy",
                              "text_embeddings_preprocessed.npy",
                              "text_embeddings.npy"]:
                p = cache_dir / candidate
                if p.exists():
                    self.text_cond = np.load(p)
                    logger.info(f"DiTDataset: loaded text conditions from {candidate}")
                    break
            else:
                raise FileNotFoundError(
                    f"No text embedding file found in {cache_dir}. "
                    f"Expected projected_text.npy (from CLOP), "
                    f"text_embeddings_preprocessed.npy, or text_embeddings.npy"
                )

        # ── Load sample IDs (use dedup version when available) ──
        if use_deduplicated and (cache_dir / "sample_ids_dedup.npy").exists():
            self.sample_ids = np.load(cache_dir / "sample_ids_dedup.npy")
        else:
            self.sample_ids = np.load(cache_dir / "sample_ids.npy")

        # ── Validate shapes match ──
        if self.cell_emb.shape[0] != self.text_cond.shape[0]:
            raise ValueError(
                f"Shape mismatch: cell_emb has {self.cell_emb.shape[0]} rows "
                f"but text_cond has {self.text_cond.shape[0]} rows. "
                f"Ensure both use the same dedup/full version. "
                f"Cell source: {cell_path.name}, Text source: projected_text.npy"
            )

        logger.info(
            f"DiTDataset: {self.cell_emb.shape[0]} cells × "
            f"{self.cell_emb.shape[1]}d (cell) + {self.text_cond.shape[1]}d (cond)"
        )

        # Pre-convert to torch tensors for zero-copy __getitem__
        self._cell_tensor = torch.from_numpy(self.cell_emb).float()
        self._cond_tensor = torch.from_numpy(self.text_cond).float()

        # ── Load group IDs for per-class variance loss ──
        self.group_ids = None
        for gid_name in ["text_group_ids_dedup.npy", "text_group_ids.npy"]:
            gid_path = cache_dir / gid_name
            if gid_path.exists():
                gid_arr = np.load(gid_path)
                if len(gid_arr) == len(self.cell_emb):
                    self.group_ids = gid_arr
                    self._gid_tensor = torch.from_numpy(gid_arr).long()
                    logger.info(f"DiTDataset: loaded {gid_name} ({len(np.unique(gid_arr))} groups)")
                    break
        if self.group_ids is None:
            logger.info("DiTDataset: no group_ids found (per-class variance loss unavailable)")

    def _sample_timestep(self) -> torch.Tensor:
        """Sample a timestep t ∈ (0, 1).

        - 'uniform': t ~ U[0, 1] (standard baseline)
        - 'logit_normal': t = sigmoid(N(mean, std²))
          Concentrates density near t=0.5 where the velocity field
          has highest curvature. Used by SD3 and Flux.
        """
        if self.time_sampling == "logit_normal":
            # Sample from logit-normal: sigmoid(N(mean, std))
            u = torch.normal(
                mean=self.time_sampling_mean,
                std=self.time_sampling_std,
                size=(1,),
            )
            t = torch.sigmoid(u).squeeze()
            # Clamp to avoid numerical issues at boundaries
            t = t.clamp(1e-5, 1.0 - 1e-5)
        else:
            t = torch.rand(1).squeeze()
        return t

    def __len__(self) -> int:
        return len(self.cell_emb)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        z_1 = self._cell_tensor[idx]
        cond = self._cond_tensor[idx]

        # Sample noise and timestep on-the-fly
        z_0 = torch.randn_like(z_1)
        t = self._sample_timestep()

        # Flow matching interpolation: z_t = (1-t) * z_0 + t * z_1
        z_t = (1 - t) * z_0 + t * z_1

        # Target velocity: v = z_1 - z_0
        v_target = z_1 - z_0

        result = {
            "z_t": z_t,           # Noisy interpolated embedding
            "t": t,               # Timestep
            "v_target": v_target,  # Target velocity
            "cond": cond,          # Text condition
            "z_1": z_1,           # Real embedding (for evaluation)
        }

        # Include group_id for per-class variance loss (if available)
        if self.group_ids is not None:
            result["group_id"] = self._gid_tensor[idx]

        return result

    @property
    def latent_dim(self) -> int:
        return self.cell_emb.shape[1]

    @property
    def cond_dim(self) -> int:
        return self.text_cond.shape[1]


# ============================================================================
#  Inference Dataset (Text → Generation)
# ============================================================================

class InferenceDataset(Dataset):
    """Dataset for inference: takes text descriptions and returns condition vectors.

    Parameters
    ----------
    descriptions : list of str
        Natural language biological descriptions.
    text_encoder_name : str
        HuggingFace model name for text encoding.
    clop_projector : nn.Module, optional
        Trained CLOP text projector.
    num_cells_per_condition : int
        Number of cells to generate per condition.
    device : str
        Device for encoding.
    """

    def __init__(
        self,
        descriptions: list,
        text_encoder_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        clop_projector=None,
        num_cells_per_condition: int = 100,
        device: str = "cuda",
    ):
        self.descriptions = descriptions
        self.num_cells = num_cells_per_condition
        self.device = device

        # Encode texts
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        model = AutoModel.from_pretrained(text_encoder_name).to(device)
        model.eval()

        with torch.no_grad():
            inputs = tokenizer(
                descriptions, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            text_emb = outputs.last_hidden_state[:, 0, :].cpu()

        # Project through CLOP if available
        if clop_projector is not None:
            with torch.no_grad():
                text_emb = clop_projector(text_emb.to(device)).cpu()

        self.text_emb = text_emb  # (N_conds, proj_dim)

        # Repeat for num_cells_per_condition
        self.conditions = text_emb.repeat_interleave(num_cells_per_condition, dim=0)

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.conditions[idx]


# ============================================================================
#  DataLoader Factory
# ============================================================================

def create_dataloaders(
    cache_dir: str,
    batch_size: int = 256,
    val_split: float = 0.1,
    n_folds: int = 1,
    fold_idx: int = 0,
    num_workers: int = 4,
    stage: str = "clop",
    projected_text_path: Optional[str] = None,
    time_sampling: str = "logit_normal",
    time_sampling_mean: float = 0.0,
    time_sampling_std: float = 1.0,
    use_preprocessed: bool = False,
    variant_prob: float = 0.0,
    preprocess_text: bool = True,
    preprocess_cell: bool = True,
    group_aware_sampling: bool = False,
    groups_per_batch: int = 128,
    hard_negative_ratio: float = 0.5,
    hard_negative_k: int = 20,
    text_embeddings_path: Optional[str] = None,
    variant_emb_path: Optional[str] = None,
    variant_map_path: Optional[str] = None,
    use_deduplicated: bool = False,
    split_strategy: str = "stratified",
    class_weight_power: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for CLOP or DiT training.

    Parameters
    ----------
    cache_dir : str
        Path to cached latents directory.
    batch_size : int
    val_split : float
        Fraction of data for validation (used when n_folds <= 1).
    n_folds : int
        Number of group-level folds for cross-validation. If > 1, val_split
        is ignored and fold-based split is used.
    fold_idx : int
        Validation fold index in [0, n_folds-1] when n_folds > 1.
    num_workers : int
    stage : str
        'clop' for CLOP alignment, 'dit' for DiT flow matching.
    projected_text_path : str, optional
        Path to CLOP-projected text embeddings (for DiT stage).
    time_sampling : str
        Timestep sampling strategy for DiT ('uniform' or 'logit_normal').
    time_sampling_mean : float
        Mean for logit-normal sampling.
    time_sampling_std : float
        Std for logit-normal sampling.
    use_preprocessed : bool
        If True, load preprocessed (whitened) embeddings for CLOP.
    split_strategy : str
        'stratified' (default): Cell-type-stratified split ensuring ALL cell
        types appear in both train and val.  For each type, ~val_split
        fraction of cells are held out.  This prevents the blind-spot
        problem where dataset-level splits can leave many types unevaluated.
        'dataset': Legacy dataset-level split by sample_id.  Conservative
        against batch-effect leakage but may leave many cell types absent
        from validation.

    Returns
    -------
    train_loader, val_loader : DataLoader pair
    """
    if stage == "clop":
        dataset = CLOPDataset(
            cache_dir,
            use_preprocessed=use_preprocessed,
            variant_prob=variant_prob,
            preprocess_text=preprocess_text,
            preprocess_cell=preprocess_cell,
            text_embeddings_path=text_embeddings_path,
            variant_emb_path=variant_emb_path,
            variant_map_path=variant_map_path,
            use_deduplicated=use_deduplicated,
        )
    elif stage == "dit":
        dataset = DiTDataset(
            cache_dir,
            projected_text_path=projected_text_path,
            time_sampling=time_sampling,
            time_sampling_mean=time_sampling_mean,
            time_sampling_std=time_sampling_std,
            use_deduplicated=use_deduplicated,
            use_preprocessed=use_preprocessed,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # ── Split strategy selection ──
    # "stratified": Cell-type-stratified split.  For each cell type,
    #   ~val_split of its cells go to validation.  Guarantees ALL types
    #   appear in both train and val, eliminating the blind-spot problem
    #   where dataset-level splits leave many types unevaluated.
    # "dataset": Legacy dataset-level split by sample_id.  Conservative
    #   against batch-effect leakage but may miss entire cell types.
    sample_ids = dataset.sample_ids
    unique_ids = np.unique(sample_ids)
    rng = np.random.default_rng(42)

    use_stratified = (
        split_strategy == "stratified"
        and stage == "clop"
        and hasattr(dataset, "text_group_ids")
        and dataset.text_group_ids is not None
    )

    if use_stratified:
        # ── Stratified split by cell type ──
        # For each cell type, hold out ~val_split fraction of cells.
        # This ensures ALL types are represented in validation.
        group_ids = np.asarray(dataset.text_group_ids)
        unique_types = np.unique(group_ids)

        train_indices = []
        val_indices = []

        for gid in unique_types:
            type_indices = np.where(group_ids == gid)[0]
            rng.shuffle(type_indices)

            n_val = max(1, int(len(type_indices) * val_split))
            val_indices.extend(type_indices[:n_val].tolist())
            train_indices.extend(type_indices[n_val:].tolist())

        # Shuffle to avoid ordering artifacts
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        # Count types per split for logging
        train_types = len(np.unique(group_ids[train_indices]))
        val_types = len(np.unique(group_ids[val_indices]))

        logger.info(
            f"Stratified split: {len(train_indices)} train cells "
            f"({train_types} types) / {len(val_indices)} val cells "
            f"({val_types} types) — ALL {len(unique_types)} types in both splits"
        )

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

    elif len(unique_ids) < 2:
        # Fallback: only 1 dataset, use random cell-level split
        logger.warning("Only 1 unique sample_id — falling back to random split")
        n_total = len(dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        # ── Dataset-level split by sample_id ──
        # Conservative choice against batch-effect leakage.
        # WARNING: May leave many cell types absent from validation.
        shuffled_ids = unique_ids.copy()
        rng.shuffle(shuffled_ids)

        # K-fold group split (preferred when requested)
        if n_folds and n_folds > 1:
            effective_folds = min(int(n_folds), len(shuffled_ids))
            if effective_folds != n_folds:
                logger.warning(
                    f"Requested n_folds={n_folds} but only {len(shuffled_ids)} groups available; "
                    f"using n_folds={effective_folds}."
                )

            if fold_idx < 0 or fold_idx >= effective_folds:
                raise ValueError(
                    f"fold_idx={fold_idx} out of range for n_folds={effective_folds}."
                )

            fold_chunks = np.array_split(shuffled_ids, effective_folds)
            val_ids = fold_chunks[fold_idx]
            n_val_groups = len(val_ids)
            val_id_set = set(val_ids.tolist())

            logger.info(
                f"K-fold group split enabled: fold {fold_idx + 1}/{effective_folds}"
            )
        else:
            n_val_groups = max(1, int(len(shuffled_ids) * val_split))
            val_id_set = set(shuffled_ids[:n_val_groups].tolist())

        train_indices = [i for i, sid in enumerate(sample_ids) if sid not in val_id_set]
        val_indices = [i for i, sid in enumerate(sample_ids) if sid in val_id_set]

        logger.info(
            f"Dataset-level split: {len(shuffled_ids) - n_val_groups} train datasets "
            f"({len(train_indices)} cells) / {n_val_groups} val datasets "
            f"({len(val_indices)} cells)"
        )

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

    if stage in ("clop", "dit") and group_aware_sampling:
        if hasattr(train_dataset, "dataset") and hasattr(train_dataset, "indices"):
            base_dataset = train_dataset.dataset
            local_to_global = np.asarray(train_dataset.indices)
        else:
            base_dataset = train_dataset
            local_to_global = np.arange(len(train_dataset))

        # Get group IDs from the appropriate attribute (CLOP vs DiT)
        _group_ids_raw = None
        if hasattr(base_dataset, "text_group_ids") and base_dataset.text_group_ids is not None:
            _group_ids_raw = np.asarray(base_dataset.text_group_ids)
        elif hasattr(base_dataset, "group_ids") and base_dataset.group_ids is not None:
            _group_ids_raw = np.asarray(base_dataset.group_ids)

        if _group_ids_raw is not None:
            local_group_ids = _group_ids_raw[local_to_global]
            text_embeddings = getattr(base_dataset, "text_emb_unique", None)

            batch_sampler = GroupAwareBatchSampler(
                group_ids=local_group_ids,
                batch_size=batch_size,
                groups_per_batch=groups_per_batch,
                hard_negative_ratio=hard_negative_ratio,
                hard_negative_k=hard_negative_k,
                text_embeddings=text_embeddings,
                drop_last=True,
                class_weight_power=class_weight_power,
            )

            logger.info(
                "Using GroupAwareBatchSampler: "
                f"groups_per_batch={groups_per_batch}, "
                f"hard_negative_ratio={hard_negative_ratio}, "
                f"hard_negative_k={hard_negative_k}"
            )

            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
        else:
            logger.warning(
                "group_aware_sampling requested but text_group_ids unavailable; "
                "falling back to shuffle=True"
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=num_workers > 0,
            )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

    # Shuffle=True for validation: contrastive accuracy requires mixed
    # batches (cells from different datasets/texts).  Without shuffling,
    # data is sorted by sample_id → each batch has identical text embeddings
    # → acc ≈ 1/B (random chance).  Shuffling ensures diverse text content
    # within each batch for meaningful contrastive evaluation.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
