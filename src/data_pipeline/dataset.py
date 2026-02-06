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
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Union, Dict, Tuple


# ============================================================================
#  CLOP Training Dataset
# ============================================================================

class CLOPDataset(Dataset):
    """Dataset for contrastive text-cell alignment training.

    Loads pre-computed embeddings and forms (text, cell) pairs.
    Optionally applies data augmentation via embedding noise.

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing cached embeddings.
    noise_std : float
        Standard deviation of Gaussian noise for augmentation.
        0.0 = no augmentation.
    sample_level : bool
        If True, sample at the dataset level (one text per dataset).
        If False, sample at the cell level (duplicated texts).
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cached_latents",
        noise_std: float = 0.0,
        sample_level: bool = False,
    ):
        cache_dir = Path(cache_dir)

        self.cell_emb = np.load(cache_dir / "cell_embeddings.npy", mmap_mode="r")
        self.text_emb = np.load(cache_dir / "text_embeddings.npy", mmap_mode="r")
        self.sample_ids = np.load(cache_dir / "sample_ids.npy")
        self.noise_std = noise_std
        self.sample_level = sample_level

        if sample_level:
            # Aggregate to sample level: use mean cell embedding per sample
            unique_ids = np.unique(self.sample_ids)
            self._cell_means = []
            self._text_reps = []
            for sid in unique_ids:
                mask = self.sample_ids == sid
                self._cell_means.append(self.cell_emb[mask].mean(axis=0))
                self._text_reps.append(self.text_emb[mask][0])
            self._cell_means = np.stack(self._cell_means)
            self._text_reps = np.stack(self._text_reps)

        assert len(self.cell_emb) == len(self.text_emb), \
            f"Mismatch: {len(self.cell_emb)} cells vs {len(self.text_emb)} texts"

    def __len__(self) -> int:
        if self.sample_level:
            return len(self._cell_means)
        return len(self.cell_emb)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.sample_level:
            cell = torch.from_numpy(self._cell_means[idx].copy()).float()
            text = torch.from_numpy(self._text_reps[idx].copy()).float()
            sid = idx
        else:
            cell = torch.from_numpy(np.array(self.cell_emb[idx])).float()
            text = torch.from_numpy(np.array(self.text_emb[idx])).float()
            sid = int(self.sample_ids[idx])

        # Augmentation
        if self.noise_std > 0 and self.training_mode:
            cell = cell + torch.randn_like(cell) * self.noise_std

        return text, cell, sid

    @property
    def training_mode(self) -> bool:
        return self.noise_std > 0

    @property
    def cell_dim(self) -> int:
        return self.cell_emb.shape[1]

    @property
    def text_dim(self) -> int:
        return self.text_emb.shape[1]


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
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "data/cached_latents",
        projected_text_path: Optional[Union[str, Path]] = None,
    ):
        cache_dir = Path(cache_dir)

        self.cell_emb = np.load(cache_dir / "cell_embeddings.npy", mmap_mode="r")

        # Use projected text if available, otherwise raw
        if projected_text_path and Path(projected_text_path).exists():
            self.text_cond = np.load(projected_text_path, mmap_mode="r")
        else:
            self.text_cond = np.load(cache_dir / "text_embeddings.npy", mmap_mode="r")

        self.sample_ids = np.load(cache_dir / "sample_ids.npy")

    def __len__(self) -> int:
        return len(self.cell_emb)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        z_1 = torch.from_numpy(np.array(self.cell_emb[idx])).float()
        cond = torch.from_numpy(np.array(self.text_cond[idx])).float()

        # Sample noise and timestep on-the-fly
        z_0 = torch.randn_like(z_1)
        t = torch.rand(1).squeeze()  # t ∈ [0, 1]

        # Flow matching interpolation: z_t = (1-t) * z_0 + t * z_1
        z_t = (1 - t) * z_0 + t * z_1

        # Target velocity: v = z_1 - z_0
        v_target = z_1 - z_0

        return {
            "z_t": z_t,           # Noisy interpolated embedding
            "t": t,               # Timestep
            "v_target": v_target,  # Target velocity
            "cond": cond,          # Text condition
            "z_1": z_1,           # Real embedding (for evaluation)
        }

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
    num_workers: int = 4,
    stage: str = "clop",
    projected_text_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for CLOP or DiT training.

    Parameters
    ----------
    cache_dir : str
        Path to cached latents directory.
    batch_size : int
    val_split : float
        Fraction of data for validation.
    num_workers : int
    stage : str
        'clop' for CLOP alignment, 'dit' for DiT flow matching.
    projected_text_path : str, optional
        Path to CLOP-projected text embeddings (for DiT stage).

    Returns
    -------
    train_loader, val_loader : DataLoader pair
    """
    if stage == "clop":
        dataset = CLOPDataset(cache_dir)
    elif stage == "dit":
        dataset = DiTDataset(cache_dir, projected_text_path=projected_text_path)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Split
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
