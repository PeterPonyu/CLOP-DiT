# run_metrics.py — Shared API for CLOP-DiT evaluation
"""
Single entry point for loading models and running embedding-space metrics.
Used by scripts/06_evaluate.py, scripts/10_full_pipeline.py, and scripts/run_5fold_cv.py
so defaults and logic stay consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_clop(checkpoint_path: str | Path, device: str = "cuda"):
    """Load trained CLOP model from checkpoint.

    Uses config stored in checkpoint when present; otherwise sensible defaults
    compatible with both minimal (06_evaluate) and full (04a/10) checkpoints.
    """
    from src.architecture.clop import CLOPAligner

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = CLOPAligner(
        text_dim=config.get("text_dim", 1024),
        cell_dim=config.get("cell_dim", 512),
        proj_dim=config.get("proj_dim", 256),
        text_layers=config.get("text_layers", 3),
        cell_layers=config.get("cell_layers", 3),
        dropout=config.get("dropout", 0.1),
        use_batch_norm=config.get("use_batch_norm", True),
        label_smoothing=config.get("label_smoothing", 0.1),
        loss_type=config.get("loss_type", "prototype_siglip"),
        auto_duplicate_mask=config.get("auto_duplicate_mask", True),
        temperature=config.get("temperature", 10.0),
        use_whitening=config.get("use_whitening", False),
        cohesion_weight=config.get("cohesion_weight", 0.1),
        max_temperature=config.get("max_temperature", 100.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, config


def load_dit(
    checkpoint_path: str | Path,
    cond_dim: int | None = None,
    clop_config: dict | None = None,
    device: str = "cuda",
):
    """Load trained DiT model from checkpoint.

    cond_dim defaults to clop_config['proj_dim'] if clop_config is provided,
    else 256.
    """
    from src.architecture.dit import DiT1D

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    if cond_dim is None and clop_config is not None:
        cond_dim = clop_config.get("proj_dim", 256)
    if cond_dim is None:
        cond_dim = 256

    model = DiT1D(
        latent_dim=config.get("latent_dim", 512),
        hidden_dim=config.get("hidden_dim", 384),
        cond_dim=cond_dim,
        num_tokens=config.get("num_tokens", 16),
    )
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        logger.info("  Using EMA weights for DiT")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, config


def load_models_for_pipeline(
    clop_checkpoint: str | Path,
    dit_checkpoint: str | Path,
    device: str = "cuda",
):
    """Load CLOP and DiT models plus raw checkpoints (for 10_full_pipeline epoch/metrics)."""
    clop_ckpt = torch.load(clop_checkpoint, map_location=device, weights_only=False)
    dit_ckpt = torch.load(dit_checkpoint, map_location=device, weights_only=False)
    clop, _ = load_clop(clop_checkpoint, device)
    dit, _ = load_dit(dit_checkpoint, clop_config=clop_ckpt.get("config", {}), device=device)
    return clop, dit, clop_ckpt, dit_ckpt


@torch.no_grad()
def evaluate_clop(clop_model, cache_dir: str | Path, device: str = "cuda") -> dict[str, Any]:
    """Evaluate CLOP alignment quality (retrieval, cosine similarity)."""
    from src.evaluation.metrics import GenerationMetrics

    logger.info("\n" + "=" * 60)
    logger.info("CLOP Alignment Evaluation")
    logger.info("=" * 60)

    cache_path = Path(cache_dir)
    cell_emb = np.load(cache_path / "cell_embeddings.npy")

    text_emb = None
    for candidate in ["text_embeddings_preprocessed.npy", "text_embeddings.npy"]:
        p = cache_path / candidate
        if p.exists():
            text_emb = np.load(p)
            break
    if text_emb is None:
        for candidate in [
            "text_embeddings_unique_preprocessed.npy",
            "text_embeddings_unique.npy",
            "text_embeddings_dedup.npy",
        ]:
            p = cache_path / candidate
            if p.exists():
                text_emb_unique = np.load(p)
                for gid_name in ["text_group_ids.npy", "text_group_ids_dedup.npy"]:
                    gid_path = cache_path / gid_name
                    if gid_path.exists():
                        group_ids = np.load(gid_path)
                        text_emb = text_emb_unique[group_ids]
                        break
                if text_emb is not None:
                    break
    if text_emb is None:
        raise FileNotFoundError(
            f"No text embedding file found in {cache_dir}. "
            "Expected text_embeddings*.npy and optionally text_group_ids*.npy."
        )

    sample_ids = np.load(cache_path / "sample_ids.npy")

    cell_proj_list = []
    text_proj_list = []
    batch_size = 512
    for i in range(0, len(cell_emb), batch_size):
        c_batch = torch.from_numpy(cell_emb[i : i + batch_size]).float().to(device)
        t_batch = torch.from_numpy(text_emb[i : i + batch_size]).float().to(device)
        cell_proj_list.append(clop_model.project_cell(c_batch).cpu().numpy())
        text_proj_list.append(clop_model.project_text(t_batch).cpu().numpy())

    cell_proj = np.concatenate(cell_proj_list, axis=0)
    text_proj = np.concatenate(text_proj_list, axis=0)

    unique_ids = np.unique(sample_ids)
    sample_cell_proj = np.array([cell_proj[sample_ids == sid].mean(axis=0) for sid in unique_ids])
    sample_text_proj = np.array([text_proj[sample_ids == sid].mean(axis=0) for sid in unique_ids])
    sample_cell_proj = sample_cell_proj / (np.linalg.norm(sample_cell_proj, axis=1, keepdims=True) + 1e-8)
    sample_text_proj = sample_text_proj / (np.linalg.norm(sample_text_proj, axis=1, keepdims=True) + 1e-8)

    retrieval_metrics = GenerationMetrics.clop_retrieval(
        sample_text_proj, sample_cell_proj, k_values=(1, 3, 5, 10)
    )
    cosine_sims = np.sum(cell_proj * text_proj, axis=1) / (
        np.linalg.norm(cell_proj, axis=1) * np.linalg.norm(text_proj, axis=1) + 1e-8
    )

    return {
        "retrieval": retrieval_metrics,
        "cosine_sim_mean": float(cosine_sims.mean()),
        "cosine_sim_std": float(cosine_sims.std()),
        "cell_proj": cell_proj,
        "text_proj": text_proj,
        "sample_ids": sample_ids,
    }


@torch.no_grad()
def evaluate_generation(
    dit_model,
    clop_model,
    cache_dir: str | Path,
    num_samples: int = 500,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    device: str = "cuda",
) -> dict[str, Any] | None:
    """Evaluate DiT generation quality (FD, MMD, coverage, density, KL)."""
    from src.evaluation.metrics import GenerationMetrics

    logger.info("\n" + "=" * 60)
    logger.info("DiT Generation Evaluation")
    logger.info("=" * 60)

    cache_path = Path(cache_dir)
    projected_path = cache_path / "projected_text.npy"
    if not projected_path.exists():
        logger.error("projected_text.npy not found. Run CLOP training first.")
        return None

    cell_emb = np.load(cache_path / "cell_embeddings.npy")
    projected_text = np.load(projected_path)
    sample_ids = np.load(cache_path / "sample_ids.npy")

    unique_ids = np.unique(sample_ids)
    conditions_list = []
    real_list = []
    for sid in unique_ids:
        mask = sample_ids == sid
        n_per = min(num_samples // max(1, len(unique_ids)) + 1, mask.sum())
        idx = np.where(mask)[0][:n_per]
        conditions_list.append(projected_text[idx])
        real_list.append(cell_emb[idx])

    conditions = np.concatenate(conditions_list, axis=0)[:num_samples]
    real = np.concatenate(real_list, axis=0)[:num_samples]

    cond_tensor = torch.from_numpy(conditions).float().to(device)
    generated_list = []
    batch_size = 256
    for i in range(0, len(cond_tensor), batch_size):
        batch = cond_tensor[i : i + batch_size]
        gen = dit_model.sample(batch, num_steps=num_steps, cfg_scale=cfg_scale)
        generated_list.append(gen.cpu().numpy())
    generated = np.concatenate(generated_list, axis=0)

    metrics = GenerationMetrics.full_evaluation(real, generated)
    real_norms = np.linalg.norm(real, axis=1)
    gen_norms = np.linalg.norm(generated, axis=1)
    cosine_sims = np.sum(generated * real, axis=1) / (gen_norms * real_norms + 1e-8)
    metrics["real_norm_mean"] = float(real_norms.mean())
    metrics["gen_norm_mean"] = float(gen_norms.mean())
    metrics["paired_cosine_mean"] = float(cosine_sims.mean())

    return {
        "metrics": metrics,
        "real": real,
        "generated": generated,
    }


def run_embedding_metrics(
    cache_dir: str | Path,
    clop_checkpoint: str | Path,
    dit_checkpoint: str | Path,
    output_dir: str | Path | None = None,
    num_samples: int = 500,
    num_steps: int = 20,
    cfg_scale: float = 3.0,
    device: str = "cuda",
    skip_clop: bool = False,
    skip_gen: bool = False,
) -> dict[str, Any]:
    """Run CLOP and generation evaluation; return combined metrics and raw results for viz.

    Returns
    -------
    result : dict
        "clop": clop metrics dict (or None if skip_clop)
        "generation": generation metrics dict (or None if skip_gen)
        "clop_results": full clop_results for visualization (or None)
        "gen_results": full gen_results (real, generated) for visualization (or None)
    """
    cache_path = Path(cache_dir)
    clop_path = Path(clop_checkpoint)
    dit_path = Path(dit_checkpoint)

    out = {
        "clop": None,
        "generation": None,
        "clop_results": None,
        "gen_results": None,
    }

    clop_model = None
    clop_config = {}

    if not skip_clop and clop_path.exists():
        clop_model, clop_config = load_clop(clop_path, device)
        out["clop_results"] = evaluate_clop(clop_model, cache_dir, device)
        out["clop"] = {
            "retrieval": out["clop_results"]["retrieval"],
            "cosine_sim_mean": out["clop_results"]["cosine_sim_mean"],
            "cosine_sim_std": out["clop_results"]["cosine_sim_std"],
        }

    cond_dim = clop_config.get("proj_dim", 256)
    if not skip_gen and dit_path.exists():
        if clop_model is None and clop_path.exists():
            _, clop_config = load_clop(clop_path, device)
            cond_dim = clop_config.get("proj_dim", 256)
        dit_model, _ = load_dit(dit_path, cond_dim=cond_dim, device=device)
        if clop_model is None:
            clop_model, _ = load_clop(clop_path, device)
        out["gen_results"] = evaluate_generation(
            dit_model, clop_model, cache_dir,
            num_samples=num_samples, num_steps=num_steps, cfg_scale=cfg_scale, device=device,
        )
        if out["gen_results"]:
            out["generation"] = out["gen_results"]["metrics"]

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        import json
        to_save = {"clop": out["clop"], "generation": out["generation"]}
        with open(Path(output_dir) / "embedding_metrics.json", "w") as f:
            json.dump(to_save, f, indent=2)

    return out
