# reproducibility.py — Deterministic training utilities
"""
Utilities for reproducible training via deterministic algorithms and
seed management. Import and call `seed_everything(seed)` before
training to get bitwise-reproducible results (within the constraints
of the hardware).

Usage:
    from src.training.reproducibility import seed_everything
    seed_everything(42)
"""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds and optionally enable deterministic algorithms.

    Parameters
    ----------
    seed : int
        Random seed.
    deterministic : bool
        If True, enable ``torch.use_deterministic_algorithms(True)``
        and set the CUBLAS workspace configuration for reproducibility.
        This may reduce performance slightly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
            logger.info(
                "Deterministic mode enabled (seed=%d, CUBLAS workspace=:4096:8)",
                seed,
            )
        except RuntimeError:
            # Some operations don't have deterministic implementations
            torch.use_deterministic_algorithms(True, warn_only=True)
            logger.warning(
                "Deterministic mode enabled with warn_only=True (seed=%d); "
                "some operations may still be non-deterministic.",
                seed,
            )
    else:
        torch.backends.cudnn.benchmark = True
        logger.info("Non-deterministic mode (seed=%d, cudnn.benchmark=True)", seed)
