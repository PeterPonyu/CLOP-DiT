# schedulers.py — Learning rate schedulers
"""
Custom learning rate schedulers for CLOP-DiT training.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_steps : int
        Number of warmup steps with linear ramp.
    total_steps : int
        Total training steps.
    min_lr_ratio : float
        Minimum LR as fraction of base LR.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            factor = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )

        return [base_lr * factor for base_lr in self.base_lrs]
