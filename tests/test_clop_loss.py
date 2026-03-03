import torch

from src.architecture.clop import PrototypeSigLIPLoss


def test_positive_bag_loss_and_backward():
    torch.manual_seed(0)

    logits = torch.randn(4, 8, requires_grad=True)

    # Two positives per row
    variant_mask = torch.tensor([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
    ], dtype=torch.bool)

    # Single-positive baseline (first positive in each row)
    single_mask = torch.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=torch.bool)

    loss_with_bag = PrototypeSigLIPLoss._bag_positive_nce_loss(logits, variant_mask)
    loss_single = PrototypeSigLIPLoss._bag_positive_nce_loss(logits, single_mask)

    # More positives should not increase the lower bound loss
    assert loss_with_bag <= loss_single + 1e-6

    loss_with_bag.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
