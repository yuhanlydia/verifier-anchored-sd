import torch

from verifier_anchored_sd.training.block_acceptance_loss import block_acceptance_loss


def test_block_loss_weights_later_positions_by_prefix_acceptance():
    p = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    q = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
    loss, metrics = block_acceptance_loss(p, q)
    assert torch.allclose(metrics["alpha"], torch.tensor([[0.5, 1.0]]))
    assert torch.allclose(metrics["expected_length"], torch.tensor([1.0]))
    assert loss.item() == -0.5
