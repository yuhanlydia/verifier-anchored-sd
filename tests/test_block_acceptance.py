import torch

from verifier_anchored_sd.training.block_acceptance_loss import (
    block_acceptance_loss,
    one_step_acceptance_loss,
)


def test_block_loss_weights_later_positions_by_prefix_acceptance():
    p = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    q = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
    loss, metrics = block_acceptance_loss(p, q)
    assert torch.allclose(metrics["alpha"], torch.tensor([[0.5, 1.0]]))
    assert torch.allclose(metrics["expected_length"], torch.tensor([1.0]))
    assert loss.item() == -0.5


def test_one_step_baseline_ignores_later_positions():
    p = torch.tensor([[[0.5, 0.5], [1.0, 0.0]]])
    q_a = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    q_b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    loss_a, metrics_a = one_step_acceptance_loss(p, q_a)
    loss_b, metrics_b = one_step_acceptance_loss(p, q_b)
    assert torch.allclose(loss_a, loss_b)
    assert torch.allclose(metrics_a["first_acceptance"], torch.tensor([0.5]))
    assert torch.allclose(metrics_b["first_acceptance"], torch.tensor([0.5]))
