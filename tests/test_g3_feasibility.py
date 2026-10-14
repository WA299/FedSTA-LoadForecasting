"""Numerical checks for the isolated G3.0 per-window-gradient reference."""

import pytest
import torch
from torch import nn

from src.models.forecasting import build_model
from src.privacy.per_sample import (masked_huber_per_sample,
                                    per_sample_clipped_gradients,
                                    torch_func_clipped_gradients)
from src.training.engine import masked_huber


class ScalarForecaster(nn.Module):
    def __init__(self, initial=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(initial)))

    def forward(self, x, edge_index=None):
        return (x[:, -1, :, 0] * self.weight).unsqueeze(1)


def _edges(nodes, device="cpu"):
    source = torch.arange(nodes - 1, device=device)
    target = source + 1
    return torch.stack((torch.cat((source, target)), torch.cat((target, source))))


@pytest.mark.parametrize("model_name", ["gru_attention", "stfednet_gated"])
def test_existing_models_forward_backward_and_per_window_loss(model_name):
    model = build_model(model_name, hidden_dim=8, dropout=0.0).train()
    x = torch.randn(2, 48, 4, 1)
    y = torch.randn(2, 1, 4)
    mask = torch.tensor([True, True, False, True])
    prediction = model(x, _edges(4))
    assert prediction.shape == (2, 1, 4)
    batch_loss = masked_huber(prediction, y, mask)
    batch_loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)
    per_record = masked_huber_per_sample(prediction.detach(), y, mask)
    assert per_record.shape == (2,)
    assert per_record.mean().item() == pytest.approx(batch_loss.item(), abs=1e-7)


@pytest.mark.parametrize("model_name", ["gru_attention", "stfednet_gated"])
def test_mean_per_window_gradient_matches_batch_gradient(model_name):
    torch.manual_seed(55)
    model = build_model(model_name, hidden_dim=8, dropout=0.0).train()
    x = torch.randn(3, 48, 3, 1)
    y = torch.randn(3, 1, 3)
    mask = torch.tensor([True, True, False])
    edge = _edges(3)
    batch_loss = masked_huber(model(x, edge), y, mask)
    batch_grads = torch.autograd.grad(batch_loss, tuple(model.parameters()))
    per_sample = per_sample_clipped_gradients(model, x, y, edge, mask,
                                              max_grad_norm=1e6)
    assert len(per_sample["losses"]) == 3
    for parameter, batch_gradient in zip(model.parameters(), batch_grads):
        assert torch.allclose(per_sample["gradients"][next(
            name for name, value in model.named_parameters() if value is parameter)],
            batch_gradient, atol=2e-5, rtol=2e-4)


def test_per_record_clipping_precedes_aggregation_and_bounds_norm():
    model = ScalarForecaster()
    x = torch.tensor([[[[2.0]]], [[[10.0]]], [[[-3.0]]]])
    y = torch.zeros(3, 1, 1)
    mask = torch.tensor([True])
    clip = 0.5
    result = per_sample_clipped_gradients(model, x, y, torch.empty((2, 0), dtype=torch.long),
                                          mask, max_grad_norm=clip)
    expected = []
    for index in range(x.shape[0]):
        loss = masked_huber_per_sample(model(x[index:index + 1]), y[index:index + 1], mask)[0]
        gradient, = torch.autograd.grad(loss, (model.weight,))
        norm = gradient.norm()
        expected.append(gradient * min(1.0, clip / (norm.item() + 1e-12)))
    assert max(result["clipped_norms"]) <= clip + 1e-7
    expected_mean = torch.stack(expected).mean()
    assert result["gradients"]["weight"].item() == pytest.approx(expected_mean.item(), abs=1e-7)


def test_window_gradients_do_not_mix_different_records():
    model = ScalarForecaster(initial=0.7)
    x = torch.tensor([[[[1.0]]], [[[4.0]]], [[[-2.0]]]])
    y = torch.tensor([[[0.0]], [[1.0]], [[-3.0]]])
    edge = torch.empty((2, 0), dtype=torch.long)
    mask = torch.tensor([True])
    combined = per_sample_clipped_gradients(model, x, y, edge, mask, 0.8)
    individual = [per_sample_clipped_gradients(model, x[i:i + 1], y[i:i + 1],
                                               edge, mask, 0.8)
                  for i in range(x.shape[0])]
    assert combined["pre_clip_norms"] == pytest.approx(
        [row["pre_clip_norms"][0] for row in individual])
    expected = torch.stack([row["gradients"]["weight"] for row in individual]).mean()
    assert combined["gradients"]["weight"].item() == pytest.approx(expected.item())


def test_partial_batch_uses_actual_divisor_and_poisson_empty_batch_keeps_shape():
    model = ScalarForecaster()
    x = torch.ones(1, 2, 2, 1)
    y = torch.zeros(1, 1, 2)
    mask = torch.ones(2, dtype=torch.bool)
    edge = torch.empty((2, 0), dtype=torch.long)
    actual = per_sample_clipped_gradients(model, x, y, edge, mask, 1e6)
    expected_normalizer = per_sample_clipped_gradients(model, x, y, edge, mask,
                                                       1e6, normalizer=4)
    assert actual["normalizer"] == 1
    assert expected_normalizer["normalizer"] == 4
    assert expected_normalizer["gradients"]["weight"].item() == pytest.approx(
        actual["gradients"]["weight"].item() / 4)
    empty = per_sample_clipped_gradients(model, x[:0], y[:0], edge, mask,
                                         max_grad_norm=0.7, noise_multiplier=1.3,
                                         normalizer=4, generator=torch.Generator().manual_seed(4))
    assert empty["record_count"] == 0
    assert empty["noise_std_on_sum"] == pytest.approx(0.91)
    assert empty["noise_std_on_mean"] == pytest.approx(0.91 / 4)
    assert empty["gradients"]["weight"].shape == model.weight.shape


def test_gaussian_noise_is_added_to_sum_before_normalization():
    model = ScalarForecaster()
    x = torch.tensor([[[[2.0]]], [[[3.0]]]])
    y = torch.zeros(2, 1, 1)
    edge = torch.empty((2, 0), dtype=torch.long)
    mask = torch.tensor([True])
    clip, sigma = 0.4, 1.5
    clean = per_sample_clipped_gradients(model, x, y, edge, mask, clip)
    noisy = per_sample_clipped_gradients(
        model, x, y, edge, mask, clip, noise_multiplier=sigma,
        generator=torch.Generator().manual_seed(101))
    expected_noise = torch.randn(model.weight.shape, generator=torch.Generator().manual_seed(101))
    difference = noisy["gradients"]["weight"] - clean["gradients"]["weight"]
    assert torch.allclose(difference, expected_noise * (sigma * clip / 2))
    assert noisy["noise_std_on_mean"] == pytest.approx(sigma * clip / 2)


@pytest.mark.parametrize("model_name", ["gru_attention", "stfednet_gated"])
def test_both_models_support_the_five_client_node_sizes(model_name):
    model = build_model(model_name, hidden_dim=8, dropout=0.0).train()
    for nodes in (39, 50, 56, 80, 76):
        x = torch.randn(1, 3, nodes, 1)
        y = torch.randn(1, 1, nodes)
        mask = torch.ones(nodes, dtype=torch.bool)
        output = model(x, _edges(nodes))
        loss = masked_huber(output, y, mask)
        loss.backward()
        assert output.shape == (1, 1, nodes)
        assert torch.isfinite(loss)
        model.zero_grad(set_to_none=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_name", ["gru_attention", "stfednet_gated"])
def test_cuda_torch_func_clipped_gradients_match_loop(model_name):
    torch.manual_seed(123)
    device = torch.device("cuda")
    model = build_model(model_name, hidden_dim=8, dropout=0.0).to(device).train()
    x = torch.randn(2, 48, 3, 1, device=device)
    y = torch.randn(2, 1, 3, device=device)
    mask = torch.tensor([True, True, False], device=device)
    edge = _edges(3, device)
    reference = per_sample_clipped_gradients(model, x, y, edge, mask, 1e6)
    transformed = torch_func_clipped_gradients(model, x, y, edge, mask, 1e6)
    assert transformed["gradient_stack_bytes"] > 0
    for name in reference["gradients"]:
        assert torch.allclose(reference["gradients"][name],
                              transformed["gradients"][name], atol=5e-5, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_name", ["gru_attention", "stfednet_gated"])
def test_cuda_torch_func_handles_training_dropout(model_name):
    device = torch.device("cuda")
    model = build_model(model_name, hidden_dim=8, dropout=0.1).to(device).train()
    x = torch.randn(2, 48, 3, 1, device=device)
    y = torch.randn(2, 1, 3, device=device)
    mask = torch.tensor([True, True, False], device=device)
    result = torch_func_clipped_gradients(model, x, y, _edges(3, device), mask, 1.0)
    assert result["record_count"] == 2
    assert all(torch.isfinite(gradient).all() for gradient in result["gradients"].values())
