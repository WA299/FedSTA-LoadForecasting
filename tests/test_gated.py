"""G1.2 gate initialization, residual preservation and parameter sharing."""

import pytest
import torch

from src.models.forecasting import build_model
from src.training.engine import masked_huber
from src.training.gate_statistics import GateStatistics


def chain(nodes):
    edges = torch.stack([torch.arange(nodes - 1), torch.arange(1, nodes)])
    return torch.cat([edges, edges.flip(0)], dim=1)


@pytest.mark.parametrize("nodes", [39, 50, 56, 76, 80])
def test_gated_initialization_residual_and_updates(nodes):
    torch.manual_seed(42)
    model = build_model("stfednet_gated", dropout=0)
    x, edges = torch.randn(2, 48, nodes, 1), chain(nodes)
    fused, gate, spatial = model.spatial(x, edges, return_gate=True)
    assert fused.shape == gate.shape == spatial.shape == x.shape
    assert torch.allclose(gate, torch.full_like(gate, torch.sigmoid(torch.tensor(-3.)).item()))
    assert torch.equal(model.spatial.gate.weight, torch.zeros(1, 2))
    assert model.spatial.gate.bias.item() == -3.0
    assert torch.all((gate >= 0) & (gate <= 1))
    assert torch.allclose(fused, x + gate * spatial)
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    prediction = model(x, edges)
    assert prediction.shape == (2, 1, nodes)
    loss = masked_huber(prediction, torch.randn(2, 1, nodes), torch.ones(nodes, dtype=torch.bool))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.spatial.gate.weight.grad.abs().sum() > 0
    assert model.spatial.gate.bias.grad.abs().sum() > 0
    assert model.spatial.gat.projection.weight.grad.abs().sum() > 0
    torch.optim.AdamW(model.parameters(), lr=0.001).step()
    for key in ("spatial.gate.weight", "spatial.gate.bias", "spatial.gat.projection.weight"):
        assert not torch.equal(model.state_dict()[key], before[key])
    # Force the spatial correction to zero: original signal is kept exactly.
    with torch.no_grad():
        model.spatial.to_load_channel.weight.zero_()
        model.spatial.to_load_channel.bias.zero_()
    assert torch.equal(model.spatial(x, edges), x)


def test_gate_is_node_time_sample_adaptive_and_node_independent():
    model = build_model("stfednet_gated", dropout=0)
    with torch.no_grad():
        model.spatial.gate.weight.copy_(torch.tensor([[1.0, 0.0]]))
    x = torch.arange(2*48*39, dtype=torch.float32).reshape(2, 48, 39, 1) / 1000
    _, gates, _ = model.spatial(x, chain(39), return_gate=True)
    assert torch.allclose(gates, torch.sigmoid(x - 3.0))
    expected = {key: value.shape for key, value in model.state_dict().items()}
    for nodes in (39, 50, 56, 76, 80):
        target = build_model("stfednet_gated").eval()
        assert {key: value.shape for key, value in target.state_dict().items()} == expected
        target.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            assert target(torch.zeros(1, 48, nodes, 1), chain(nodes)).shape == (1, 1, nodes)


def test_gate_statistics_weight_unequal_batches_and_population_std():
    a = torch.tensor([0.1, 0.9], dtype=torch.float64).reshape(2, 1, 1, 1)
    b = torch.tensor([0.3], dtype=torch.float64).reshape(1, 1, 1, 1)
    stats = GateStatistics()
    stats.update(a)
    stats.update(b)
    result = stats.compute(3)
    values = torch.cat([a.flatten(), b.flatten()])
    assert result["gate_mean"] == pytest.approx(values.mean().item())
    assert result["gate_std"] == pytest.approx(values.std(correction=0).item())
    assert result["gate_count"] == 3 and result["complete_validation"]
    assert not stats.compute(10)["complete_validation"]
    constant = GateStatistics()
    constant.update(torch.full((3, 48, 39, 1), 0.04742587))
    assert constant.compute(3)["gate_std"] == 0.0
