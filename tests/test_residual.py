"""G1.1 residual path, update and cross-client parameter compatibility checks."""

import pytest
import torch

from src.models.forecasting import build_model
from src.training.engine import masked_huber


def chain(nodes):
    edges = torch.stack([torch.arange(nodes - 1), torch.arange(1, nodes)])
    return torch.cat([edges, edges.flip(0)], dim=1)


@pytest.mark.parametrize("nodes", [39, 50, 56, 76, 80])
def test_residual_shape_original_path_and_parameter_updates(nodes):
    torch.manual_seed(42)
    model = build_model("stfednet_residual", dropout=0)
    edges, x = chain(nodes), torch.randn(2, 48, nodes, 1)
    fused, spatial = model.spatial(x, edges, return_spatial=True)
    assert fused.shape == spatial.shape == x.shape
    assert model.space_scale.ndim == 0
    assert model.space_scale.item() == pytest.approx(0.05)
    assert torch.allclose(fused, x + model.space_scale * spatial)
    with torch.no_grad():
        model.space_scale.zero_()
    assert torch.equal(model.spatial(x, edges), x)
    # The raw-load path is an identity even for gradients when spatial is disabled.
    probe = x.clone().requires_grad_()
    model.spatial(probe, edges).sum().backward()
    assert torch.equal(probe.grad, torch.ones_like(probe))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.space_scale.fill_(0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss = masked_huber(model(x, edges), torch.randn(2, 1, nodes), torch.ones(nodes, dtype=torch.bool))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.space_scale.grad.abs() > 0
    gat_parameters = list(model.spatial.gat.parameters())
    assert any(parameter.grad.abs().sum() > 0 for parameter in gat_parameters)
    optimizer.step()
    assert any(not torch.equal(parameter.detach(), before[name])
               for name, parameter in model.named_parameters()
               if name.startswith("spatial.gat."))


def test_residual_parameters_identical_between_client_node_counts():
    source = build_model("stfednet_residual").eval()
    expected = {name: tuple(p.shape) for name, p in source.named_parameters()}
    for nodes in (39, 50, 56, 76, 80):
        target = build_model("stfednet_residual").eval()
        assert {name: tuple(p.shape) for name, p in target.named_parameters()} == expected
        target.load_state_dict(source.state_dict(), strict=True)
        with torch.no_grad():
            assert target(torch.zeros(1, 48, nodes, 1), chain(nodes)).shape == (1, 1, nodes)


@pytest.mark.parametrize("name,input_features", [("stfednet", 32), ("gat_gru", 32), ("gru_attention", 1)])
def test_legacy_model_structure_and_strict_checkpoint_roundtrip(name, input_features, tmp_path):
    model = build_model(name).eval()
    assert not hasattr(model, "space_scale")
    assert model.temporal.gru.input_size == input_features
    assert not any("space_scale" in key or "to_load_channel" in key for key in model.state_dict())
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    restored = build_model(name).eval()
    restored.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"], strict=True)
    x, edges = torch.randn(1, 48, 39, 1), chain(39)
    with torch.no_grad():
        assert torch.equal(model(x, edges), restored(x, edges))
