import numpy as np
import pytest
import torch

from src.models.forecasting import (MODEL_NAMES, AdditiveTemporalAttention,
                                    adjacency, build_model)
from src.training.engine import masked_huber
from src.training.metrics import MetricAccumulator, aggregate_results


@pytest.fixture(scope="module", autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("nodes", [39, 50, 56, 76, 80])
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_variable_graph_forward_backward(name, nodes):
    edges = torch.stack([torch.arange(nodes - 1), torch.arange(1, nodes)])
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    x = torch.randn(2, 48, nodes, 1)
    target = torch.randn(2, 1, nodes)
    mask = torch.arange(nodes) % 3 != 0
    model = build_model(name)
    prediction = model(x, edges)
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    loss = masked_huber(prediction, target, mask)
    assert torch.isfinite(loss)
    if list(model.parameters()):
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_naive_correct_time_offsets():
    x = torch.arange(48.0).reshape(1, 48, 1, 1)
    assert build_model("persistence")(x).item() == 47
    # Target index 48: the observation 24 hours earlier is index 24.
    assert build_model("seasonal_naive")(x).item() == 24


def test_attention_normalises_time_and_graph_is_masked():
    attention = AdditiveTemporalAttention()
    context, weights = attention(torch.randn(2, 5, 48, 32), return_weights=True)
    assert context.shape == (2, 5, 32)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 5))
    graph = adjacency(torch.tensor([[0], [1]]), 3, "cpu")
    assert graph.diag().all() and graph[1, 0] and not graph[0, 1]
    model = build_model("stfednet", dropout=0).eval()
    x = torch.randn(1, 48, 3, 1)
    edges = torch.tensor([[0, 1], [1, 0]])
    changed = x.clone()
    changed[:, :, 2] += 100
    assert torch.allclose(model(x, edges)[:, :, :2], model(changed, edges)[:, :, :2])


def test_metrics_mask_threshold_and_equal_client_aggregation():
    first = MetricAccumulator()
    first.update(np.array([[[0.05, 2, 1000]]]), np.array([[[0.15, 1, 0]]]), [True, True, False])
    metrics = first.compute()
    assert metrics["mae_kw"] == pytest.approx(0.55)
    assert metrics["rmse_kw"] == pytest.approx(np.sqrt(1.01 / 2))
    assert metrics["wape_percent"] == pytest.approx(100 * 1.1 / 2.05)
    assert metrics["mape_percent"] == pytest.approx(50)
    assert metrics["mape_valid_fraction"] == 0.5
    second = MetricAccumulator()
    second.update(np.array([[[100.0]]]), np.array([[[99.0]]]), [True])
    common = {"model_name": "gru", "seed": 42, "input_hours": 48, "horizon_hours": 1,
              "mape_threshold_kw": 0.1, "smoke": False}
    results = [dict(common, client_id="a", metrics=metrics, statistics=first.statistics()),
               dict(common, client_id="b", metrics=second.compute(), statistics=second.statistics())]
    summary = aggregate_results(results)
    assert summary["client_equal_wape_percent"] == pytest.approx((metrics["wape_percent"] + 1) / 2)
    assert summary["overall"]["wape_percent"] == pytest.approx(100 * 2.1 / 102.05)
    zero = MetricAccumulator()
    zero.update(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), [True])
    assert zero.compute()["wape_percent"] is None
    assert zero.compute()["mape_percent"] is None
