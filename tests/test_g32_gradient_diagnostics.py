import numpy as np
import pytest
import torch
from types import SimpleNamespace

from scripts import diagnose_dp_gradient_snr as diagnostics
from scripts.diagnose_dp_gradient_snr import (fixed_train_indices,
                                               parameter_groups,
                                               summarize_repeats,
                                               vector_diagnostics)
from src.models.forecasting import build_model
from src.privacy.per_sample import per_sample_clipped_gradients


def test_fixed_train_indices_are_unique_stratified_and_reproducible():
    first = fixed_train_indices(6084, batch_size=8, repeats=3)
    second = fixed_train_indices(6084, batch_size=8, repeats=3)
    assert first.shape == (3, 8)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == first.size
    assert first[0, 0] < first[-1, -1]


def test_parameter_groups_partition_all_gated_model_parameters():
    model = build_model("stfednet_gated", hidden_dim=8, dropout=0.0)
    groups = parameter_groups(model)
    assert set(groups) == {
        "gat_spatial_attention", "spatial_gate", "spatial_output_projection", "gru",
        "temporal_attention", "prediction_head",
    }
    flattened = [name for names in groups.values() for name in names]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {name for name, p in model.named_parameters() if p.requires_grad}


def test_snr_uses_aggregated_gradient_vector_not_per_window_norms():
    # The diagnostic API receives the post-clipping, batch-mean vector only;
    # no per-window norms are accepted as an SNR numerator.
    signal = {"a": torch.tensor([3.0, 4.0])}
    noise = {"a": torch.tensor([0.0, 2.0])}
    observed = vector_diagnostics(signal, noise)
    assert observed["signal_l2_norm"] == pytest.approx(5.0)
    assert observed["noise_l2_norm"] == pytest.approx(2.0)
    assert observed["snr"] == pytest.approx(2.5)
    assert observed["direction_cosine"] == pytest.approx(33 / (5 * np.sqrt(45)))


def test_finite_repeat_summary_reports_mean_and_sample_dispersion():
    summary = summarize_repeats([
        {"signal_l2_norm": 1.0, "noise_l2_norm": 2.0, "snr": 0.5,
         "direction_cosine": 0.8, "noise_std_mean_expected": 0.1,
         "noise_std_mean_observed": 0.09},
        {"signal_l2_norm": 3.0, "noise_l2_norm": 4.0, "snr": 0.75,
         "direction_cosine": 0.6, "noise_std_mean_expected": 0.1,
         "noise_std_mean_observed": 0.11},
    ])
    assert summary["signal_l2_norm"] == {"n": 2, "mean": 2.0, "std": pytest.approx(np.sqrt(2.0))}


def test_train_only_preparation_never_slices_post_train_rows(monkeypatch):
    class GuardedRows(np.ndarray):
        def __new__(cls, values, train_end):
            obj = np.asarray(values, dtype=np.float32).view(cls)
            obj.train_end = train_end
            return obj

        def __array_finalize__(self, source):
            self.train_end = getattr(source, "train_end", None)

        def __getitem__(self, key):
            if isinstance(key, slice):
                stop = key.stop
                if stop is None or stop > self.train_end:
                    raise AssertionError("non-TRAIN load rows were accessed")
            return super().__getitem__(key)

    timestamps = np.arange(100).astype("datetime64[h]")
    train_end = 70
    grid = SimpleNamespace(
        timestamps=timestamps,
        load_matrix=GuardedRows(np.random.default_rng(1).normal(size=(100, 2)), train_end),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        load_mask=np.array([True, True]),
        node_ids=["a", "b"],
        client_id="unused",
    )
    monkeypatch.setattr(diagnostics, "discover_grids", lambda root: {cid: cid for cid in diagnostics.CLIENT_IDS})
    monkeypatch.setattr(diagnostics, "load_grid", lambda path, client_id: grid)
    prepared = diagnostics._prepare_train_only_clients({
        "data_root": "unused", "train_ratio": 0.7, "input_hours": 4,
        "horizon_hours": 2, "expected_train_records": 65,
    })
    assert tuple(prepared) == diagnostics.CLIENT_IDS
    assert all(len(client.train) == 65 for client in prepared.values())
    assert all(not hasattr(client, "val") and not hasattr(client, "test")
               for client in prepared.values())


@pytest.mark.parametrize("module_name", ["stfednet_gated"])
def test_real_clipped_per_window_path_keeps_parameters_unchanged(module_name):
    model = build_model(module_name, hidden_dim=8, dropout=0.0).train()
    x = torch.randn(3, 4, 3, 1)
    y = torch.randn(3, 1, 3)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    mask = torch.tensor([True, True, True])
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = per_sample_clipped_gradients(model, x, y, edges, mask,
                                          max_grad_norm=0.01,
                                          noise_multiplier=0.0,
                                          normalizer=3)
    assert len(result["pre_clip_norms"]) == 3
    assert result["record_count"] == 3
    assert all(torch.isfinite(value).all() for value in result["gradients"].values())
    assert any(norm > 0.01 for norm in result["pre_clip_norms"])
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
