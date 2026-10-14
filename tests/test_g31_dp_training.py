"""G3.1 sampling, per-window clipping, accounting and resume tests."""

from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn

from src.data.loader import GridData, prepare_client
from src.fl.common import CLIENT_IDS
from src.fl import dp_server
from src.fl.dp_server import run_dp_federated
from src.models.forecasting import build_model
from src.privacy.accountant import ClientRDPAccountant, require_privacy_claim_backend
from src.privacy.per_sample import (per_sample_clipped_gradients,
                                    torch_func_clipped_gradients)
from src.privacy.poisson import nominal_steps_per_epoch, poisson_indices
from src.privacy.training import client_dp_update
from src.training.engine import load_checkpoint, seed_everything


class ForbiddenTest:
    def __len__(self):
        raise AssertionError("G3.1 accessed the test split")

    def __getitem__(self, index):
        raise AssertionError("G3.1 accessed the test split")


def synthetic_clients():
    timestamps = np.arange(np.datetime64("2021-01-01T00", "h"),
                           np.datetime64("2021-01-31T00", "h"), np.timedelta64(1, "h"))
    clients = OrderedDict()
    for index, client_id in enumerate(CLIENT_IDS):
        nodes = 4 + index
        hours = np.arange(timestamps.size, dtype=np.float32)
        values = np.stack([2 + index + node + np.sin(hours / (5 + node))
                           for node in range(nodes)], axis=1).astype(np.float32)
        source = np.arange(nodes - 1, dtype=np.int64)
        edge = np.stack((np.concatenate((source, source + 1)),
                         np.concatenate((source + 1, source))))
        mask = np.ones(nodes, dtype=bool)
        if nodes > 4:
            mask[-1] = False
        grid = GridData(client_id, client_id, [str(n) for n in range(nodes)], edge,
                        values, timestamps, mask)
        client = prepare_client(grid, 48, 1, 0.70, 0.15)
        client.test = ForbiddenTest()
        clients[client_id] = client
    return clients


def edges(nodes, device):
    source = torch.arange(nodes - 1, device=device)
    return torch.stack((torch.cat((source, source + 1)),
                        torch.cat((source + 1, source))))


@pytest.mark.parametrize("model_name", ["gru_attention", "stfednet_gated"])
def test_clipped_cuda_torch_func_matches_loop(model_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the selected G3.1 path")
    torch.manual_seed(901)
    device = torch.device("cuda")
    model = build_model(model_name, hidden_dim=8, dropout=0.0).to(device).train()
    x = torch.randn(3, 48, 5, 1, device=device)
    y = torch.randn(3, 1, 5, device=device)
    mask = torch.tensor([True, True, True, True, False], device=device)
    edge = edges(5, device)
    clip = 0.001  # force record-level clipping for these nontrivial gradients
    loop = per_sample_clipped_gradients(model, x, y, edge, mask, clip)
    func = torch_func_clipped_gradients(model, x, y, edge, mask, clip)
    assert max(loop["pre_clip_norms"]) > clip
    assert len({round(value, 8) for value in loop["pre_clip_norms"]}) > 1
    assert max(loop["clipped_norms"]) <= clip + 1e-7
    assert max(func["clipped_norms"]) <= clip + 1e-7
    assert any(value < norm - 1e-10 for value, norm in zip(loop["clipped_norms"],
                                                          loop["pre_clip_norms"]))
    for name in loop["gradients"]:
        torch.testing.assert_close(loop["gradients"][name], func["gradients"][name],
                                   atol=1e-7, rtol=8e-4)
    assert func["losses"] == pytest.approx(loop["losses"], abs=8e-5)


@pytest.mark.parametrize("model_name", ["gru_attention", "stfednet_gated"])
def test_dropout_executes_for_both_per_sample_routes(model_name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the selected G3.1 path")
    device = torch.device("cuda")
    x = torch.randn(2, 48, 4, 1, device=device)
    y = torch.randn(2, 1, 4, device=device)
    mask = torch.ones(4, dtype=torch.bool, device=device)
    edge = edges(4, device)
    for route in (per_sample_clipped_gradients, torch_func_clipped_gradients):
        model = build_model(model_name, hidden_dim=8, dropout=0.1).to(device).train()
        result = route(model, x, y, edge, mask, 0.01)
        assert all(torch.isfinite(value).all() for value in result["gradients"].values())
        assert max(result["clipped_norms"]) <= 0.01 + 1e-7


def test_poisson_sampler_is_independent_bernoulli_and_epoch_steps_are_nominal():
    generator = torch.Generator(device="cpu").manual_seed(101)
    q, n, trials = 0.2, 8, 5000
    counts = torch.zeros(n)
    both_first_two = 0
    sizes = []
    for _ in range(trials):
        picked = poisson_indices(n, q, generator)
        assert picked.unique().numel() == picked.numel()
        counts[picked] += 1
        sizes.append(picked.numel())
        both_first_two += int(0 in picked and 1 in picked)
    rates = counts / trials
    assert torch.all((rates - q).abs() < 0.02)
    assert abs(float(np.mean(sizes)) - n * q) < 0.15
    assert abs(both_first_two / trials - q * q) < 0.015
    assert nominal_steps_per_epoch(6084, 32) == 191


def test_rdp_accumulator_restores_actual_steps_and_respects_sigma_order():
    low_noise = ClientRDPAccountant(0.5, 32 / 6084, 1.0, 1e-5)
    high_noise = ClientRDPAccountant(2.0, 32 / 6084, 1.0, 1e-5)
    for _ in range(10):
        low_noise.step()
        high_noise.step()
    assert low_noise.steps == high_noise.steps == 10
    assert high_noise.epsilon() < low_noise.epsilon()
    previous = high_noise.epsilon()
    high_noise.step()
    assert high_noise.epsilon() >= previous
    restored = ClientRDPAccountant(2.0, 32 / 6084, 1.0, 1e-5,
                                   high_noise.state_dict())
    assert restored.steps == high_noise.steps
    assert restored.epsilon() == pytest.approx(high_noise.epsilon())
    sigma_zero = ClientRDPAccountant(0.0, 32 / 6084, 1.0, 1e-5)
    sigma_zero.step()
    assert sigma_zero.epsilon() is None
    assert sigma_zero.state_dict()["status"] == "non_dp_sigma_zero_epsilon_undefined"


def test_privacy_claim_mode_refuses_unreviewed_rng():
    with pytest.raises(RuntimeError, match="no reviewed"):
        require_privacy_claim_backend("privacy_claim")
    require_privacy_claim_backend("research")


def test_empty_poisson_batch_still_noise_updates_adamw():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, x, edge_index=None):
            return (x[:, -1, :, 0] * self.weight).unsqueeze(1)

    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    before = model.weight.detach().clone()
    result = per_sample_clipped_gradients(
        model, torch.empty(0, 1, 1, 1), torch.empty(0, 1, 1),
        torch.empty(2, 0, dtype=torch.long), torch.ones(1, dtype=torch.bool),
        max_grad_norm=1.0, noise_multiplier=1.0, normalizer=4.0,
        generator=torch.Generator().manual_seed(90))
    optimizer.zero_grad(set_to_none=True)
    model.weight.grad = result["gradients"]["weight"]
    optimizer.step()
    assert torch.isfinite(model.weight)
    assert model.weight.item() != before.item()
    assert result["record_count"] == 0
    assert result["noise_std_on_mean"] == pytest.approx(0.25)


@pytest.mark.parametrize("selected_count", [0, 1, 4])
def test_client_update_accounts_empty_and_uses_fixed_expected_normalizer(
        selected_count, monkeypatch):
    clients = synthetic_clients()
    prepared = clients[CLIENT_IDS[0]]
    cfg = _config("unused", rounds=1)
    cfg["expected_batch_size"] = 2
    cfg["batch_size"] = 2
    cfg["max_dp_steps_per_epoch"] = 1
    n = len(prepared.train)
    q = 2 / n
    account = ClientRDPAccountant(1.0, q, 1.0, 1e-5)
    sampler = torch.Generator().manual_seed(300)
    noise = torch.Generator().manual_seed(301)
    captured = {}
    monkeypatch.setattr("src.privacy.training.poisson_indices",
                        lambda size, sample_rate, generator:
                        torch.arange(selected_count, dtype=torch.long))
    original = per_sample_clipped_gradients

    def checked(*args, **kwargs):
        captured["normalizer"] = kwargs["normalizer"]
        return original(*args, **kwargs)

    monkeypatch.setattr("src.privacy.training.per_sample_clipped_gradients", checked)
    initial = build_model("gru_attention", hidden_dim=8, dropout=0.0).state_dict()
    update = client_dp_update(initial, prepared, cfg, "gru_attention", torch.device("cpu"),
                              account, sampler, noise, gradient_method="loop",
                              max_steps_per_epoch=1)
    assert account.steps == 1
    assert captured["normalizer"] == pytest.approx(q * n)
    assert update["diagnostics"]["selected_record_draws"] == selected_count
    if selected_count == 0:
        assert update["diagnostics"]["empty_steps"] == 1
        assert update["diagnostics"]["method_counts"] == {"empty_noise_only": 1}
    else:
        assert update["diagnostics"]["mean_actual_batch_size"] == selected_count


def test_privacy_claim_rejects_before_loading_client_data(monkeypatch):
    monkeypatch.setattr(dp_server, "prepare_clients",
                        lambda config: (_ for _ in ()).throw(AssertionError("data was loaded")))
    cfg = _config("unused", rounds=1)
    cfg["privacy_mode"] = "privacy_claim"
    with pytest.raises(RuntimeError, match="no reviewed"):
        run_dp_federated(cfg, "gru_attention")


def _config(root, rounds=2):
    return {
        "project_name": "test", "data_root": "unused", "output_root": str(root),
        "input_hours": 48, "horizon_hours": 1, "train_ratio": 0.70, "val_ratio": 0.15,
        "hidden_dim": 8, "dropout": 0.0, "gat_heads": 2, "batch_size": 2,
        "learning_rate": 0.001, "weight_decay": 0.0001, "rounds": rounds,
        "local_epochs": 1, "client_participation": 5, "mu": 0.0, "seed": 42,
        "device": "cpu", "num_workers": 0, "cpu_threads": 1,
        "mape_threshold_kw": 0.1, "huber_delta": 1.0, "gradient_clip": 1.0,
        "max_train_batches": None, "max_val_batches": 1, "clients": list(CLIENT_IDS),
        "expected_batch_size": 2, "noise_multiplier": 1.0, "max_grad_norm": 1.0,
        "expected_train_records": None,
        "delta": 1e-5, "gradient_method": "loop", "privacy_mode": "research",
        "max_dp_steps_per_epoch": 1,
    }


def _same_state(first, second):
    return set(first) == set(second) and all(torch.equal(first[key], second[key])
                                             for key in first)


def test_five_client_dp_fedavg_checkpoint_and_resume_are_exact(tmp_path):
    clients = synthetic_clients()
    continuous_cfg = _config(tmp_path / "continuous")
    seed_everything(42, 1)
    continuous = run_dp_federated(continuous_cfg, "gru_attention",
                                  run_dir=tmp_path / "continuous" / "run",
                                  prepared_clients=clients)
    full = load_checkpoint(continuous / "latest_dp_global.pt")
    assert full["completed_round"] == 2
    assert full["privacy_claim_allowed"] is False
    assert all(full["accountant_states"][cid]["steps"] == 2 for cid in CLIENT_IDS)
    assert full["test_evaluated"] is False
    assert all(row["smoke"] for row in full["round_history"])
    public_accounting = __import__("json").loads((continuous / "privacy_accounting.json").read_text())
    assert public_accounting["privacy_claim_allowed"] is False
    assert all(item["committed_update_steps"] == 2
               for item in public_accounting["clients"].values())
    assert "noise_generators" not in public_accounting
    assert "seed" not in public_accounting

    first_cfg = _config(tmp_path / "resumed", rounds=1)
    seed_everything(42, 1)
    first_run = run_dp_federated(first_cfg, "gru_attention",
                                 run_dir=tmp_path / "resumed" / "run",
                                 prepared_clients=clients)
    next_cfg = _config(tmp_path / "resumed", rounds=2)
    resumed_dir = run_dp_federated(next_cfg, "gru_attention",
                                   resume=first_run / "latest_dp_global.pt",
                                   prepared_clients=clients)
    resumed = load_checkpoint(resumed_dir / "latest_dp_global.pt")
    assert _same_state(full["global_model_state_dict"], resumed["global_model_state_dict"])
    assert full["accountant_states"] == resumed["accountant_states"]
    assert [row["client_equal_wape_percent"] for row in full["round_history"]] == [
        row["client_equal_wape_percent"] for row in resumed["round_history"]]
    assert resumed["accountant_states"][CLIENT_IDS[0]]["steps"] == 2


def test_mid_client_checkpoint_resume_does_not_repeat_committed_step(tmp_path, monkeypatch):
    clients = synthetic_clients()
    cfg = _config(tmp_path / "continuous", rounds=1)
    cfg["max_dp_steps_per_epoch"] = 2
    cfg["dropout"] = 0.1
    seed_everything(42, 1)
    continuous_dir = run_dp_federated(cfg, "gru_attention",
                                      run_dir=tmp_path / "continuous" / "run",
                                      prepared_clients=clients)
    continuous = load_checkpoint(continuous_dir / "latest_dp_global.pt")

    interrupted_cfg = _config(tmp_path / "interrupted", rounds=1)
    interrupted_cfg["max_dp_steps_per_epoch"] = 2
    interrupted_cfg["dropout"] = 0.1
    original = dp_server.client_dp_update
    triggered = {"done": False}

    def interrupt_after_committed_step(*args, **kwargs):
        call_args = list(args)
        callback = call_args[-1]

        def commit_then_interrupt(progress, account_state):
            callback(progress, account_state)
            if not triggered["done"]:
                triggered["done"] = True
                raise RuntimeError("injected process stop after atomic DP step checkpoint")

        call_args[-1] = commit_then_interrupt
        return original(*call_args, **kwargs)

    monkeypatch.setattr(dp_server, "client_dp_update", interrupt_after_committed_step)
    seed_everything(42, 1)
    with pytest.raises(RuntimeError, match="injected process stop"):
        run_dp_federated(interrupted_cfg, "gru_attention",
                         run_dir=tmp_path / "interrupted" / "run",
                         prepared_clients=clients)
    interrupted_latest = load_checkpoint(tmp_path / "interrupted" / "run" / "latest_dp_global.pt")
    assert interrupted_latest["accountant_states"][CLIENT_IDS[0]]["steps"] == 1
    assert interrupted_latest["accountant_states"][CLIENT_IDS[1]]["steps"] == 0

    monkeypatch.setattr(dp_server, "client_dp_update", original)
    resumed_dir = run_dp_federated(interrupted_cfg, "gru_attention",
                                   resume=tmp_path / "interrupted" / "run" / "latest_dp_global.pt",
                                   prepared_clients=clients)
    resumed = load_checkpoint(resumed_dir / "latest_dp_global.pt")
    assert _same_state(continuous["global_model_state_dict"], resumed["global_model_state_dict"])
    assert continuous["accountant_states"] == resumed["accountant_states"]
    assert all(resumed["accountant_states"][cid]["steps"] == 2 for cid in CLIENT_IDS)
