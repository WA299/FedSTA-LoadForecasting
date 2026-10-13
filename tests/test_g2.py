"""Core G2 aggregation, objectives, round recovery and centralized coverage."""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from src.data.loader import GridData, prepare_client
from src.fl.aggregation import fedavg_aggregate, fedprox_objective, proximal_penalty
from src.fl.centralized import run_centralized
from scripts.compare_g2 import check_protocol, find_best
from src.fl.client import client_update
from src.fl.common import CLIENT_IDS, pooled_and_equal_metrics
from src.fl.server import run_federated
from src.models.forecasting import build_model
from src.training.engine import load_checkpoint, seed_everything
from src.training.metrics import MetricAccumulator


class ForbiddenTest:
    def __len__(self):
        raise AssertionError("G2 training accessed test windows")

    def __getitem__(self, index):
        raise AssertionError("G2 training accessed test windows")


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    torch.set_num_threads(1)


def synthetic_clients():
    timestamps = np.arange(np.datetime64("2021-01-01T00", "h"),
                           np.datetime64("2021-01-21T00", "h"), np.timedelta64(1, "h"))
    prepared = OrderedDict()
    for index, client_id in enumerate(CLIENT_IDS):
        nodes = 3 + index
        hours = np.arange(timestamps.size, dtype=np.float32)
        values = np.stack([2 + index + node + np.sin(hours / (5 + node))
                           for node in range(nodes)], axis=1).astype(np.float32)
        sources = np.arange(nodes - 1, dtype=np.int64)
        targets = sources + 1
        edge_index = np.stack([np.concatenate([sources, targets]),
                               np.concatenate([targets, sources])])
        mask = np.ones(nodes, dtype=bool)
        if nodes > 3:
            mask[-1] = False
        grid = GridData(client_id, client_id, [str(n) for n in range(nodes)], edge_index,
                        values, timestamps, mask)
        client = prepare_client(grid, input_hours=48, horizon_hours=1,
                                train_ratio=0.70, val_ratio=0.15)
        client.test = ForbiddenTest()
        prepared[client_id] = client
    return prepared


def config(tmp_path, rounds=2):
    return {"project_name": "test", "data_root": "unused", "output_root": str(tmp_path),
            "input_hours": 48, "horizon_hours": 1, "train_ratio": 0.70, "val_ratio": 0.15,
            "hidden_dim": 32, "dropout": 0.0, "gat_heads": 2, "batch_size": 2,
            "learning_rate": 0.001, "weight_decay": 0.0001, "rounds": rounds,
            "local_epochs": 1, "mu": 0.01, "seed": 42, "device": "cpu",
            "num_workers": 0, "cpu_threads": 1, "mape_threshold_kw": 0.1,
            "huber_delta": 1.0, "gradient_clip": 1.0, "max_train_batches": None,
            "max_val_batches": None, "clients": list(CLIENT_IDS)}


def test_fedavg_exact_and_sample_weighted_aggregation():
    a = {"weight": torch.tensor([1.0, 3.0]), "bias": torch.tensor(2.0)}
    b = {"weight": torch.tensor([5.0, 7.0]), "bias": torch.tensor(6.0)}
    equal = fedavg_aggregate([a, b], [1, 1])
    weighted = fedavg_aggregate([a, b], [1, 3])
    assert torch.equal(equal["weight"], torch.tensor([3.0, 5.0]))
    assert torch.equal(weighted["weight"], torch.tensor([4.0, 6.0]))
    assert weighted["bias"].item() == 5.0
    assert torch.equal(a["weight"], torch.tensor([1.0, 3.0]))
    with pytest.raises(ValueError):
        fedavg_aggregate([a, b], [1, 0])


def test_fedprox_penalty_and_gradient_and_mu_zero_equivalence():
    model = torch.nn.Linear(1, 1, bias=False)
    anchor = {"weight": torch.zeros_like(model.weight)}
    with torch.no_grad():
        model.weight.fill_(2.0)
    penalty = proximal_penalty(model, anchor)
    assert penalty.item() == 4.0
    huber = torch.tensor(3.0, requires_grad=True)
    objective = fedprox_objective(huber, model, anchor, 0.5)
    assert objective.item() == 4.0
    objective.backward()
    assert model.weight.grad.item() == pytest.approx(1.0)
    assert fedprox_objective(huber, model, anchor, 0.0) is huber


def test_zero_mu_local_update_matches_fedavg_exactly(tmp_path):
    prepared = synthetic_clients()["lv_39"]
    seed_everything(17, 1)
    state = build_model("gru_attention", dropout=0.0).state_dict()
    settings = config(tmp_path)
    avg = client_update(state, prepared, settings, "gru_attention", 1, 0,
                        torch.device("cpu"), algorithm="fedavg", max_train_batches=1)
    prox = client_update(state, prepared, settings, "gru_attention", 1, 0,
                         torch.device("cpu"), algorithm="fedprox", mu=0.0,
                         max_train_batches=1)
    assert avg["mean_loss"] == prox["mean_loss"]
    assert all(torch.equal(avg["state_dict"][key], prox["state_dict"][key])
               for key in avg["state_dict"])


def test_variable_node_clients_share_same_model_parameters_and_updates_are_model_only(tmp_path):
    prepared = synthetic_clients()
    seed_everything(42, 1)
    model = build_model("stfednet_gated", dropout=0.0)
    names_and_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    state = build_model("gru_attention", dropout=0.0).state_dict()
    updates = []
    for index, client in enumerate(prepared.values()):
        other = build_model("stfednet_gated", dropout=0.0)
        assert {name: tuple(value.shape) for name, value in other.state_dict().items()} == names_and_shapes
        receiver = build_model("gru_attention", dropout=0.0)
        receiver.load_state_dict(state, strict=True)
        update = client_update(state, client, config(tmp_path), "gru_attention", 1, index,
                               torch.device("cpu"), max_train_batches=1)
        assert set(update) == {"state_dict", "sample_count", "processed_windows", "mean_loss"}
        assert set(update["state_dict"]) == set(state)
        assert update["sample_count"] == 2
        assert update["processed_windows"] == 2
        updates.append(update)
    aggregate = fedavg_aggregate([item["state_dict"] for item in updates],
                                 [item["sample_count"] for item in updates])
    assert set(aggregate) == set(state)
    assert not any("scaler" in key or "edge" in key or "node" in key for key in aggregate)
    assert len({item["sample_count"] for item in updates}) == 1


def test_equal_client_and_pooled_validation_metrics_are_distinct():
    rows = []
    for truth, prediction in ((np.array([1.0, 1.0]), np.array([0.0, 1.0])),
                              (np.array([100.0, 100.0]), np.array([99.0, 100.0]))):
        accumulator = MetricAccumulator()
        accumulator.update(truth, prediction, [True, True])
        rows.append({"metrics": accumulator.compute(), "statistics": accumulator.statistics()})
    equal, pooled = pooled_and_equal_metrics(rows)
    assert equal == pytest.approx((50.0 + 0.5) / 2)
    assert pooled["wape_percent"] == pytest.approx(100 * 2 / 202)
    assert equal != pooled["wape_percent"]


def test_federated_rounds_validation_only_checkpoint_and_exact_resume(tmp_path):
    prepared = synthetic_clients()
    settings = config(tmp_path, rounds=2)
    full = run_federated(settings, "gru_attention", "fedavg", run_dir=tmp_path / "full",
                         prepared_clients=prepared, max_train_batches=1, max_val_batches=1)
    uninterrupted = load_checkpoint(full / "latest_global.pt")
    assert uninterrupted["round"] == 2 and uninterrupted["smoke"]
    assert uninterrupted["current_validation"]["test_evaluated"] is False
    assert set(uninterrupted["current_validation"]["clients"]) == set(CLIENT_IDS)
    assert (full / "best_global.pt").exists() and (full / "best_clients_validation.json").exists()
    first = run_federated(config(tmp_path, rounds=1), "gru_attention", "fedavg",
                          run_dir=tmp_path / "resumed", prepared_clients=prepared,
                          max_train_batches=1, max_val_batches=1)
    train_resume = run_federated(settings, "gru_attention", "fedavg",
                                 resume=first / "latest_global.pt", prepared_clients=prepared,
                                 max_train_batches=1, max_val_batches=1)
    resumed = load_checkpoint(train_resume / "latest_global.pt")
    assert resumed["round"] == 2
    for name, value in uninterrupted["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][name])
    assert "client_metrics_json" in (full / "rounds.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "Round 02/2" in (full / "train.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("algorithm", ["fedavg", "fedprox"])
def test_fedavg_and_fedprox_train_all_variable_graph_clients(tmp_path, algorithm):
    prepared = synthetic_clients()
    run = run_federated(config(tmp_path, rounds=1), "stfednet_gated", algorithm,
                        run_dir=tmp_path / algorithm, prepared_clients=prepared,
                        max_train_batches=1, max_val_batches=1)
    latest = load_checkpoint(run / "latest_global.pt")
    assert latest["round"] == 1
    assert latest["algorithm"] == algorithm
    assert len(latest["round_sample_counts"]) == 5
    assert all(latest["round_sample_counts"][cid] == 2 for cid in CLIENT_IDS)
    assert latest["current_validation"]["test_evaluated"] is False


def test_centralized_uses_one_optimizer_sequential_variable_size_graphs(tmp_path):
    prepared = synthetic_clients()
    run = run_centralized(config(tmp_path, rounds=1), "stfednet_gated",
                          run_dir=tmp_path / "centralized", prepared_clients=prepared,
                          max_train_batches=1, max_val_batches=1)
    latest = load_checkpoint(run / "latest_global.pt")
    assert latest["algorithm"] == "centralized"
    assert latest["round"] == 1 and latest["optimizer_state_dict"] is not None
    assert latest["current_validation"]["test_evaluated"] is False
    assert tuple(latest["client_ids"]) == CLIENT_IDS


def test_centralized_resume_restores_latest_completed_round(tmp_path):
    prepared = synthetic_clients()
    settings = config(tmp_path, rounds=2)
    full_dir = run_centralized(settings, "gru_attention", run_dir=tmp_path / "central_full",
                               prepared_clients=prepared, max_train_batches=1, max_val_batches=1)
    full = load_checkpoint(full_dir / "latest_global.pt")
    first_dir = run_centralized(config(tmp_path, rounds=1), "gru_attention",
                                run_dir=tmp_path / "central_resume", prepared_clients=prepared,
                                max_train_batches=1, max_val_batches=1)
    resumed_dir = run_centralized(settings, "gru_attention",
                                  resume=first_dir / "latest_global.pt", prepared_clients=prepared,
                                  max_train_batches=1, max_val_batches=1)
    resumed = load_checkpoint(resumed_dir / "latest_global.pt")
    assert resumed["round"] == 2
    for name, value in full["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][name])


def test_g2_comparator_skips_smoke_and_rejects_protocol_mismatch(tmp_path):
    import os

    root = tmp_path / "results"
    base = {"kind": "best_global", "smoke": False, "algorithm": "fedavg",
            "model_name": "gru_attention", "seed": 42, "client_ids": list(CLIENT_IDS),
            "data_signatures": {client_id: "sig-" + client_id for client_id in CLIENT_IDS},
            "best_validation_wape": 20.0,
            "config": {"input_hours": 48, "horizon_hours": 1, "batch_size": 32,
                       "learning_rate": 0.001, "hidden_dim": 32, "dropout": 0.1},
            "current_validation": {"test_evaluated": False}}
    candidates = [
        (root / "fedavg" / "gru_attention" / "seed42_old" / "best_global.pt", dict(base, best_validation_wape=10.0)),
        (root / "fedavg" / "gru_attention" / "seed42_smoke" / "best_global.pt", dict(base, smoke=True)),
        (root / "fedavg" / "gru_attention" / "seed42_new" / "best_global.pt", dict(base, best_validation_wape=20.0)),
    ]
    for index, (path, payload) in enumerate(candidates):
        path.parent.mkdir(parents=True)
        torch.save(payload, path)
        os.utime(path, (1_700_000_000 + index, 1_700_000_000 + index))
    path, selected = find_best(root, "fedavg", "gru_attention", 42)
    assert path.parent.name == "seed42_new"
    assert selected["smoke"] is False
    with pytest.raises(ValueError, match="data_signatures"):
        check_protocol(dict(base, data_signatures={}), base)


@pytest.mark.parametrize("algorithm", ["fedavg", "fedprox"])
def test_mid_round_interruption_restarts_only_uncommitted_round(tmp_path, monkeypatch, algorithm):
    import csv
    import src.fl.server as server

    clients, settings = synthetic_clients(), config(tmp_path)
    # Nonzero dropout and multiple local steps exercise both the RNG and Prox.
    settings["dropout"] = 0.1
    full = run_federated(settings, "gru_attention", algorithm, run_dir=tmp_path / "full",
                         prepared_clients=clients, max_train_batches=2, max_val_batches=1)
    original_update = server.client_update

    def interrupted_update(*args, **kwargs):
        if args[4] == 2 and args[5] == 2:
            raise KeyboardInterrupt("Simulated interruption in client 3 of round 2")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(server, "client_update", interrupted_update)
    directory = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt):
        run_federated(settings, "gru_attention", algorithm, run_dir=directory,
                      prepared_clients=clients, max_train_batches=2, max_val_batches=1)
    latest = load_checkpoint(directory / "latest_global.pt")
    assert latest["round"] == 1
    # A process can stop after CSV/Best writes but before Latest is committed.
    with (directory / "rounds.csv").open("a", encoding="utf-8") as handle:
        handle.write("999,uncommitted\n")
    (directory / "best_global.pt").unlink()
    monkeypatch.setattr(server, "client_update", original_update)
    run_federated(settings, "gru_attention", algorithm, resume=directory / "latest_global.pt",
                  prepared_clients=clients, max_train_batches=2, max_val_batches=1)
    expected, actual = load_checkpoint(full / "latest_global.pt"), load_checkpoint(directory / "latest_global.pt")
    assert actual["round"] == 2
    assert all(torch.equal(expected["model_state_dict"][key], actual["model_state_dict"][key])
               for key in expected["model_state_dict"])
    assert expected["current_validation"] == actual["current_validation"]
    with (directory / "rounds.csv").open(encoding="utf-8") as handle:
        assert [int(row["round"]) for row in csv.DictReader(handle)] == [1, 2]
    assert (directory / "best_global.pt").exists()


def test_resume_cannot_relabel_smoke_as_formal(tmp_path):
    settings, clients = config(tmp_path, rounds=1), synthetic_clients()
    directory = run_federated(settings, "gru_attention", "fedavg", run_dir=tmp_path / "smoke",
                              prepared_clients=clients, max_train_batches=1, max_val_batches=1)
    with pytest.raises(ValueError, match="max_train_batches"):
        run_federated(dict(settings, rounds=2), "gru_attention", "fedavg",
                      resume=directory / "latest_global.pt", prepared_clients=clients)


@pytest.mark.parametrize("name", ["stfednet_gated", "gru_attention"])
def test_real_five_node_counts_share_global_weights(name):
    state = build_model(name).state_dict()
    for nodes in (39, 50, 56, 80, 76):
        model = build_model(name).eval()
        model.load_state_dict(state, strict=True)
        edges = torch.stack([torch.arange(nodes - 1), torch.arange(1, nodes)])
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        with torch.no_grad():
            prediction = model(torch.ones(1, 48, nodes, 1), edges)
        assert prediction.shape == (1, 1, nodes) and torch.isfinite(prediction).all()
