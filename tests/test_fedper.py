"""FedPer parameter partitioning, personalization, and recovery tests."""

import pytest
import torch

from scripts.compare_fedper import _scan_fedper
from test_g2 import ForbiddenTest, config, synthetic_clients
from src.fl.common import CLIENT_IDS
from src.fl.fedper import (combine_fedper_state, evaluate_personalized,
                           fedper_aggregate, fedper_client_update, run_fedper,
                           split_fedper_state)
from src.models.forecasting import build_model
from src.training.engine import load_checkpoint, seed_everything, validate


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    torch.set_num_threads(1)


def g2_config(tmp_path, rounds=2):
    value = config(tmp_path, rounds)
    value["output_root"] = str(tmp_path / "fedper")
    return value


@pytest.mark.parametrize("model_name", ["stfednet_gated", "gru_attention"])
def test_fedper_partition_and_node_independent_backbone(model_name):
    seed_everything(42, 1)
    model = build_model(model_name, hidden_dim=32, dropout=0.0)
    state = model.state_dict()
    shared, head = split_fedper_state(state)
    assert set(head) == {"head.weight", "head.bias"}
    assert not any(key.startswith("head.") for key in shared)
    assert {name: tuple(value.shape) for name, value in shared.items()} == {
        name: tuple(value.shape) for name, value in state.items() if name not in head}
    combined = combine_fedper_state(shared, head)
    for nodes in (39, 50, 56, 80, 76):
        receiver = build_model(model_name, hidden_dim=32, dropout=0.0)
        receiver.load_state_dict(combined, strict=True)
        edge = torch.stack([torch.arange(nodes - 1), torch.arange(1, nodes)])
        edge = torch.cat([edge, edge.flip(0)], dim=1)
        prediction = receiver(torch.ones(1, 48, nodes, 1), edge)
        assert prediction.shape == (1, 1, nodes)


def test_fedper_aggregation_is_weighted_and_rejects_heads():
    a = {"weight": torch.tensor([1.0]), "bias": torch.tensor([3.0])}
    b = {"weight": torch.tensor([5.0]), "bias": torch.tensor([7.0])}
    result = fedper_aggregate([a, b], [1, 3])
    assert result["weight"].item() == pytest.approx(4.0)
    assert result["bias"].item() == pytest.approx(6.0)
    with pytest.raises(ValueError, match="heads"):
        fedper_aggregate([{"head.weight": torch.ones(1)}], [1])


def test_initial_shared_backbone_and_heads_match_seeded_single_model():
    seed_everything(42, 1)
    fedavg_initial = build_model("stfednet_gated", hidden_dim=32, dropout=0.1).state_dict()
    seed_everything(42, 1)
    fedper_initial = build_model("stfednet_gated", hidden_dim=32, dropout=0.1).state_dict()
    shared, initial_head = split_fedper_state(fedper_initial)
    assert all(torch.equal(fedavg_initial[key], shared[key]) for key in shared)
    assert all(torch.equal(fedavg_initial[key], initial_head[key]) for key in initial_head)
    client_heads = [{key: value.clone() for key, value in initial_head.items()}
                    for _ in CLIENT_IDS]
    assert all(all(torch.equal(client_heads[0][key], head[key]) for key in initial_head)
               for head in client_heads[1:])


def test_head_updates_and_personalized_validation_uses_the_matching_client_head(tmp_path):
    clients = synthetic_clients()
    cfg = g2_config(tmp_path)
    seed_everything(42, 1)
    model = build_model("gru_attention", dropout=0.0)
    shared, initial_head = split_fedper_state(model.state_dict())
    heads = {}
    for index, cid in enumerate(CLIENT_IDS):
        heads[cid] = {key: value.clone() for key, value in initial_head.items()}
        heads[cid]["head.bias"].add_(float(index))
    updated = {}
    for index, cid in enumerate(CLIENT_IDS):
        item = fedper_client_update(shared, heads[cid], clients[cid], cfg, "gru_attention",
                                    1, index, torch.device("cpu"), 1)
        updated[cid] = item["head_state_dict"]
        assert not torch.equal(updated[cid]["head.bias"], heads[cid]["head.bias"])
    assert len({float(updated[cid]["head.bias"].item()) for cid in CLIENT_IDS}) == 5

    for index, cid in enumerate(CLIENT_IDS):
        updated[cid]["head.bias"].add_(index * 0.25)
    eval_config = cfg | {"model_name": "gru_attention"}
    actual = evaluate_personalized(shared, updated, clients, eval_config, torch.device("cpu"), 1)
    for cid in CLIENT_IDS:
        model_copy = build_model("gru_attention", dropout=0.0)
        model_copy.load_state_dict(combine_fedper_state(shared, updated[cid]), strict=True)
        expected, _, _ = validate(model_copy, clients[cid].val, clients[cid], eval_config,
                                  torch.device("cpu"), 1)
        assert actual["clients"][cid] == expected


def test_best_round_contains_all_heads_and_resume_matches_continuous_training(tmp_path):
    clients = synthetic_clients()
    settings = g2_config(tmp_path, rounds=2) | {"dropout": 0.1}
    full_dir = run_fedper(settings, "gru_attention", run_dir=tmp_path / "full",
                          prepared_clients=clients, max_train_batches=1, max_val_batches=1)
    full = load_checkpoint(full_dir / "latest_fedper.pt")
    best = load_checkpoint(full_dir / "best_fedper.pt")
    assert best["best_round"] == best["round"] == best["current_validation"]["round"]
    assert set(best["head_state_dicts"]) == set(CLIENT_IDS)
    assert all(set(head) == {"head.weight", "head.bias"}
               for head in best["head_state_dicts"].values())
    assert best["smoke"] and best["test_evaluated"] is False

    first_dir = run_fedper(g2_config(tmp_path, rounds=1) | {"dropout": 0.1},
                           "gru_attention", run_dir=tmp_path / "resume",
                           prepared_clients=clients, max_train_batches=1, max_val_batches=1)
    resumed_dir = run_fedper(settings, "gru_attention",
                              resume=first_dir / "latest_fedper.pt",
                              prepared_clients=clients, max_train_batches=1, max_val_batches=1)
    resumed = load_checkpoint(resumed_dir / "latest_fedper.pt")
    assert resumed["round"] == full["round"] == 2
    for key in full["shared_state_dict"]:
        assert torch.equal(full["shared_state_dict"][key], resumed["shared_state_dict"][key])
    for cid in CLIENT_IDS:
        for key in full["head_state_dicts"][cid]:
            assert torch.equal(full["head_state_dicts"][cid][key],
                              resumed["head_state_dicts"][cid][key])
    assert full["current_validation"] == resumed["current_validation"]


def test_fedper_does_not_access_test_dataset_and_smoke_checkpoint_is_excluded(tmp_path):
    clients = synthetic_clients()
    for client in clients.values():
        client.test = ForbiddenTest()
    root = tmp_path / "fedper"
    run = run_fedper(g2_config(tmp_path, rounds=1), "gru_attention",
                     run_dir=root / "gru_attention" / "seed42_smoke",
                     prepared_clients=clients, max_train_batches=1, max_val_batches=1)
    checkpoint = load_checkpoint(run / "latest_fedper.pt")
    assert checkpoint["current_validation"]["test_evaluated"] is False
    selected, skipped = _scan_fedper(root, "gru_attention", 42, rounds=1)
    assert selected is None
    assert any("smoke" in item["reason"] for item in skipped)


def test_old_g2_model_checkpoints_remain_strict_load_compatible():
    state = build_model("stfednet_gated").state_dict()
    receiver = build_model("stfednet_gated")
    receiver.load_state_dict(state, strict=True)
    assert set(receiver.state_dict()) == set(state)
