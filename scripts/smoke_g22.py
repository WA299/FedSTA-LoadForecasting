"""Five-real-client bounded CUDA FedPer smoke checks."""

import json
from pathlib import Path
from unittest.mock import patch
import time

import torch

from src.fl.common import prepare_clients, read_g2_config
import src.fl.fedper as fedper
from src.fl.fedper import combine_fedper_state, run_fedper, split_fedper_state
from src.fl.aggregation import fedavg_aggregate
from src.models.forecasting import build_model
from src.training.engine import load_checkpoint, seed_everything


class ForbiddenTest:
    def __len__(self):
        raise AssertionError("FedPer smoke accessed test split")

    def __getitem__(self, index):
        raise AssertionError("FedPer smoke accessed test split")


def equal_state(first, second):
    return set(first) == set(second) and all(torch.equal(first[key], second[key]) for key in first)


def main():
    root = Path(".pytest_tmp/g2_2_cuda_smoke") / str(time.time_ns())
    root.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("G2.2 CUDA smoke requested but CUDA is unavailable")
    summaries = []
    base_config = read_g2_config("configs/g2_2.yaml")
    prepared = prepare_clients(base_config)
    for client in prepared.values():
        client.test = ForbiddenTest()
    for model_name in ("stfednet_gated", "gru_attention"):
        seed_everything(42, 1)
        cfg = read_g2_config("configs/g2_2.yaml", {
            "output_root": str(root / model_name), "rounds": 2, "device": "cuda",
            "cpu_threads": 1, "batch_size": 2, "max_train_batches": 1,
            "max_val_batches": 1})
        initial_shared, initial_head = split_fedper_state(
            build_model(model_name, cfg["hidden_dim"], cfg["dropout"]).to("cuda").state_dict())
        observed_heads = {cid: {key: value.clone() for key, value in initial_head.items()}
                          for cid in cfg["clients"]}
        seen_updates = []
        original_update, original_aggregate = fedper.fedper_client_update, fedper.fedper_aggregate

        def checked_update(shared, head, client, *args):
            assert equal_state(head, observed_heads[client.client_id])
            result = original_update(shared, head, client, *args)
            assert not equal_state(result["head_state_dict"], head)
            assert not equal_state(result["shared_state_dict"], shared)
            observed_heads[client.client_id] = result["head_state_dict"]
            seen_updates.append((client.client_id, result))
            return result

        def checked_aggregate(states, counts):
            assert len(states) == 5 and all(not key.startswith("head.") for state in states for key in state)
            result = original_aggregate(states, counts)
            assert equal_state(result, fedavg_aggregate(states, counts))
            return result

        with patch.object(fedper, "fedper_client_update", checked_update), patch.object(
                fedper, "fedper_aggregate", checked_aggregate):
            run = run_fedper(cfg, model_name, run_dir=root / model_name / "full",
                             prepared_clients=prepared)
        latest = load_checkpoint(run / "latest_fedper.pt")
        best = load_checkpoint(run / "best_fedper.pt")
        assert latest["round"] == 2 and latest["smoke"] and best["smoke"]
        assert latest["test_evaluated"] is False
        assert set(latest["head_state_dicts"]) == set(cfg["clients"])
        assert set(best["head_state_dicts"]) == set(cfg["clients"])
        assert len(seen_updates) == 10
        assert not equal_state(initial_shared, latest["shared_state_dict"])
        assert all(equal_state(latest["head_state_dicts"][cid], observed_heads[cid]) for cid in cfg["clients"])
        for checkpoint in (best, latest):
            for cid in cfg["clients"]:
                receiver = build_model(model_name, cfg["hidden_dim"], cfg["dropout"]).to("cuda")
                receiver.load_state_dict(combine_fedper_state(
                    checkpoint["shared_state_dict"], checkpoint["head_state_dicts"][cid]), strict=True)
        first = run_fedper(dict(cfg, rounds=1), model_name,
                           run_dir=root / model_name / "resumed", prepared_clients=prepared)
        run_fedper(cfg, model_name, resume=first / "latest_fedper.pt", prepared_clients=prepared)
        resumed = load_checkpoint(first / "latest_fedper.pt")
        assert equal_state(latest["shared_state_dict"], resumed["shared_state_dict"])
        assert all(equal_state(latest["head_state_dicts"][cid], resumed["head_state_dicts"][cid])
                   for cid in cfg["clients"])
        assert latest["current_validation"] == resumed["current_validation"]
        summaries.append({"model_name": model_name, "run_dir": str(run.resolve()),
                          "round": latest["round"], "smoke": True,
                          "best_round": best["best_round"],
                          "client_equal_validation_wape_percent": best["best_validation_wape"],
                          "head_clients": list(latest["head_state_dicts"]),
                          "head_updates_and_cross_round_retention_checked": True,
                          "weighted_backbone_only_aggregation_checked": True,
                          "strict_load_combinations": 10,
                          "resume_exact_on_cuda": True,
                          "node_counts": {cid: len(prepared[cid].grid.node_ids) for cid in cfg["clients"]},
                          "train_batches_per_client_per_round": 1,
                          "validation_batches_per_client_per_round": 1,
                          "batch_size": cfg["batch_size"],
                          "test_evaluated": False})
    result = {"smoke": True, "device": str(torch.cuda.get_device_name(0)),
              "clients": ["lv_39", "lv_50", "lv_56", "lv_80", "industrial_mvlv"],
              "models": summaries, "formal_training": False, "test_evaluated": False}
    Path("results/g2_2").mkdir(parents=True, exist_ok=True)
    Path("results/g2_2/cuda_smoke_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
