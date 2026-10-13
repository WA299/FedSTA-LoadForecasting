"""Two-round, five-client G2 CUDA smoke check; test split is forbidden."""

import argparse
from pathlib import Path

import torch

from src.fl.centralized import run_centralized
from src.fl.common import CLIENT_IDS, MODEL_NAMES, prepare_clients, read_g2_config
from src.fl.server import run_federated
from src.models.forecasting import build_model
from src.training.engine import load_checkpoint, seed_everything, select_device, write_json


class ForbiddenTest:
    def __len__(self):
        raise AssertionError("G2 smoke accessed the test split")

    def __getitem__(self, index):
        raise AssertionError("G2 smoke accessed the test split")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output-root", default="results/g2_smoke")
    args = parser.parse_args()
    config = read_g2_config(overrides={"device": args.device, "rounds": 2,
                                       "local_epochs": 1, "output_root": args.output_root,
                                       "batch_size": 2})
    prepared = prepare_clients(config)
    for client in prepared.values():
        client.test = ForbiddenTest()
    records = []
    recovery_logs = []
    for model_name in MODEL_NAMES:
        seed_everything(config["seed"], config["cpu_threads"])
        initial_state = {key: value.detach().cpu().clone()
                         for key, value in build_model(model_name, config["hidden_dim"],
                                                       config["dropout"], config["horizon_hours"]).state_dict().items()}
        federated_states = {}
        for algorithm in ("fedavg", "fedprox"):
            run = run_federated(config, model_name, algorithm, prepared_clients=prepared,
                                max_train_batches=2, max_val_batches=1)
            best = load_checkpoint(run / "best_global.pt")
            latest = load_checkpoint(run / "latest_global.pt")
            assert best["smoke"] and latest["smoke"] and latest["round"] == 2
            assert best["current_validation"]["test_evaluated"] is False
            diagnostic = latest["current_diagnostics"]
            assert len(diagnostic["clients"]) == 5
            assert diagnostic["summary"]["mean_total_loss"] == latest["train_loss"]
            assert 0 <= diagnostic["summary"]["clip_fraction"] <= 1
            if algorithm == "fedavg":
                assert diagnostic["summary"]["mean_prox_loss"] == 0
            else:
                assert diagnostic["summary"]["mean_prox_loss"] > 0
            federated_states[algorithm] = latest["model_state_dict"]
            assert any(not torch.equal(latest["model_state_dict"][name], initial_state[name])
                       for name in initial_state), "Global model parameters did not update"
            restored = build_model(model_name, config["hidden_dim"], config["dropout"],
                                   config["horizon_hours"])
            restored.load_state_dict(best["model_state_dict"], strict=True)
            recovery_config = dict(config, rounds=1, output_root=str(Path(args.output_root) / "recovery"))
            recovery_run = run_federated(recovery_config, model_name, algorithm,
                                         prepared_clients=prepared, max_train_batches=2, max_val_batches=1)
            run_federated(dict(recovery_config, rounds=2), model_name, algorithm,
                          resume=recovery_run / "latest_global.pt", prepared_clients=prepared,
                          max_train_batches=2, max_val_batches=1)
            recovered = load_checkpoint(recovery_run / "latest_global.pt")
            assert recovered["round"] == 2
            assert all(torch.equal(recovered["model_state_dict"][name], latest["model_state_dict"][name])
                       for name in latest["model_state_dict"])
            assert recovered["current_validation"] == latest["current_validation"]
            assert recovered["current_diagnostics"] == latest["current_diagnostics"]
            recovery_logs.append(recovery_run / "train.log")
            records.append({"model_name": model_name, "algorithm": algorithm,
                            "run_dir": str(run), "best_round": best["best_round"],
                            "latest_round": latest["round"], "smoke": True,
                            "test_split_accessed": False, "checkpoint_reload_passed": True,
                            "parameter_update_passed": True, "exact_resume_passed": True,
                            "best_validation": best["current_validation"],
                            "latest_train_loss": latest["train_loss"],
                            "round_sample_counts": latest["round_sample_counts"]})
            records[-1]["diagnostics"] = diagnostic
            records[-1]["diagnostics_resume_passed"] = True
        assert any(not torch.equal(federated_states["fedavg"][name],
                                   federated_states["fedprox"][name])
                   for name in federated_states["fedavg"]), "FedProx proximal update was not exercised"
        records[-1]["fedprox_update_distinct_from_fedavg"] = True
    for model_name in MODEL_NAMES:
        seed_everything(config["seed"], config["cpu_threads"])
        initial_state = {key: value.detach().cpu().clone()
                         for key, value in build_model(model_name, config["hidden_dim"],
                                                       config["dropout"], config["horizon_hours"]).state_dict().items()}
        run = run_centralized(config, model_name, prepared_clients=prepared,
                              max_train_batches=2, max_val_batches=1)
        latest = load_checkpoint(run / "latest_global.pt")
        assert latest["smoke"] and latest["round"] == 2
        assert any(not torch.equal(latest["model_state_dict"][name], initial_state[name])
                   for name in initial_state), "Centralized model parameters did not update"
        restored = build_model(model_name, config["hidden_dim"], config["dropout"],
                               config["horizon_hours"])
        restored.load_state_dict(load_checkpoint(run / "best_global.pt")["model_state_dict"], strict=True)
        recovery_config = dict(config, rounds=1, output_root=str(Path(args.output_root) / "recovery"))
        recovery_run = run_centralized(recovery_config, model_name, prepared_clients=prepared,
                                       max_train_batches=2, max_val_batches=1)
        run_centralized(dict(recovery_config, rounds=2), model_name,
                        resume=recovery_run / "latest_global.pt", prepared_clients=prepared,
                        max_train_batches=2, max_val_batches=1)
        recovered = load_checkpoint(recovery_run / "latest_global.pt")
        assert recovered["round"] == 2
        assert all(torch.equal(recovered["model_state_dict"][name], latest["model_state_dict"][name])
                   for name in latest["model_state_dict"])
        assert recovered["current_validation"] == latest["current_validation"]
        recovery_logs.append(recovery_run / "train.log")
        records.append({"model_name": model_name, "algorithm": "centralized",
                        "run_dir": str(run), "latest_round": latest["round"], "smoke": True,
                        "test_split_accessed": False, "checkpoint_reload_passed": True,
                        "parameter_update_passed": True, "exact_resume_passed": True,
                        "best_validation": latest["best_validation"],
                        "latest_train_loss": latest["train_loss"]})
    root = Path(args.output_root)
    write_json(root / "summary.json", {"smoke": True, "device": str(select_device(args.device)),
                                        "torch": str(torch.__version__), "test_split_accessed": False,
                                        "gpu": torch.cuda.get_device_name(select_device(args.device)) if select_device(args.device).type == "cuda" else None,
                                        "batch_size": config["batch_size"], "rounds": config["rounds"],
                                        "max_train_batches": 2, "max_val_batches": 1,
                                        "clients": list(CLIENT_IDS),
                                        "client_graphs": {cid: {"num_nodes": item.grid.num_nodes,
                                                                 "train_windows": len(item.train),
                                                                 "validation_windows": len(item.val)}
                                                          for cid, item in prepared.items()},
                                        "resume_recovery_passed": True, "records": records})
    logs = [Path(record["run_dir"]) / "train.log" for record in records] + recovery_logs
    (root / "smoke.log").write_text("\n".join(path.read_text(encoding="utf-8") for path in logs), encoding="utf-8")
    print("[G2 Smoke] smoke=true; six algorithm/model checks passed; test windows forbidden: " +
          str(root / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
