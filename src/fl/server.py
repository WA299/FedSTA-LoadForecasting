"""FedAvg/FedProx single-machine orchestration; validation only."""

import csv
import json
import time
from pathlib import Path

import torch

from src.fl.aggregation import fedavg_aggregate
from src.fl.client import client_update
from src.fl.common import (MODEL_NAMES, evaluate_clients, prepare_clients, signature_map,
                           validate_config, restore_round_outputs)
from src.models.forecasting import build_model
from src.training.engine import (atomic_save, load_checkpoint, random_state,
                                 restore_random_state, seed_everything,
                                 select_device, write_json)


def _new_run_dir(config, algorithm, model_name):
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns() % 1_000_000).zfill(6)
    return Path(config["output_root"]) / algorithm / model_name / ("seed" + str(config["seed"]) + "_" + stamp)


def _config_compatible(saved, current):
    mutable = {"rounds", "device", "cpu_threads", "output_root"}
    for key, value in saved.items():
        if key not in mutable and current.get(key) != value:
            raise ValueError("Resume configuration mismatch: " + key)


def _round_csv(run_dir, record):
    path = Path(run_dir) / "rounds.csv"
    fields = ["round", "algorithm", "model_name", "train_loss", "client_equal_wape_percent",
              "overall_wape_percent", "client_wapes_json", "client_metrics_json", "seconds", "is_best", "smoke"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key) for key in fields})


def run_federated(config, model_name="stfednet_gated", algorithm="fedavg", mu=None,
                  run_dir=None, resume=None, prepared_clients=None,
                  max_train_batches=None, max_val_batches=None):
    """Run complete communication rounds; never reads any client's test split."""
    if algorithm not in {"fedavg", "fedprox"}:
        raise ValueError("algorithm must be fedavg or fedprox")
    if model_name not in MODEL_NAMES:
        raise ValueError("G2 supports only " + ", ".join(MODEL_NAMES))
    config = dict(config)
    if max_train_batches is not None:
        config["max_train_batches"] = max_train_batches
    if max_val_batches is not None:
        config["max_val_batches"] = max_val_batches
    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")
    for key in ("max_train_batches", "max_val_batches"):
        if config[key] is not None and int(config[key]) <= 0:
            raise ValueError(key + " must be positive or null")
    config["mu"] = float(config.get("mu", 0.01) if mu is None else mu)
    validate_config(config)
    if config["mu"] < 0:
        raise ValueError("mu must be non-negative")
    seed_everything(config["seed"], config["cpu_threads"])
    device = select_device(config["device"])
    prepared_clients = prepared_clients or prepare_clients(config)
    if tuple(prepared_clients) != tuple(config["clients"]):
        raise ValueError("Prepared client order must match the configured five-client order")
    signatures = signature_map(prepared_clients)
    model = build_model(model_name, config["hidden_dim"], config["dropout"], config["horizon_hours"]).to(device)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    generator = torch.Generator().manual_seed(config["seed"])
    best_wape, best_round, best_state, best_validation = float("inf"), 0, None, None
    start_round = 1
    round_history = []
    if resume:
        latest = load_checkpoint(resume)
        if latest.get("kind") != "latest_global":
            raise ValueError("Resume requires latest_global.pt")
        if latest["algorithm"] != algorithm or latest["model_name"] != model_name:
            raise ValueError("Resume checkpoint algorithm/model mismatch")
        if latest["client_ids"] != list(prepared_clients) or latest["data_signatures"] != signatures:
            raise ValueError("Resume checkpoint client/data signature mismatch")
        _config_compatible(latest["config"], config)
        state = latest["model_state_dict"]
        best_state = latest["best_model_state_dict"]
        best_wape = latest["best_validation_wape"]
        best_round = latest["best_round"]
        best_validation = latest["best_validation"]
        start_round = int(latest["round"]) + 1
        restore_random_state(latest["rng_state"], generator)
        run_dir = Path(resume).resolve().parent
        round_history = list(latest.get("round_history", []))
        restore_round_outputs(latest, run_dir)
    else:
        run_dir = Path(run_dir) if run_dir else _new_run_dir(config, algorithm, model_name)
        if (run_dir / "latest_global.pt").exists():
            raise ValueError("Run directory already contains a checkpoint; use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {**config, "algorithm": algorithm, "model_name": model_name,
                                          "mu": config["mu"], "data_signatures": signatures,
                                          "client_ids": list(prepared_clients),
                                          "smoke": bool(max_train_batches or max_val_batches)})

    def log(message):
        print(message, flush=True)
        with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    smoke = bool(max_train_batches or max_val_batches)
    log("Algorithm=" + algorithm + " | Model=" + model_name + " | Clients=5/5 | Device=" + str(device) +
        " | BatchSize=" + str(config["batch_size"]) + " | LearningRate=" + str(config["learning_rate"]) +
        " | Rounds=" + str(config["rounds"]) + " | LocalEpochs=" + str(config["local_epochs"]) +
        " | Seed=" + str(config["seed"]) + (" | mu=" + str(config["mu"]) if algorithm == "fedprox" else ""))
    if smoke:
        log("[Smoke] Limited batches; smoke=true, validation only, test split not accessed")
    if start_round > int(config["rounds"]):
        log("[Resume] Latest checkpoint already completed all configured rounds")
        return run_dir

    for round_index in range(start_round, int(config["rounds"]) + 1):
        begin = time.perf_counter()
        round_start = {key: value.detach().cpu().clone() for key, value in state.items()}
        updates, counts, losses = [], [], []
        for client_index, client_id in enumerate(prepared_clients):
            update = client_update(round_start, prepared_clients[client_id], config, model_name,
                                   round_index, client_index, device, algorithm, config["mu"],
                                   max_train_batches)
            updates.append(update["state_dict"])
            counts.append(update["sample_count"])
            losses.append((update["mean_loss"], update["processed_windows"]))
        state = fedavg_aggregate(updates, counts)
        model.load_state_dict(state, strict=True)
        validation = evaluate_clients(model, prepared_clients, config, device, max_val_batches)
        equal_wape = validation["client_equal_wape_percent"]
        if equal_wape is None:
            raise ValueError("Undefined client-equal validation WAPE")
        improved = equal_wape < best_wape
        if improved:
            best_wape, best_round = equal_wape, round_index
            best_state = {key: value.detach().cpu().clone() for key, value in state.items()}
            best_validation = validation
        total = sum(count for _, count in losses)
        mean_loss = sum(loss * count for loss, count in losses) / total
        elapsed = time.perf_counter() - begin
        client_wapes = {cid: validation["clients"][cid]["wape_percent"] for cid in prepared_clients}
        record = {"round": round_index, "algorithm": algorithm, "model_name": model_name,
                  "train_loss": mean_loss, "client_equal_wape_percent": equal_wape,
                  "overall_wape_percent": validation["overall"]["wape_percent"],
                  "client_wapes_json": json.dumps(client_wapes, sort_keys=True),
                  "client_metrics_json": json.dumps(validation["clients"], sort_keys=True),
                  "seconds": elapsed, "is_best": improved, "smoke": smoke}
        _round_csv(run_dir, record)
        round_history.append(record)
        current_validation = {"round": round_index, **validation, "smoke": smoke,
                              "test_evaluated": False}
        if improved:
            best_validation = current_validation
        payload = {"format_version": 1, "kind": "latest_global", "model_name": model_name,
                   "algorithm": algorithm, "model_state_dict": state,
                   "best_model_state_dict": best_state, "round": round_index,
                   "best_round": best_round, "best_validation_wape": best_wape,
                   "best_validation": best_validation, "current_validation": current_validation,
                   "train_loss": mean_loss, "client_ids": list(prepared_clients),
                   "data_signatures": signatures, "config": config, "seed": config["seed"],
                   "mu": config["mu"] if algorithm == "fedprox" else None,
                   "rng_state": random_state(generator), "smoke": smoke,
                   "test_evaluated": False,
                   "optimizer_policy": "AdamW is reinitialized independently for each client at each round",
                   "round_sample_counts": dict(zip(prepared_clients, counts))}
        payload["round_history"] = round_history
        atomic_save(payload, run_dir / "latest_global.pt")
        if improved:
            best_payload = dict(payload, kind="best_global", model_state_dict=best_state,
                                round=best_round, current_validation=best_validation)
            atomic_save(best_payload, run_dir / "best_global.pt")
            best_json = {"algorithm": algorithm, "model_name": model_name, "seed": config["seed"],
                         "mu": config["mu"] if algorithm == "fedprox" else None,
                         "best_round": best_round, "best_validation_wape": best_wape,
                         "data_signatures": signatures, "client_ids": list(prepared_clients),
                         "smoke": smoke, "test_evaluated": False, **best_validation}
            write_json(run_dir / "best_clients_validation.json", best_json)
            log("[Checkpoint] New best global model saved")
        log("Round " + str(round_index).zfill(2) + "/" + str(config["rounds"]) +
            " | Algorithm=" + algorithm + " | Model=" + model_name +
            " | TrainLoss=" + format(mean_loss, ".6f") +
            " | AvgValWAPE=" + format(equal_wape, ".4f") + "% | OverallValWAPE=" +
            format(validation["overall"]["wape_percent"], ".4f") + "% | Best=" +
            format(best_wape, ".4f") + "% | ClientWAPEs=" + json.dumps(client_wapes, sort_keys=True) +
            " | Time=" + format(elapsed, ".2f") + "s")
    log("[Run] Global checkpoints and validation logs: " + str(run_dir))
    return run_dir
