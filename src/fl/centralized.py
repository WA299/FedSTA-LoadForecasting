"""Centralized shared-model control with variable-size client graphs."""

import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.fl.client import client_seed
from src.fl.common import (evaluate_clients, prepare_clients, signature_map, validate_config,
                           restore_round_outputs)
from src.models.forecasting import build_model
from src.training.engine import (atomic_save, load_checkpoint, masked_huber,
                                 random_state, restore_random_state,
                                 seed_everything, select_device, write_json)


def _new_run_dir(config, model_name):
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns() % 1_000_000).zfill(6)
    return Path(config["output_root"]) / "centralized" / model_name / ("seed" + str(config["seed"]) + "_" + stamp)


def _write_round(run_dir, record):
    path = Path(run_dir) / "rounds.csv"
    fields = ["round", "algorithm", "model_name", "train_loss", "client_equal_wape_percent",
              "overall_wape_percent", "client_wapes_json", "client_metrics_json", "seconds", "is_best", "smoke"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: record.get(key) for key in fields})


def run_centralized(config, model_name="stfednet_gated", run_dir=None, resume=None,
                    prepared_clients=None, max_train_batches=None, max_val_batches=None):
    """One model/optimizer trains sequential client batches in a fixed schedule."""
    if model_name not in {"stfednet_gated", "gru_attention"}:
        raise ValueError("G2 centralized supports stfednet_gated and gru_attention")
    config = dict(config)
    if max_train_batches is not None:
        config["max_train_batches"] = max_train_batches
    if max_val_batches is not None:
        config["max_val_batches"] = max_val_batches
    max_train_batches = config.get("max_train_batches")
    max_val_batches = config.get("max_val_batches")
    validate_config(config)
    for key in ("max_train_batches", "max_val_batches"):
        if config[key] is not None and int(config[key]) <= 0:
            raise ValueError(key + " must be positive or null")
    seed_everything(config["seed"], config["cpu_threads"])
    device = select_device(config["device"])
    prepared_clients = prepared_clients or prepare_clients(config)
    if tuple(prepared_clients) != tuple(config["clients"]):
        raise ValueError("Prepared client order must match all configured clients")
    signatures = signature_map(prepared_clients)
    model = build_model(model_name, config["hidden_dim"], config["dropout"], config["horizon_hours"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    start_round, best_round, best_wape = 1, 0, float("inf")
    best_state, best_validation = None, None
    round_history = []
    if resume:
        latest = load_checkpoint(resume)
        if latest.get("kind") != "latest_global" or latest.get("algorithm") != "centralized":
            raise ValueError("Resume requires a centralized latest_global.pt")
        if latest["model_name"] != model_name or latest["client_ids"] != list(prepared_clients):
            raise ValueError("Resume checkpoint model/client mismatch")
        if latest["data_signatures"] != signatures:
            raise ValueError("Resume checkpoint data signature mismatch")
        mutable = {"rounds", "device", "cpu_threads", "output_root"}
        for key, value in latest["config"].items():
            if key not in mutable and config.get(key) != value:
                raise ValueError("Resume configuration mismatch: " + key)
        model.load_state_dict(latest["model_state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        start_round, best_round, best_wape = latest["round"] + 1, latest["best_round"], latest["best_validation_wape"]
        best_state, best_validation = latest["best_model_state_dict"], latest["best_validation"]
        restore_random_state(latest["rng_state"], torch.Generator().manual_seed(config["seed"]))
        run_dir = Path(resume).resolve().parent
        round_history = list(latest.get("round_history", []))
        restore_round_outputs(latest, run_dir)
    else:
        run_dir = Path(run_dir) if run_dir else _new_run_dir(config, model_name)
        if (run_dir / "latest_global.pt").exists():
            raise ValueError("Run directory already contains a checkpoint; use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {**config, "algorithm": "centralized", "model_name": model_name,
                                          "data_signatures": signatures, "client_ids": list(prepared_clients),
                                          "centralized_schedule": "fixed client order, one shuffled train pass per client per round",
                                          "smoke": bool(max_train_batches or max_val_batches)})

    def log(message):
        print(message, flush=True)
        with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    smoke = bool(max_train_batches or max_val_batches)
    log("Algorithm=Centralized | Model=" + model_name + " | Clients=5/5 | Device=" + str(device) +
        " | BatchSize=" + str(config["batch_size"]) + " | LearningRate=" + str(config["learning_rate"]) +
        " | Rounds=" + str(config["rounds"]) + " | Seed=" + str(config["seed"]))
    if smoke:
        log("[Smoke] Limited batches; smoke=true, validation only, test split not accessed")
    if start_round > int(config["rounds"]):
        log("[Resume] Latest checkpoint already completed all configured rounds")
        return run_dir

    for round_index in range(start_round, int(config["rounds"]) + 1):
        begin = time.perf_counter()
        model.train()
        total_loss, observations = 0.0, 0
        for client_index, (client_id, prepared) in enumerate(prepared_clients.items()):
            local_seed = client_seed(config["seed"], round_index, client_index)
            seed_everything(local_seed, config["cpu_threads"])
            dataset = prepared.train
            if max_train_batches:
                dataset = Subset(dataset, range(min(len(dataset), int(max_train_batches) * config["batch_size"])))
            batches = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                                 generator=torch.Generator().manual_seed(local_seed),
                                 num_workers=config.get("num_workers", 0))
            edges = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
            mask = torch.as_tensor(prepared.grid.load_mask, dtype=torch.bool, device=device)
            for _ in range(int(config["local_epochs"])):
                for x, y in batches:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = masked_huber(model(x, edges), y, mask, config["huber_delta"])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite centralized loss on " + client_id)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                    optimizer.step()
                    count = int(x.shape[0])
                    total_loss += float(loss.detach()) * count
                    observations += count
        if observations <= 0:
            raise ValueError("Centralized round processed no windows")
        validation = evaluate_clients(model, prepared_clients, config, device, max_val_batches)
        equal_wape = validation["client_equal_wape_percent"]
        if equal_wape is None:
            raise ValueError("Undefined centralized client-equal validation WAPE")
        improved = equal_wape < best_wape
        if improved:
            best_wape, best_round = equal_wape, round_index
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_validation = {"round": round_index, **validation, "smoke": smoke,
                               "test_evaluated": False}
        mean_loss = total_loss / observations
        elapsed = time.perf_counter() - begin
        client_wapes = {cid: validation["clients"][cid]["wape_percent"] for cid in prepared_clients}
        record = {"round": round_index, "algorithm": "centralized", "model_name": model_name,
                  "train_loss": mean_loss, "client_equal_wape_percent": equal_wape,
                  "overall_wape_percent": validation["overall"]["wape_percent"],
                  "client_wapes_json": json.dumps(client_wapes, sort_keys=True),
                  "client_metrics_json": json.dumps(validation["clients"], sort_keys=True), "seconds": elapsed,
                  "is_best": improved, "smoke": smoke}
        _write_round(run_dir, record)
        round_history.append(record)
        payload = {"format_version": 1, "kind": "latest_global", "algorithm": "centralized",
                   "model_name": model_name,
                   "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                   "optimizer_state_dict": optimizer.state_dict(), "best_model_state_dict": best_state,
                   "round": round_index, "best_round": best_round, "best_validation_wape": best_wape,
                   "best_validation": best_validation, "current_validation": {"round": round_index, **validation,
                                                                                "smoke": smoke, "test_evaluated": False},
                   "train_loss": mean_loss, "client_ids": list(prepared_clients),
                   "data_signatures": signatures, "config": config, "seed": config["seed"],
                   "mu": None, "rng_state": None, "smoke": smoke,
                   "test_evaluated": False,
                   "training_schedule": "fixed client order, one shuffled train pass per client per round"}
        payload["round_history"] = round_history
        generator = torch.Generator().manual_seed(config["seed"])
        payload["rng_state"] = random_state(generator)
        atomic_save(payload, run_dir / "latest_global.pt")
        if improved:
            atomic_save(dict(payload, kind="best_global", model_state_dict=best_state,
                             round=best_round, current_validation=best_validation), run_dir / "best_global.pt")
            write_json(run_dir / "best_clients_validation.json", {
                "algorithm": "centralized", "model_name": model_name, "seed": config["seed"],
                "best_round": best_round, "best_validation_wape": best_wape,
                "data_signatures": signatures, "client_ids": list(prepared_clients),
                "smoke": smoke, "test_evaluated": False, **best_validation})
            log("[Checkpoint] New best global model saved")
        log("Round " + str(round_index).zfill(2) + "/" + str(config["rounds"]) +
            " | Algorithm=Centralized | Model=" + model_name + " | TrainLoss=" + format(mean_loss, ".6f") +
            " | AvgValWAPE=" + format(equal_wape, ".4f") + "% | OverallValWAPE=" +
            format(validation["overall"]["wape_percent"], ".4f") + "% | Best=" + format(best_wape, ".4f") +
            "% | ClientWAPEs=" + json.dumps(client_wapes, sort_keys=True) + " | Time=" + format(elapsed, ".2f") + "s")
    log("[Run] Centralized checkpoints and validation logs: " + str(run_dir))
    return run_dir
