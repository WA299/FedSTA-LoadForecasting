"""Five-client FedPer simulation with a shared backbone and private heads."""

import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.fl.aggregation import fedavg_aggregate
from src.fl.client import client_seed
from src.fl.common import (MODEL_NAMES, pooled_and_equal_metrics,
                           prepare_clients, signature_map, validate_config)
from src.fl.diagnostics import summarize_diagnostics
from src.fl.server import _diagnostics_json
from src.models.forecasting import build_model
from src.training.engine import (atomic_save, load_checkpoint, random_state,
                                 restore_random_state, seed_everything, select_device,
                                 masked_huber, validate, write_json)


HEAD_KEYS = ("head.weight", "head.bias")


def split_fedper_state(state_dict):
    """Separate exactly the two local prediction-head tensors from a model state."""
    keys = set(state_dict)
    if not set(HEAD_KEYS).issubset(keys):
        raise ValueError("FedPer model must expose head.weight and head.bias")
    head = {key: state_dict[key].detach().cpu().clone() for key in HEAD_KEYS}
    shared = {key: value.detach().cpu().clone() for key, value in state_dict.items()
              if key not in HEAD_KEYS}
    if any(key.startswith("head.") for key in shared):
        raise ValueError("FedPer local head must contain only head.weight and head.bias")
    return shared, head


def combine_fedper_state(shared_state, head_state):
    """Construct one strict-loadable personalized model state."""
    if set(head_state) != set(HEAD_KEYS):
        raise ValueError("Personalized head keys must be exactly head.weight and head.bias")
    if any(key.startswith("head.") for key in shared_state):
        raise ValueError("Shared FedPer state must not contain prediction-head parameters")
    return {**shared_state, **head_state}


def fedper_aggregate(shared_states, sample_counts):
    """FedAvg aggregation restricted to shared backbone tensors."""
    if any(any(key.startswith("head.") for key in state) for state in shared_states):
        raise ValueError("Personalized prediction heads must never enter aggregation")
    return fedavg_aggregate(shared_states, sample_counts)


def fedper_client_update(shared_state, head_state, prepared, config, model_name,
                         round_index, client_index, device, max_train_batches=None):
    """Train one full personalized model and return separate backbone/head updates."""
    seed = client_seed(config["seed"], round_index, client_index)
    seed_everything(seed, config["cpu_threads"])
    model = build_model(model_name, config["hidden_dim"], config["dropout"],
                        config["horizon_hours"]).to(device)
    model.load_state_dict(combine_fedper_state(shared_state, head_state), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    dataset = prepared.train
    if max_train_batches:
        dataset = Subset(dataset, range(min(len(dataset), int(max_train_batches) * config["batch_size"])))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=config.get("num_workers", 0))
    edges = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
    mask = torch.as_tensor(prepared.grid.load_mask, dtype=torch.bool, device=device)
    observations, batches, loss_sum, grad_sum, clipped = 0, 0, 0.0, 0.0, 0
    for _ in range(int(config["local_epochs"])):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, edges)
            loss = masked_huber(prediction, y, mask, config["huber_delta"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite FedPer local objective for " + prepared.client_id)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            count = int(x.shape[0])
            observations += count
            batches += 1
            loss_sum += float(loss.detach()) * count
            grad_value = float(norm.detach())
            grad_sum += grad_value
            clipped += int(grad_value + 1e-6 > config["gradient_clip"])
    if observations <= 0 or batches <= 0:
        raise ValueError("FedPer client processed no training windows: " + prepared.client_id)
    new_shared, new_head = split_fedper_state(model.state_dict())
    mean_loss = loss_sum / observations
    return {"shared_state_dict": new_shared, "head_state_dict": new_head,
            "sample_count": len(dataset), "processed_windows": observations,
            "mean_loss": mean_loss,
            "diagnostics": {"huber_loss": mean_loss, "prox_loss": 0.0,
                            "total_loss": mean_loss, "grad_norm_before_clip": grad_sum / batches,
                            "clip_fraction": clipped / batches, "processed_windows": observations,
                            "batch_count": batches, "clipped_batch_count": clipped}}


@torch.no_grad()
def evaluate_personalized(shared_state, head_states, prepared_clients, config, device,
                          max_val_batches=None):
    """Validate each client with the shared backbone and its own retained head."""
    records = []
    if set(head_states) != set(prepared_clients):
        raise ValueError("Personalized head/client mapping mismatch")
    model = build_model(config["model_name"], config["hidden_dim"], config["dropout"],
                        config["horizon_hours"]).to(device)
    for client_id, prepared in prepared_clients.items():
        model.load_state_dict(combine_fedper_state(shared_state, head_states[client_id]), strict=True)
        metrics, statistics, _ = validate(model, prepared.val, prepared, config, device,
                                           max_val_batches, collect=False)
        records.append({"client_id": client_id, "metrics": metrics, "statistics": statistics})
    equal, overall = pooled_and_equal_metrics(records)
    return {"clients": {row["client_id"]: row["metrics"] for row in records},
            "client_equal_wape_percent": equal, "overall": overall,
            "statistics": {row["client_id"]: row["statistics"] for row in records}}


def _config_compatible(saved, current):
    mutable = {"rounds", "device", "cpu_threads", "output_root"}
    for key, value in saved.items():
        if key not in mutable and current.get(key) != value:
            raise ValueError("Resume configuration mismatch: " + key)


def _round_csv(run_dir, history):
    fields = ["round", "algorithm", "model_name", "train_loss", "client_equal_wape_percent",
              "overall_wape_percent", "client_wapes_json", "client_metrics_json", "seconds",
              "is_best", "smoke", "mean_huber_loss", "mean_prox_loss", "mean_total_loss",
              "mean_grad_norm_before_clip", "clip_fraction", "client_diagnostics_json"]
    with (Path(run_dir) / "rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in history)


def _restore_outputs(payload, run_dir):
    history = payload.get("round_history", [])
    if history:
        _round_csv(run_dir, history)
    best_validation = payload["best_validation"]
    best_payload = dict(payload, kind="best_fedper", round=payload["best_round"],
                        shared_state_dict=payload["best_shared_state_dict"],
                        head_state_dicts=payload["best_head_state_dicts"],
                        current_validation=best_validation,
                        current_diagnostics=payload["best_diagnostics"])
    best_payload.pop("rng_state", None)
    atomic_save(best_payload, Path(run_dir) / "best_fedper.pt")
    _diagnostics_json(run_dir, history)
    write_json(Path(run_dir) / "best_clients_validation.json", {
        "algorithm": "fedper", "model_name": payload["model_name"], "seed": payload["seed"],
        "best_round": payload["best_round"],
        "best_validation_wape": payload["best_validation_wape"],
        "data_signatures": payload["data_signatures"], "client_ids": payload["client_ids"],
        "smoke": payload["smoke"], "test_evaluated": False, **best_validation})


def _new_run_dir(config, model_name):
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns() % 1_000_000).zfill(6)
    return Path(config["output_root"]) / model_name / ("seed" + str(config["seed"]) + "_" + stamp)


def run_fedper(config, model_name="stfednet_gated", run_dir=None, resume=None,
               prepared_clients=None, max_train_batches=None, max_val_batches=None):
    """Run complete FedPer rounds; only validation data is read after aggregation."""
    if model_name not in MODEL_NAMES:
        raise ValueError("FedPer supports only " + ", ".join(MODEL_NAMES))
    config = dict(config)
    config["model_name"] = model_name
    if max_train_batches is not None:
        config["max_train_batches"] = max_train_batches
    if max_val_batches is not None:
        config["max_val_batches"] = max_val_batches
    for key in ("max_train_batches", "max_val_batches"):
        if config.get(key) is not None and int(config[key]) <= 0:
            raise ValueError(key + " must be positive or null")
    validate_config(config)
    seed_everything(config["seed"], config["cpu_threads"])
    device = select_device(config["device"])
    prepared_clients = prepared_clients or prepare_clients(config)
    if tuple(prepared_clients) != tuple(config["clients"]):
        raise ValueError("Prepared client order must match the configured five-client order")
    signatures = signature_map(prepared_clients)
    initial_model = build_model(model_name, config["hidden_dim"], config["dropout"],
                                config["horizon_hours"]).to(device)
    shared_state, initial_head = split_fedper_state(initial_model.state_dict())
    head_states = {client_id: {key: value.clone() for key, value in initial_head.items()}
                   for client_id in prepared_clients}
    generator = torch.Generator().manual_seed(config["seed"])
    best_wape, best_round = float("inf"), 0
    best_shared, best_heads, best_validation, best_diagnostics = None, None, None, None
    start_round, history = 1, []
    if resume:
        latest = load_checkpoint(resume)
        if latest.get("kind") != "latest_fedper":
            raise ValueError("Resume requires latest_fedper.pt")
        if latest["model_name"] != model_name or latest["algorithm"] != "fedper":
            raise ValueError("Resume checkpoint model/algorithm mismatch")
        if latest["client_ids"] != list(prepared_clients) or latest["data_signatures"] != signatures:
            raise ValueError("Resume checkpoint client/data signature mismatch")
        _config_compatible(latest["config"], config)
        if bool(latest["smoke"]) != bool(config.get("max_train_batches") or config.get("max_val_batches")):
            raise ValueError("Cannot resume FedPer across smoke/formal modes")
        shared_state = latest["shared_state_dict"]
        head_states = latest["head_state_dicts"]
        if set(head_states) != set(prepared_clients):
            raise ValueError("FedPer checkpoint personalized head/client mapping mismatch")
        best_shared = latest["best_shared_state_dict"]
        best_heads = latest["best_head_state_dicts"]
        if set(best_heads) != set(prepared_clients):
            raise ValueError("FedPer best checkpoint personalized head/client mapping mismatch")
        best_wape, best_round = latest["best_validation_wape"], latest["best_round"]
        best_validation = latest["best_validation"]
        best_diagnostics = latest.get("best_diagnostics")
        start_round = int(latest["round"]) + 1
        history = list(latest.get("round_history", []))
        restore_random_state(latest["rng_state"], generator)
        run_dir = Path(resume).resolve().parent
        _restore_outputs(latest, run_dir)
    else:
        run_dir = Path(run_dir) if run_dir else _new_run_dir(config, model_name)
        if (run_dir / "latest_fedper.pt").exists():
            raise ValueError("Run directory already contains a checkpoint; use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    smoke = bool(config.get("max_train_batches") or config.get("max_val_batches"))
    metadata = {**config, "algorithm": "fedper", "client_ids": list(prepared_clients),
                "data_signatures": signatures, "smoke": smoke,
                "shared_parameter_keys": list(shared_state), "personalized_parameter_keys": list(HEAD_KEYS),
                "optimizer_policy": "one fresh AdamW optimizer per client per communication round"}
    write_json(run_dir / "config.json", metadata)

    def log(message):
        print(message, flush=True)
        with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log("Algorithm=FedPer | Model=" + model_name + " | Clients=5/5 | Device=" + str(device) +
        " | BatchSize=" + str(config["batch_size"]) + " | LearningRate=" + str(config["learning_rate"]) +
        " | Rounds=" + str(config["rounds"]) + " | LocalEpochs=" + str(config["local_epochs"]) +
        " | Seed=" + str(config["seed"]))
    if smoke:
        log("[Smoke] Limited batches; smoke=true, validation only, test split not accessed")
    if start_round > int(config["rounds"]):
        log("[Resume] Latest FedPer checkpoint already completed all configured rounds")
        return run_dir

    for round_index in range(start_round, int(config["rounds"]) + 1):
        begin = time.perf_counter()
        round_start = {key: value.detach().cpu().clone() for key, value in shared_state.items()}
        updates, counts, client_losses, diagnostics = [], [], [], {}
        next_heads = {}
        for client_index, client_id in enumerate(prepared_clients):
            result = fedper_client_update(round_start, head_states[client_id],
                                          prepared_clients[client_id], config, model_name,
                                          round_index, client_index, device,
                                          config.get("max_train_batches"))
            updates.append(result["shared_state_dict"])
            counts.append(result["sample_count"])
            client_losses.append((result["mean_loss"], result["processed_windows"]))
            diagnostics[client_id] = result["diagnostics"]
            next_heads[client_id] = result["head_state_dict"]
        shared_state = fedper_aggregate(updates, counts)
        head_states = next_heads
        validation = evaluate_personalized(shared_state, head_states, prepared_clients, config,
                                           device, config.get("max_val_batches"))
        equal_wape = validation["client_equal_wape_percent"]
        if equal_wape is None:
            raise ValueError("Undefined FedPer client-equal validation WAPE")
        improved = equal_wape < best_wape
        if improved:
            best_wape, best_round = equal_wape, round_index
            best_shared = {key: value.detach().cpu().clone() for key, value in shared_state.items()}
            best_heads = {cid: {key: value.detach().cpu().clone() for key, value in head_states[cid].items()}
                          for cid in prepared_clients}
            best_validation = {"round": round_index, **validation, "smoke": smoke,
                               "test_evaluated": False}
        total = sum(count for _, count in client_losses)
        mean_loss = sum(loss * count for loss, count in client_losses) / total
        diag_summary = summarize_diagnostics(diagnostics)
        if improved:
            best_diagnostics = {"round": round_index, "summary": diag_summary, "clients": diagnostics}
        client_wapes = {cid: validation["clients"][cid]["wape_percent"] for cid in prepared_clients}
        elapsed = time.perf_counter() - begin
        row = {"round": round_index, "algorithm": "fedper", "model_name": model_name,
               "train_loss": mean_loss, "client_equal_wape_percent": equal_wape,
               "overall_wape_percent": validation["overall"]["wape_percent"],
               "client_wapes_json": json.dumps(client_wapes, sort_keys=True),
               "client_metrics_json": json.dumps(validation["clients"], sort_keys=True),
               "client_diagnostics_json": json.dumps(diagnostics, sort_keys=True),
               "seconds": elapsed, "is_best": improved, "smoke": smoke, **diag_summary}
        history.append(row)
        _round_csv(run_dir, history)
        current_validation = {"round": round_index, **validation, "smoke": smoke,
                              "test_evaluated": False}
        payload = {"format_version": 1, "kind": "latest_fedper", "algorithm": "fedper",
                   "model_name": model_name, "shared_state_dict": shared_state,
                   "head_state_dicts": head_states, "best_shared_state_dict": best_shared,
                   "best_head_state_dicts": best_heads, "round": round_index,
                   "best_round": best_round, "best_validation_wape": best_wape,
                   "best_validation": best_validation, "current_validation": current_validation,
                   "train_loss": mean_loss, "client_ids": list(prepared_clients),
                   "data_signatures": signatures, "config": config, "seed": config["seed"],
                   "rng_state": random_state(generator), "smoke": smoke,
                   "test_evaluated": False,
                   "round_sample_counts": dict(zip(prepared_clients, counts)),
                   "current_diagnostics": {"round": round_index, "summary": diag_summary,
                                            "clients": diagnostics},
                   "best_diagnostics": best_diagnostics,
                   "shared_parameter_keys": list(shared_state),
                   "personalized_parameter_keys": list(HEAD_KEYS),
                   "optimizer_policy": metadata["optimizer_policy"],
                   "round_history": history}
        atomic_save(payload, run_dir / "latest_fedper.pt")
        _diagnostics_json(run_dir, history)
        if improved:
            best_payload = dict(payload, kind="best_fedper", round=best_round,
                                shared_state_dict=best_shared, head_state_dicts=best_heads,
                                current_validation=best_validation)
            best_payload.pop("rng_state", None)
            atomic_save(best_payload, run_dir / "best_fedper.pt")
            write_json(run_dir / "best_clients_validation.json", {
                "algorithm": "fedper", "model_name": model_name, "seed": config["seed"],
                "best_round": best_round, "best_validation_wape": best_wape,
                "data_signatures": signatures, "client_ids": list(prepared_clients),
                "smoke": smoke, "test_evaluated": False, **best_validation})
            log("[Checkpoint] New best personalized model set saved")
        log("Round " + str(round_index).zfill(2) + "/" + str(config["rounds"]) +
            " | Algorithm=FedPer | Model=" + model_name + " | TrainLoss=" + format(mean_loss, ".6f") +
            " | AvgValWAPE=" + format(equal_wape, ".4f") + "% | OverallValWAPE=" +
            format(validation["overall"]["wape_percent"], ".4f") + "% | Best=" +
            format(best_wape, ".4f") + "% | ClientWAPEs=" + json.dumps(client_wapes, sort_keys=True) +
            " | Huber=" + format(diag_summary["mean_huber_loss"], ".6f") +
            " | GradNorm=" + format(diag_summary["mean_grad_norm_before_clip"], ".6f") +
            " | ClipFraction=" + format(diag_summary["clip_fraction"], ".4f") +
            " | Time=" + format(elapsed, ".2f") + "s")
    log("[Run] Personalized Best/Latest checkpoints and validation logs: " + str(run_dir))
    return run_dir
