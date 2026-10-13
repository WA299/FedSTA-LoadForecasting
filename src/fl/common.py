"""Shared G2 configuration, data preparation and validation utilities."""

import csv
import os
from pathlib import Path

import numpy as np
import yaml

from src.training.engine import atomic_save, data_signature, prepare_data, write_json
from src.training.metrics import MetricAccumulator


CLIENT_IDS = ("lv_39", "lv_50", "lv_56", "lv_80", "industrial_mvlv")
MODEL_NAMES = ("stfednet_gated", "gru_attention")


def restore_round_outputs(checkpoint, run_dir):
    """The atomic latest checkpoint is the commit point for a complete round.

    Rebuild the CSV/Best outputs if interruption occurred between their writes.
    The append-only text log may contain interrupted attempts; CSV is canonical.
    """
    run_dir = Path(run_dir)
    history = checkpoint.get("round_history")
    if history:
        path = run_dir / "rounds.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            fields = list(dict.fromkeys(key for row in history for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, path)
    validation = checkpoint["best_validation"]
    best = dict(checkpoint, kind="best_global", round=checkpoint["best_round"],
                model_state_dict=checkpoint["best_model_state_dict"], current_validation=validation)
    if "best_diagnostics" in checkpoint:
        best["current_diagnostics"] = checkpoint["best_diagnostics"]
    best.pop("optimizer_state_dict", None)
    atomic_save(best, run_dir / "best_global.pt")
    write_json(run_dir / "best_clients_validation.json", {
        "algorithm": checkpoint["algorithm"], "model_name": checkpoint["model_name"],
        "seed": checkpoint["seed"], "mu": checkpoint.get("mu"),
        "best_round": checkpoint["best_round"], "best_validation_wape": checkpoint["best_validation_wape"],
        "data_signatures": checkpoint["data_signatures"], "client_ids": checkpoint["client_ids"],
        "smoke": checkpoint["smoke"], "test_evaluated": False, **validation})


def read_g2_config(path="configs/g2.yaml", overrides=None):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    config.update(overrides or {})
    return validate_config(config)


def validate_config(config):
    if config["input_hours"] != 48 or config["horizon_hours"] != 1:
        raise ValueError("G2 protocol requires input_hours=48 and horizon_hours=1")
    if not np.isclose(config["train_ratio"], 0.70) or not np.isclose(config["val_ratio"], 0.15):
        raise ValueError("G2 must preserve the G1 chronological 70/15/15 split")
    if tuple(config["clients"]) != CLIENT_IDS:
        raise ValueError("G2 client order must include all five fixed clients")
    if int(config.get("client_participation", len(CLIENT_IDS))) != len(CLIENT_IDS):
        raise ValueError("G2 currently requires full 5/5 client participation")
    if config.get("gat_heads", 2) != 2:
        raise ValueError("Frozen G1/G1.2 architecture requires two GAT heads")
    for key in ("rounds", "local_epochs", "batch_size", "cpu_threads"):
        if int(config[key]) <= 0:
            raise ValueError(key + " must be positive")
    for key in ("learning_rate", "huber_delta", "gradient_clip", "mape_threshold_kw"):
        if not np.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(key + " must be positive")
    if not np.isfinite(float(config["weight_decay"])) or float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not np.isfinite(float(config["mu"])) or float(config["mu"]) < 0:
        raise ValueError("mu must be non-negative")
    for key in ("max_train_batches", "max_val_batches"):
        if config.get(key) is not None and int(config[key]) <= 0:
            raise ValueError(key + " must be positive or null")
    if not config.get("output_root"):
        raise ValueError("output_root is required")
    return config


def prepare_clients(config, clients=None):
    ids = tuple(clients or config["clients"])
    if len(ids) != len(set(ids)) or any(cid not in CLIENT_IDS for cid in ids):
        raise ValueError("Unknown or duplicate G2 clients")
    return {cid: prepare_data(config, cid) for cid in ids}


def signature_map(prepared_clients):
    return {cid: data_signature(prepared_clients[cid]) for cid in prepared_clients}


def pooled_and_equal_metrics(records):
    """Combine validation sufficient statistics without conflating WAPEs."""
    if not records:
        raise ValueError("At least one client validation record is required")
    pooled = MetricAccumulator(records[0]["metrics"]["mape_threshold_kw"])
    for record in records:
        stats = record["statistics"]
        for key in pooled.statistics():
            setattr(pooled, key, getattr(pooled, key) + stats[key])
    client_wapes = [record["metrics"]["wape_percent"] for record in records]
    equal = float(np.mean(client_wapes)) if all(value is not None for value in client_wapes) else None
    return equal, pooled.compute()


def evaluate_clients(model, prepared_clients, config, device, max_val_batches=None):
    """Evaluate the same model separately on each client's validation split."""
    from src.training.engine import validate

    records = []
    for cid, prepared in prepared_clients.items():
        metrics, statistics, _ = validate(
            model, prepared.val, prepared, config, device,
            max_batches=max_val_batches, collect=False)
        records.append({"client_id": cid, "metrics": metrics, "statistics": statistics})
    equal, overall = pooled_and_equal_metrics(records)
    return {"clients": {row["client_id"]: row["metrics"] for row in records},
            "client_equal_wape_percent": equal,
            "overall": overall,
            "statistics": {row["client_id"]: row["statistics"] for row in records}}
