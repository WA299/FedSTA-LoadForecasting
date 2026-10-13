"""Compare G2 validation-only best global checkpoints across algorithms/models."""

import argparse
import csv
from pathlib import Path

import numpy as np
from src.fl.common import CLIENT_IDS, MODEL_NAMES
from src.training.engine import load_checkpoint, write_json
from src.training.metrics import MetricAccumulator


def find_best(root, algorithm, model_name, seed):
    matches = []
    for path in Path(root).glob(f"{algorithm}/{model_name}/seed*/best_global.pt"):
        payload = load_checkpoint(path)
        if payload.get("kind") != "best_global" or payload.get("smoke"):
            continue
        if payload.get("algorithm") != algorithm or payload.get("model_name") != model_name:
            continue
        if payload.get("seed") != seed:
            continue
        if payload.get("client_ids") != list(CLIENT_IDS):
            continue
        if payload.get("test_evaluated", False) or payload.get("current_validation", {}).get("test_evaluated"):
            continue
        matches.append((path, payload))
    if not matches:
        raise FileNotFoundError(f"No formal G2 validation checkpoint for {algorithm}/{model_name}")
    # Do not choose a replicate by its score; use the most recently completed run.
    def run_stamp(pair):
        config_path = pair[0].parent / "config.json"
        return (config_path if config_path.exists() else pair[0]).stat().st_mtime
    return max(matches, key=run_stamp)


def check_protocol(payload, reference):
    for key in ("seed", "client_ids", "data_signatures"):
        if payload.get(key) != reference.get(key):
            raise ValueError("G2 checkpoint protocol mismatch: " + key)
    shared = ("input_hours", "horizon_hours", "train_ratio", "val_ratio", "batch_size",
              "learning_rate", "weight_decay", "hidden_dim", "dropout", "gat_heads",
              "huber_delta", "mape_threshold_kw", "local_epochs", "rounds")
    for key in shared:
        if payload.get("config", {}).get(key) != reference.get("config", {}).get(key):
            raise ValueError("G2 checkpoint protocol mismatch: " + key)
    if payload.get("algorithm") == reference.get("algorithm") == "fedprox" and payload.get("mu") != reference.get("mu"):
        raise ValueError("FedProx checkpoint mu mismatch")


def find_local_checkpoint(model_name, client_id, seed, data_signature):
    source = "results/g1_2" if model_name == "stfednet_gated" else "results/g1"
    matches = []
    for path in Path(source).glob(f"{model_name}/{client_id}/seed*/best.pt"):
        payload = load_checkpoint(path)
        if payload.get("kind") != "best" or payload.get("smoke"):
            continue
        if payload.get("model_name") != model_name or payload.get("client_id") != client_id:
            continue
        if payload.get("seed") != seed or payload.get("data_signature") != data_signature:
            continue
        matches.append((path, payload))
    if not matches:
        return None
    # The G1/G1.2 batch protocol uses the latest completed formal run per client.
    def run_stamp(pair):
        config_path = pair[0].parent / "config.json"
        return (config_path if config_path.exists() else pair[0]).stat().st_mtime
    return max(matches, key=run_stamp)


def add_local_only(records, reference, seed):
    """Summarize existing independent local checkpoints without pooling them into one model."""
    if reference is None:
        return
    for model_name in MODEL_NAMES:
        clients, paths, budgets, statistics = {}, {}, {}, []
        for client_id in CLIENT_IDS:
            found = find_local_checkpoint(model_name, client_id, seed,
                                          reference["data_signatures"][client_id])
            if found is None:
                clients = {}
                break
            path, checkpoint = found
            clients[client_id] = checkpoint["validation_metrics"]
            paths[client_id] = str(path.resolve())
            latest_path = path.with_name("latest.pt")
            latest = load_checkpoint(latest_path) if latest_path.exists() else None
            budgets[client_id] = {"configured_epochs": checkpoint["config"].get("epochs"),
                                  "best_epoch": checkpoint.get("epoch"),
                                  "completed_epochs": latest["epoch"] if latest else None}
            # G1 Best stores metrics instead of sufficient statistics. Reconstruct
            # sums algebraically, without running validation/test forwards.
            m = clients[client_id]
            error = m["mae_kw"] * m["effective_count"]
            if not m["wape_percent"]:
                statistics = []
                break
            statistics.append({"count": m["effective_count"], "mape_count": m["mape_valid_count"],
                               "absolute_error": error, "squared_error": m["rmse_kw"] ** 2 * m["effective_count"],
                               "absolute_true": 100 * error / m["wape_percent"],
                               "mape_sum": (m["mape_percent"] or 0) / 100 * m["mape_valid_count"]})
            local_protocol = ("input_hours", "horizon_hours", "train_ratio", "val_ratio", "batch_size",
                              "learning_rate", "weight_decay", "hidden_dim", "dropout", "huber_delta",
                              "mape_threshold_kw")
            for key in local_protocol:
                if checkpoint["config"].get(key) != reference["config"].get(key):
                    raise ValueError("Local-only checkpoint protocol mismatch: " + key)
        if len(clients) == len(CLIENT_IDS) and len(statistics) == len(CLIENT_IDS):
            values = [clients[cid]["wape_percent"] for cid in CLIENT_IDS]
            pooled = MetricAccumulator(reference["config"]["mape_threshold_kw"])
            for key in pooled.statistics():
                setattr(pooled, key, sum(row[key] for row in statistics))
            records.append({"algorithm": "local_only", "model_name": model_name,
                            "checkpoint_by_client": paths, "client_equal_wape_percent": float(np.mean(values)),
                            "overall": pooled.compute(), "clients": clients, "smoke": False,
                            "test_evaluated": False, "training_budget_by_client": budgets,
                            "pooled_statistics_source": "algebraic reconstruction from stored G1 validation metrics",
                            "note": "Five independent local models; no single shared global model."})


def local_reference(seed):
    """Bind a Local-only summary to stored configs and signatures, without data access."""
    for model_name in MODEL_NAMES:
        source = "results/g1_2" if model_name == "stfednet_gated" else "results/g1"
        checkpoints = []
        for client_id in CLIENT_IDS:
            matches = []
            for path in Path(source).glob(f"{model_name}/{client_id}/seed*/best.pt"):
                payload = load_checkpoint(path)
                if (payload.get("kind") == "best" and not payload.get("smoke") and
                        payload.get("seed") == seed and payload.get("client_id") == client_id and
                        payload.get("model_name") == model_name):
                    matches.append((path, payload))
            if not matches:
                break
            checkpoints.append(max(matches, key=lambda pair: pair[0].stat().st_mtime)[1])
        if len(checkpoints) == len(CLIENT_IDS):
            return {"seed": seed, "client_ids": list(CLIENT_IDS),
                    "config": checkpoints[0]["config"],
                    "data_signatures": {c["client_id"]: c["data_signature"] for c in checkpoints}}
    raise FileNotFoundError("No complete five-client formal Local-only checkpoint set found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/g2")
    parser.add_argument("--output", default="results/g2/comparison.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-only", action="store_true", help="Summarize existing Local-only runs before G2 experiments")
    args = parser.parse_args()
    records = []
    reference = None
    for algorithm in (() if args.local_only else ("fedavg", "fedprox", "centralized")):
        for model_name in MODEL_NAMES:
            try:
                path, payload = find_best(args.root, algorithm, model_name, args.seed)
            except FileNotFoundError:
                continue
            if reference is None:
                reference = payload
            else:
                check_protocol(payload, reference)
            validation = payload["current_validation"]
            latest_path = path.with_name("latest_global.pt")
            latest = load_checkpoint(latest_path) if latest_path.exists() else payload
            records.append({"algorithm": algorithm, "model_name": model_name,
                            "checkpoint": str(path.resolve()), "best_round": payload["best_round"],
                            "mu": payload.get("mu"),
                            "client_equal_wape_percent": validation["client_equal_wape_percent"],
                            "overall": validation["overall"], "clients": validation["clients"],
                            "smoke": False, "test_evaluated": False,
                            "training_budget": {"configured_rounds": payload["config"]["rounds"],
                                                "rounds_completed": latest["round"],
                                                "rounds_to_selected_model": payload["best_round"],
                                                "local_epochs_per_round": payload["config"]["local_epochs"],
                                                "total_local_epochs_per_client": latest["round"] * payload["config"]["local_epochs"]}})
    if args.local_only:
        reference = local_reference(args.seed)
    elif not records:
        raise FileNotFoundError("No matching formal G2 checkpoints found")
    add_local_only(records, reference, args.seed)
    ranked = sorted(records, key=lambda record: record["client_equal_wape_percent"])
    for rank, record in enumerate(ranked, start=1):
        record["rank_by_client_equal_wape"] = rank
    clients = {}
    wins_by_client = {}
    for client_id in CLIENT_IDS:
        clients[client_id] = {}
        for record in records:
            clients[client_id][record["algorithm"] + "/" + record["model_name"]] = record["clients"][client_id]
        comparable = [(name, metrics["wape_percent"]) for name, metrics in clients[client_id].items()
                      if metrics.get("wape_percent") is not None]
        minimum = min(value for _, value in comparable)
        wins_by_client[client_id] = {"best_wape_percent": minimum,
                                     "winning_methods": [name for name, value in comparable if value == minimum],
                                     "win_count": sum(value == minimum for _, value in comparable)}
    result = {"protocol": {"seed": args.seed, "clients": list(CLIENT_IDS),
                            "split": "validation", "test_evaluated": False},
              "selection": "latest formal completed run per method/model/seed; Best within run uses validation only",
              "runs": records, "client_metrics": clients, "wins_by_client": wins_by_client,
              "training_budget_note": "Local-only uses the recorded G1 per-client epoch budget; federated and centralized use their recorded communication-round schedules."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["algorithm", "model_name", "client_id", "mae_kw", "rmse_kw", "wape_percent", "mape_percent",
                  "mape_valid_fraction", "client_equal_wape_percent", "overall_wape_percent"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for client_id, metrics in record["clients"].items():
                writer.writerow({"algorithm": record["algorithm"], "model_name": record["model_name"],
                                 "client_id": client_id, **{key: metrics[key] for key in fields[3:8]},
                                 "client_equal_wape_percent": record["client_equal_wape_percent"],
                                 "overall_wape_percent": record["overall"]["wape_percent"]})
    print("[G2 Compare] Validation-only comparison written to " + str(Path(args.output).resolve()), flush=True)


if __name__ == "__main__":
    main()
