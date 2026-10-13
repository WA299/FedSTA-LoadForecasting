"""Compare four models using formal Best checkpoint validation metadata only."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_validation import CLIENTS, check_protocol, find_best
from src.training.engine import write_json


MODELS = ("gru_attention", "stfednet", "stfednet_residual", "stfednet_gated")
METRICS = ("mae_kw", "rmse_kw", "wape_percent", "mape_percent")


def compare(roots, clients, seed):
    rows = []
    for client in clients:
        runs = {model: find_best(roots[model], model, client, seed) for model in MODELS}
        reference = runs["stfednet_gated"][1]
        metrics = {}
        for model, (_, saved) in runs.items():
            if saved["client_id"] != client or saved["seed"] != seed or saved["config"]["seed"] != seed:
                raise ValueError("Client or seed mismatch")
            if saved["config"]["input_hours"] != 48 or saved["config"]["horizon_hours"] != 1:
                raise ValueError("G1.2 requires 48-to-1")
            if saved["config"].get("max_train_batches") or saved["config"].get("max_val_batches"):
                raise ValueError("Limited-batch checkpoints are not formal comparisons")
            check_protocol(reference, saved)
            metrics[model] = {key: saved["validation_metrics"][key] for key in METRICS}
            for key, value in metrics[model].items():
                if (value is None and key != "mape_percent") or (value is not None and not np.isfinite(value)):
                    raise ValueError("Nonfinite or undefined validation metrics")
        rows.append({"client_id": client, "seed": seed, "validation_metrics": metrics,
                     "best_epoch": {model: saved["epoch"] for model, (_, saved) in runs.items()},
                     "checkpoint": {model: str(path.resolve()) for model, (path, _) in runs.items()},
                     "gated_gate_statistics": reference.get("validation_gate_statistics"),
                     "gated_beats": {model: metrics["stfednet_gated"]["wape_percent"] < metrics[model]["wape_percent"]
                                     for model in MODELS[:-1]}})
    return {"split": "validation", "test_evaluated": False, "seed": seed,
            "client_count": len(rows), "complete_five_clients": set(clients) == set(CLIENTS),
            "selection": "latest formal best.pt per model/client/seed by modification time",
            "comparisons": rows, "client_equal_validation_wape_percent": {
                model: sum(row["validation_metrics"][model]["wape_percent"] for row in rows) / len(rows)
                for model in MODELS},
            "gated_wins": {model: sum(row["gated_beats"][model] for row in rows) for model in MODELS[:-1]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-root", default="results/g1")
    parser.add_argument("--g11-root", default="results/g1_1")
    parser.add_argument("--g12-root", default="results/g1_2")
    parser.add_argument("--clients", nargs="+", choices=CLIENTS, default=list(CLIENTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/g1_2/validation_comparison.json")
    args = parser.parse_args()
    if len(set(args.clients)) != len(args.clients):
        parser.error("Each client may appear once")
    roots = dict(zip(MODELS, (args.g1_root, args.g1_root, args.g11_root, args.g12_root)))
    result = compare(roots, args.clients, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
