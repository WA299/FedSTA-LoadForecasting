"""Compare best validation WAPE from formal checkpoints; never loads test data."""

import argparse
import json
from pathlib import Path

from src.training.engine import load_checkpoint, write_json


CLIENTS = ("lv_39", "lv_50", "lv_56", "lv_80", "industrial_mvlv")
REFERENCE_MODELS = ("gru_attention", "stfednet")
PROTOCOL_KEYS = ("input_hours", "horizon_hours", "train_ratio", "val_ratio", "seed",
                 "hidden_dim", "dropout", "batch_size", "learning_rate", "epochs",
                 "patience", "weight_decay", "huber_delta", "min_delta",
                 "gradient_clip", "mape_threshold_kw")


def check_protocol(reference, candidate):
    mismatches = [key for key in PROTOCOL_KEYS
                  if reference["config"].get(key) != candidate["config"].get(key)]
    if mismatches:
        raise ValueError(f"Incomparable validation checkpoints: protocol mismatch in {mismatches}")
    if reference["data_signature"] != candidate["data_signature"]:
        raise ValueError("Incomparable validation checkpoints: data/signature mismatch")


def find_best(root, model_name, client_id, seed):
    candidates = []
    for path in (Path(root) / model_name / client_id).glob("*/best.pt"):
        checkpoint = load_checkpoint(path)
        if (checkpoint.get("kind") != "best" or checkpoint.get("smoke")
                or checkpoint.get("model_name") != model_name
                or checkpoint.get("client_id") != client_id
                or checkpoint.get("seed") != seed):
            continue
        candidates.append((path.stat().st_mtime_ns, path, checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No formal seed={seed} best checkpoint for {model_name}/{client_id} under {root}")
    return max(candidates, key=lambda item: item[0])[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-root", default="results/g1")
    parser.add_argument("--g11-root", default="results/g1_1")
    parser.add_argument("--clients", nargs="+", choices=CLIENTS, default=list(CLIENTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/g1_1/validation_comparison.json")
    args = parser.parse_args()
    if len(set(args.clients)) != len(args.clients):
        parser.error("Each client must appear once")

    comparisons = []
    for client in args.clients:
        runs = {}
        for model in (*REFERENCE_MODELS, "stfednet_residual"):
            root = args.g11_root if model == "stfednet_residual" else args.g1_root
            path, checkpoint = find_best(root, model, client, args.seed)
            runs[model] = (path, checkpoint)
        reference = runs["stfednet_residual"][1]
        for model, (_, checkpoint) in runs.items():
            check_protocol(reference, checkpoint)
        residual_wape = runs["stfednet_residual"][1]["validation_metrics"]["wape_percent"]
        row = {"client_id": client, "seed": args.seed, "validation_wape_percent": {
            model: checkpoint["validation_metrics"]["wape_percent"] for model, (_, checkpoint) in runs.items()},
            "best_epoch": {model: checkpoint["epoch"] for model, (_, checkpoint) in runs.items()},
            "run_dir": {model: str(path.parent.resolve()) for model, (path, _) in runs.items()},
            "residual_beats": {model: residual_wape < runs[model][1]["validation_metrics"]["wape_percent"]
                               for model in REFERENCE_MODELS}}
        comparisons.append(row)
    output = {"split": "validation", "test_evaluated": False, "seed": args.seed,
              "client_count": len(comparisons), "comparisons": comparisons,
              "client_equal_validation_wape_percent": {
                  model: sum(row["validation_wape_percent"][model] for row in comparisons) / len(comparisons)
                  for model in (*REFERENCE_MODELS, "stfednet_residual")},
              "residual_wins": {model: sum(row["residual_beats"][model] for row in comparisons)
                                for model in REFERENCE_MODELS}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
