"""Sequential five-client train/validation runs; never invokes test evaluation."""

import argparse
from pathlib import Path

from src.models.forecasting import MODEL_NAMES
from src.training.engine import load_checkpoint, read_config, train, write_json


CLIENT_IDS = ("lv_39", "lv_50", "lv_56", "lv_80", "industrial_mvlv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="stfednet_residual")
    parser.add_argument("--clients", choices=CLIENT_IDS, nargs="+", default=list(CLIENT_IDS))
    parser.add_argument("--config", default="configs/g1.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-root", default="results/g1_1")
    parser.add_argument("--summary", default="results/g1_1/batch_summary.json",
                        help="JSON manifest of best validation results; use an empty string to disable")
    args = parser.parse_args()
    if len(set(args.clients)) != len(args.clients):
        parser.error("Each client must appear once")
    config = read_config(args.config)
    config["output_root"] = args.output_root
    if args.device:
        config["device"] = args.device
    records = []
    for client in args.clients:
        directory = train(config, args.model, client)
        best = load_checkpoint(directory / "best.pt")
        records.append({"client_id": client, "run_dir": str(directory.resolve()),
                        "best_epoch": best["epoch"], "validation": best["validation_metrics"],
                        "space_scale": best.get("space_scale"), "smoke": best["smoke"]})
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.summary, {"model_name": args.model, "seed": config["seed"],
                                  "split": "validation", "test_evaluated": False,
                                  "results": records})
    print("[Batch] Train/validation completed for all requested clients; no test evaluation", flush=True)


if __name__ == "__main__":
    main()
