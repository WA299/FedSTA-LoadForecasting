"""A few train/validation batches for every model/client; never formal training."""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.forecasting import MODEL_NAMES
from src.training.engine import (evaluate, load_checkpoint, prepare_data, read_config,
                                 select_device, train, write_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", default="results/g1_smoke")
    args = parser.parse_args()
    config = read_config()
    config.update(epochs=1, batch_size=2, max_train_batches=2, max_val_batches=2,
                  device=args.device, output_root=args.output_root)
    root = Path(args.output_root)
    results = []
    clients = ("lv_39", "lv_50", "lv_56", "lv_80", "industrial_mvlv")
    for client in clients:
        print(f"[Data] Loading {client} once for all seven methods", flush=True)
        prepared = prepare_data(config, client)
        for name in MODEL_NAMES:
            directory = root / name / client
            run = train(config, name, client, run_dir=directory, prepared=prepared)
            best, latest = load_checkpoint(run / "best.pt"), load_checkpoint(run / "latest.pt")
            assert best["model_name"] == name and latest["epoch"] == 1
            if name not in {"persistence", "seasonal_naive"}:
                resumed = dict(config, epochs=2)
                train(resumed, name, client, resume=run / "latest.pt", prepared=prepared)
                assert load_checkpoint(run / "latest.pt")["epoch"] == 2
            tested = evaluate(run / "best.pt", device=args.device, max_batches=2, prepared=prepared)
            assert all(value is None or np.isfinite(value) for value in tested["metrics"].values())
            results.append({"model": name, "client": client, "nodes": prepared.grid.num_nodes,
                            "best_epoch": load_checkpoint(run / "best.pt")["epoch"],
                            "validation": load_checkpoint(run / "best.pt")["validation_metrics"],
                            "smoke_test": tested["metrics"], "checkpoint_reload_passed": True,
                            "resume_passed": name not in {"persistence", "seasonal_naive"}})
    write_json(root / "summary.json", {"device": str(select_device(args.device)), "smoke_only": True,
                                      "cases": len(results), "results": results})
    print(f"[Smoke] Passed {len(results)} model/client combinations; summary: {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
