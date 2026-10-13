"""Bounded CUDA train/validation smoke run for the G1.1 residual model only."""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.train_clients import CLIENT_IDS
from src.models.forecasting import build_model
from src.training.engine import (load_checkpoint, prepare_data, read_config,
                                 seed_everything, select_device, train, write_json)


class ForbiddenTest:
    """Fail if this engineering check accidentally reads any test windows."""

    def __len__(self):
        raise AssertionError("G1.1 smoke accessed the test split")

    def __getitem__(self, index):
        raise AssertionError("G1.1 smoke accessed the test split")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", default="results/g1_1_smoke")
    args = parser.parse_args()
    config = read_config()
    config.update(output_root=args.output_root, device=args.device, epochs=1,
                  batch_size=2, max_train_batches=2, max_val_batches=2)
    root, records = Path(args.output_root), []
    for client in CLIENT_IDS:
        print(f"[G1.1 Smoke] {client}: bounded train/validation only", flush=True)
        prepared = prepare_data(config, client)
        prepared.test = ForbiddenTest()
        seed_everything(config["seed"], config["cpu_threads"])
        initial = build_model("stfednet_residual").state_dict()
        run_dir = train(config, "stfednet_residual", client, prepared=prepared)
        train(dict(config, epochs=2), "stfednet_residual", client,
              resume=run_dir / "latest.pt", prepared=prepared)
        best = load_checkpoint(run_dir / "best.pt")
        latest = load_checkpoint(run_dir / "latest.pt")
        assert best["smoke"] and best["epoch"] in (1, 2) and latest["epoch"] == 2
        assert np.isfinite(best["space_scale"])
        assert latest["space_scale"] != initial["spatial.space_scale"].item()
        assert not torch.equal(latest["model_state_dict"]["spatial.gat.projection.weight"],
                               initial["spatial.gat.projection.weight"])
        restored = build_model("stfednet_residual")
        restored.load_state_dict(best["model_state_dict"], strict=True)
        assert all(value is None or np.isfinite(value)
                   for value in best["validation_metrics"].values())
        records.append({"client_id": client, "num_nodes": prepared.grid.num_nodes,
                        "best_epoch": best["epoch"], "space_scale": best["space_scale"],
                        "latest_space_scale": latest["space_scale"],
                        "validation_metrics": best["validation_metrics"],
                        "checkpoint_reload_passed": True, "resume_passed": True,
                        "space_scale_update_passed": True, "gat_update_passed": True})
    write_json(root / "summary.json", {"model_name": "stfednet_residual", "smoke": True,
                                        "device": str(select_device(config["device"])),
                                        "torch": torch.__version__,
                                        "test_split_accessed": False,
                                        "clients": records})
    print(f"[G1.1 Smoke] All five clients passed; validation-only summary: {root / 'summary.json'}",
          flush=True)


if __name__ == "__main__":
    main()
