"""Five-client bounded CUDA checks; train/validation only, never test windows."""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.smoke_g11 import ForbiddenTest
from scripts.train_clients import CLIENT_IDS
from src.models.forecasting import build_model
from src.training.engine import (load_checkpoint, loader, prepare_data, read_config,
                                 seed_everything, select_device, train, validate, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--output-root", default="results/g1_2_smoke")
    args = parser.parse_args()
    config = dict(read_config(), epochs=1, batch_size=2, max_train_batches=2,
                  max_val_batches=2, device=args.device, output_root=args.output_root)
    device = select_device(args.device)
    records = []
    for client in CLIENT_IDS:
        prepared = prepare_data(config, client)
        prepared.test = ForbiddenTest()
        seed_everything(config["seed"], config["cpu_threads"])
        initial = build_model("stfednet_gated").state_dict()
        run = train(config, "stfednet_gated", client, prepared=prepared)
        train(dict(config, epochs=2), "stfednet_gated", client,
              resume=run / "latest.pt", prepared=prepared)
        best, latest = load_checkpoint(run / "best.pt"), load_checkpoint(run / "latest.pt")
        assert best["smoke"] and latest["epoch"] == 2
        for key in ("spatial.gate.weight", "spatial.gate.bias", "spatial.gat.projection.weight"):
            assert not torch.equal(latest["model_state_dict"][key], initial[key])
        restored = build_model("stfednet_gated").to(device).eval()
        restored.load_state_dict(best["model_state_dict"], strict=True)
        metrics, stats, _ = validate(restored, prepared.val, prepared, config, device, max_batches=2)
        assert metrics == best["validation_metrics"]
        assert stats["gate_statistics"] == best["validation_gate_statistics"]
        assert all(value is None or np.isfinite(value) for value in metrics.values())
        x, _ = next(iter(loader(prepared.val, config, max_batches=1)))
        edges = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
        with torch.no_grad():
            prediction, gate = restored.forward_with_gate(x.to(device), edges)
        assert prediction.shape == (2, 1, prepared.grid.num_nodes)
        assert gate.shape == (2, 48, prepared.grid.num_nodes, 1)
        assert torch.all((gate >= 0) & (gate <= 1))
        records.append({"client_id": client, "nodes": prepared.grid.num_nodes,
                        "run_dir": str(run), "validation_metrics": metrics,
                        "best_gate_statistics": stats["gate_statistics"],
                        "resume_passed": True, "strict_reload_passed": True,
                        "gat_and_gate_updates_passed": True, "gate_range_passed": True})
    root = Path(args.output_root)
    write_json(root / "summary.json", {"model_name": "stfednet_gated", "smoke": True,
                                        "device": str(device), "torch": torch.__version__,
                                        "test_split_accessed": False, "clients": records})
    print(f"[G1.2 Smoke] Five clients passed; no test windows accessed: {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
