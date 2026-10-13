"""Read-only selection/protocol utilities for G2.1 validation analyses."""

import math
from pathlib import Path

from src.fl.common import CLIENT_IDS, MODEL_NAMES
from src.fl.diagnostics import DIAGNOSTIC_FIELDS
from src.training.engine import load_checkpoint

PROTOCOL_FIELDS = ("input_hours", "horizon_hours", "train_ratio", "val_ratio",
                   "hidden_dim", "dropout", "gat_heads", "batch_size", "learning_rate",
                   "weight_decay", "rounds", "local_epochs", "client_participation",
                   "huber_delta", "gradient_clip", "mape_threshold_kw", "num_workers",
                   "max_train_batches", "max_val_batches")
SELECTION = "latest completed formal run by latest_global.pt mtime_ns, then absolute path; never by WAPE"


def check_matched_protocol(payload, reference):
    for key in ("model_name", "seed", "client_ids", "data_signatures"):
        if payload.get(key) != reference.get(key):
            raise ValueError("Sensitivity protocol mismatch: " + key)
    for key in PROTOCOL_FIELDS:
        if payload["config"].get(key) != reference["config"].get(key):
            raise ValueError("Sensitivity protocol mismatch: " + key)
    # mu is intentionally the one algorithmic treatment allowed to differ.


def scan_runs(root, algorithm, seed=None, rounds=20):
    """Require Best+Latest from a complete run; return skipped-run reasons too."""
    runs, skipped = [], []
    for path in sorted(Path(root).glob(f"{algorithm}/*/seed*/best_global.pt")):
        best = load_checkpoint(path)
        latest_path = path.with_name("latest_global.pt")
        reason = None
        if best.get("kind") != "best_global" or best.get("algorithm") != algorithm:
            reason = "checkpoint kind/algorithm mismatch"
        elif best.get("smoke") or best.get("config", {}).get("max_train_batches") or best.get("config", {}).get("max_val_batches"):
            reason = "smoke or limited batches"
        elif best.get("model_name") not in MODEL_NAMES or (seed is not None and best.get("seed") != seed):
            reason = "model/seed outside requested group"
        elif rounds is not None and best["config"].get("rounds") != rounds:
            reason = "different configured round budget (original baseline retained)"
        elif not latest_path.exists():
            reason = "missing latest checkpoint; completion cannot be verified"
        if reason:
            skipped.append({"checkpoint": str(path.resolve()), "reason": reason})
            continue
        latest = load_checkpoint(latest_path)
        check_matched_protocol(latest, best)
        if latest.get("kind") != "latest_global" or latest.get("algorithm") != algorithm:
            raise ValueError("Latest kind/algorithm mismatch: " + str(path))
        if latest.get("smoke") or latest.get("mu") != best.get("mu"):
            raise ValueError("Latest smoke/mu mismatch: " + str(path))
        if latest["round"] != best["config"]["rounds"]:
            skipped.append({"checkpoint": str(path.resolve()), "reason": "incomplete round budget"})
            continue
        for checkpoint in (best, latest):
            cfg = checkpoint["config"]
            if (checkpoint.get("test_evaluated") or checkpoint["current_validation"].get("test_evaluated")
                    or checkpoint["current_validation"].get("smoke")):
                raise ValueError("Only formal validation summaries are permitted")
            if checkpoint["client_ids"] != list(CLIENT_IDS) or set(checkpoint["data_signatures"]) != set(CLIENT_IDS):
                raise ValueError("Missing five-client data signatures")
            if not all(isinstance(value, str) and value for value in checkpoint["data_signatures"].values()):
                raise ValueError("Empty data signature")
            if cfg["input_hours"] != 48 or cfg["horizon_hours"] != 1 or cfg["train_ratio"] != 0.7 or cfg["val_ratio"] != 0.15:
                raise ValueError("Expected unchanged G2 48-to-1 chronological protocol")
            if cfg.get("seed") != checkpoint["seed"] or cfg.get("clients") != list(CLIENT_IDS):
                raise ValueError("Config seed/client metadata mismatch")
        if algorithm == "fedprox":
            mu = best.get("mu")
            if mu is None or not math.isfinite(mu) or mu < 0 or mu != best["config"].get("mu"):
                raise ValueError("Invalid or conflicting FedProx mu")
        validation = best["current_validation"]
        if set(validation["clients"]) != set(CLIENT_IDS):
            raise ValueError("Incomplete five-client validation metrics")
        equal = sum(validation["clients"][cid]["wape_percent"] for cid in CLIENT_IDS) / len(CLIENT_IDS)
        if not math.isclose(equal, validation["client_equal_wape_percent"], rel_tol=1e-10):
            raise ValueError("Client equal WAPE inconsistent with client metrics")
        if not math.isclose(equal, best["best_validation_wape"], rel_tol=1e-10):
            raise ValueError("Best WAPE inconsistent with validation summary")
        if not 1 <= best["best_round"] <= latest["round"] or validation["round"] != best["best_round"]:
            raise ValueError("Best validation round mismatch")
        runs.append({"path": path, "best": best, "latest": latest,
                     "selection_key": (latest_path.stat().st_mtime_ns, str(path.resolve()))})
    return runs, skipped


def select_latest(runs):
    return max(runs, key=lambda run: run["selection_key"])


def diagnostic_summary(checkpoint):
    summary = (checkpoint.get("current_diagnostics") or {}).get("summary", {})
    return {key: summary.get(key) for key in DIAGNOSTIC_FIELDS}
