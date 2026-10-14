"""One Poisson update per real client/model on CUDA; no test data or formal run."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from src.fl.common import prepare_clients
from src.fl.dp_server import read_g31_config, run_dp_federated
import src.fl.dp_server as dp_server
from src.models.forecasting import build_model
from src.training.engine import load_checkpoint, seed_everything


class ForbiddenTest:
    def __len__(self):
        raise AssertionError("G3.1 smoke tried to access the test split")

    def __getitem__(self, index):
        raise AssertionError("G3.1 smoke tried to access the test split")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("G3.1 CUDA smoke requires the configured NVIDIA GPU")
    cfg = read_g31_config("configs/g3_1.yaml", {
        "output_root": ".pytest_tmp/g3_1_cuda_smoke", "rounds": 1,
        "local_epochs": 1, "noise_multiplier": 1.0, "max_grad_norm": 1.0,
        "delta": 1e-5, "device": "cuda", "gradient_method": "auto",
        "privacy_mode": "research", "max_dp_steps_per_epoch": 1,
        "max_val_batches": 1, "seed": 42,
    })
    prepared = prepare_clients(cfg)
    for client in prepared.values():
        client.test = ForbiddenTest()
    wrong_counts = {cid: len(client.train) for cid, client in prepared.items()
                    if len(client.train) != 6084}
    if wrong_counts:
        raise AssertionError("Unexpected real-client training window counts: " + str(wrong_counts))
    summaries = []
    root = Path(".pytest_tmp/g3_1_cuda_smoke") / str(time.time_ns())
    for model_name in ("stfednet_gated", "gru_attention"):
        seed_everything(42, 1)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        measurements = {}
        original = dp_server.client_dp_update

        def measured_update(*args, **kwargs):
            client_id = args[1].client_id
            torch.cuda.synchronize()
            start_client = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            measurements[client_id] = time.perf_counter() - start_client
            return result

        with patch.object(dp_server, "client_dp_update", measured_update):
            run = run_dp_federated(cfg, model_name,
                                   run_dir=root / model_name,
                                   prepared_clients=prepared)
        elapsed = time.perf_counter() - started
        latest = load_checkpoint(run / "latest_dp_global.pt")
        best = load_checkpoint(run / "best_dp_global.pt")
        if not latest["smoke"] or not best["smoke"] or latest["test_evaluated"]:
            raise AssertionError("Smoke marker or test isolation metadata is invalid")
        if latest["completed_round"] != 1:
            raise AssertionError("CUDA smoke should complete one short communication round")
        clients = {}
        # Strict-load both artifact types without evaluating another split.
        receiver = build_model(model_name, 32, 0.1).to("cuda")
        receiver.load_state_dict(best["model_state_dict"], strict=True)
        receiver.load_state_dict(latest["global_model_state_dict"], strict=True)
        for client_id in cfg["clients"]:
            account = latest["accountant_states"][client_id]
            sampler = json.loads(latest["round_history"][0]["client_sampling_json"])[client_id]
            if account["steps"] != 1:
                raise AssertionError("Each client must account exactly one actual DP step")
            if sampler["method_counts"].get("torch_func") != 1:
                raise AssertionError("CUDA path did not use the explicitly recorded torch.func route")
            if sampler["max_gradient_stack_bytes"] <= 0:
                raise AssertionError("torch.func gradient stack size was not recorded")
            clients[client_id] = {
                "nodes": len(prepared[client_id].grid.node_ids),
                "actual_poisson_batch_size": sampler["mean_actual_batch_size"],
                "accounted_steps": account["steps"],
                "theoretical_epsilon_after_one_step": account["theoretical_epsilon"],
                "method_counts": sampler["method_counts"],
                "max_gradient_stack_bytes": sampler["max_gradient_stack_bytes"],
                "seconds_one_local_update_including_initialization_checkpoint_and_accounting": measurements[client_id],
                "rough_191_step_local_epoch_seconds_from_single_step": measurements[client_id] * 191,
            }
        summaries.append({
            "model_name": model_name,
            "rounds": latest["completed_round"],
            "smoke": True,
            "gradient_method": latest["gradient_method"],
            "seconds_for_five_client_one_step_round_and_validation": elapsed,
            "cuda_peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "client_records": clients,
            "validation_metrics_visibility": "internal_restricted_non_dp",
            "privacy_claim_allowed": False,
            "test_evaluated": False,
            "strict_best_latest_load": True,
            "epoch_estimate_caveat": "Single draw extrapolation includes one model initialization and disk/accountant overhead per measured draw; not a measured epoch or performance guarantee.",
        })
    result = {
        "phase": "G3.1 short feasibility smoke",
        "smoke": True,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "opacus_version": __import__("opacus").__version__,
        "privacy_mode": "research",
        "noise_source": "ordinary_torch_prng_not_secure",
        "privacy_claim_allowed": False,
        "test_evaluated": False,
        "formal_training": False,
        "clients": list(cfg["clients"]),
        "models": summaries,
    }
    out = Path("results/g3_1/cuda_smoke_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
