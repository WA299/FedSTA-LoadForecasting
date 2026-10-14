"""Bounded CUDA checkpoint benchmark against the immutable G3.1 baseline.

Only training windows and limited validation batches are accessed. Outputs are
restricted research diagnostics; no formal performance/privacy claim is made.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from unittest.mock import patch

import torch

from src.fl import dp_server
from src.privacy import training
from src.privacy.accountant import ClientRDPAccountant
from src.training.engine import load_checkpoint, write_json

BASELINE = "3bd0589e02490a30957c997ef9154effe87e00df"


class ForbiddenTest:
    def __len__(self):
        raise AssertionError("Checkpoint benchmark accessed test data")

    def __getitem__(self, index):
        raise AssertionError("Checkpoint benchmark accessed test data")


def baseline_modules():
    """Load tracked old code without checking out or modifying user files."""
    modules = []
    for filename in ("src/privacy/training.py", "src/fl/dp_server.py"):
        source = subprocess.check_output(["git", "show", BASELINE + ":" + filename],
                                         encoding="utf-8")
        module = types.ModuleType("checkpoint_baseline_" + Path(filename).stem)
        exec(compile(source, BASELINE + ":" + filename, "exec"), module.__dict__)
        modules.append(module)
    modules[1].client_dp_update = modules[0].client_dp_update
    return modules


class StageProfiler:
    """Synchronize CUDA for stage timing; exclude nested category durations.

    Instrumented runs are separate from uninstrumented wall-time comparisons.
    No wrappers draw randomness or modify tensors/optimizer/accountant states.
    """

    def __init__(self):
        self.seconds, self.calls, self.stack = {}, {}, []

    def wrap(self, category, function):
        def measured(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            frame = [0.0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                self.stack.pop()
                if self.stack:
                    self.stack[-1][0] += elapsed
                self.seconds[category] = self.seconds.get(category, 0.0) + elapsed - frame[0]
                self.calls[category] = self.calls.get(category, 0) + 1
        return measured

    def attach(self, stack, client_module, server_module):
        for name in ("poisson_indices", "_stack_records"):
            stack.enter_context(patch.object(client_module, name,
                self.wrap("poisson_and_batch", getattr(client_module, name))))
        for name in ("per_sample_clipped_gradients", "torch_func_clipped_gradients"):
            stack.enter_context(patch.object(client_module, name,
                self.wrap("per_window_gradients_clipping_noise", getattr(client_module, name))))
        stack.enter_context(patch.object(torch.optim.AdamW, "step",
            self.wrap("adamw_update", torch.optim.AdamW.step)))
        for module in (client_module, server_module):
            stack.enter_context(patch.object(module, "_cpu_copy",
                self.wrap("model_optimizer_state_copy", module._cpu_copy)))
        stack.enter_context(patch.object(server_module, "_capture_random_state",
            self.wrap("rng_state_copy", server_module._capture_random_state)))
        stack.enter_context(patch.object(server_module, "atomic_save",
            self.wrap("checkpoint_write", server_module.atomic_save)))
        stack.enter_context(patch.object(server_module, "write_json",
            self.wrap("json_write", server_module.write_json)))
        stack.enter_context(patch.object(ClientRDPAccountant, "state_dict",
            self.wrap("rdp_state_read", ClientRDPAccountant.state_dict)))
        stack.enter_context(patch.object(ClientRDPAccountant, "epsilon",
            self.wrap("rdp_epsilon_query", ClientRDPAccountant.epsilon)))


def tensor_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(
            tensor_tree_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(
            tensor_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--output-root", default="results/g3_1/checkpoint_benchmark")
    parser.add_argument("--report", default="results/g3_1_checkpoint_optimization.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison")
    if not 1 <= args.steps <= 40 or not 1 <= args.profile_steps <= 5:
        raise ValueError("Only short development budgets are allowed")
    root = Path(args.output_root) / time.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    config = dp_server.read_g31_config(overrides={
        "noise_multiplier": 1.0, "seed": 42, "rounds": 1, "device": "cuda",
        "max_dp_steps_per_epoch": args.steps, "max_val_batches": 1})
    clients = dp_server.prepare_clients(config)
    for prepared in clients.values():
        prepared.test = ForbiddenTest()
    old_client, old_server = baseline_modules()
    # The old immutable client/server uses the old uncached epsilon method.
    old_account_source = subprocess.check_output(
        ["git", "show", BASELINE + ":src/privacy/accountant.py"], encoding="utf-8")
    old_account = types.ModuleType("checkpoint_baseline_accountant")
    exec(compile(old_account_source, BASELINE + ":accountant.py", "exec"), old_account.__dict__)
    report = {"smoke": True, "test_evaluated": False, "privacy_claim_allowed": False,
              "visibility": "internal_restricted_non_dp", "baseline_commit": BASELINE,
              "python_version": sys.version.split()[0], "torch_version": torch.__version__,
              "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "client_count": 5, "input_hours": config["input_hours"],
              "horizon_hours": config["horizon_hours"],
              "expected_batch_size": config["expected_batch_size"],
              "rounds": 1, "steps_per_client": args.steps,
              "seed": 42, "sigma": 1.0, "models": {}}
    for model_name in ("stfednet_gated", "gru_attention"):
        runs, checkpoints = {}, {}
        for label, interval, client_module, server_module in (
                ("baseline_every_step", 1, old_client, old_server),
                ("optimized_every_step", 1, training, dp_server),
                ("optimized_every_25", 25, training, dp_server)):
            cfg = dict(config, checkpoint_every_steps=interval)
            writes = []
            save = server_module.atomic_save

            def counted_save(payload, path):
                writes.append(Path(path).name)
                return save(payload, path)

            with ExitStack() as stack:
                stack.enter_context(patch.object(server_module, "atomic_save", counted_save))
                if server_module is old_server:
                    stack.enter_context(patch.object(server_module, "ClientRDPAccountant",
                                                     old_account.ClientRDPAccountant))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                run = server_module.run_dp_federated(cfg, model_name,
                    run_dir=root / model_name / label, prepared_clients=clients)
                torch.cuda.synchronize()
                wall = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated()
            checkpoints[label] = load_checkpoint(run / "latest_dp_global.pt")
            runs[label] = {"wall_seconds": wall, "checkpoint_writes": len(writes),
                           "latest_writes": writes.count("latest_dp_global.pt"),
                           "peak_allocated_mb": peak / 1024**2}
        def equality(first, second):
            return {"model_parameters": tensor_tree_equal(first["global_model_state_dict"], second["global_model_state_dict"]),
                    "best_parameters": tensor_tree_equal(first["best_model_state_dict"], second["best_model_state_dict"]),
                    "accountants": tensor_tree_equal(first["accountant_states"], second["accountant_states"]),
                    "rng_states": tensor_tree_equal(first["rng_state"], second["rng_state"]),
                    "best_validation": first["best_validation"] == second["best_validation"]}
        equality_checks = {"old_vs_interval1": equality(checkpoints["baseline_every_step"],
                                                          checkpoints["optimized_every_step"]),
                           "old_vs_interval25": equality(checkpoints["baseline_every_step"],
                                                          checkpoints["optimized_every_25"])}
        for label, values in equality_checks.items():
            if not all(values.values()):
                raise AssertionError("Checkpoint optimization changed training " + label + ": " + str(values))
        profiles = {}
        for label, interval, client_module, server_module in (
                ("baseline_every_step", 1, old_client, old_server),
                ("optimized_every_25", 25, training, dp_server)):
            profiler = StageProfiler()
            cfg = dict(config, checkpoint_every_steps=interval,
                       max_dp_steps_per_epoch=args.profile_steps)
            with ExitStack() as stack:
                # Profile old RDP methods as well, including repeated epsilon queries.
                if server_module is old_server:
                    stack.enter_context(patch.object(server_module, "ClientRDPAccountant",
                                                     old_account.ClientRDPAccountant))
                    for name, category in (("state_dict", "rdp_state_read"), ("epsilon", "rdp_epsilon_query")):
                        stack.enter_context(patch.object(old_account.ClientRDPAccountant, name,
                            profiler.wrap(category, getattr(old_account.ClientRDPAccountant, name))))
                profiler.attach(stack, client_module, server_module)
                server_module.run_dp_federated(cfg, model_name,
                    run_dir=root / model_name / (label + "_profile"), prepared_clients=clients)
            profiles[label] = {"seconds": profiler.seconds, "calls": profiler.calls,
                               "steps_per_client": args.profile_steps,
                               "note": "CUDA-synchronized exclusive stage timing; separate from wall run"}
            if server_module is dp_server:
                profile_dir = root / model_name / (label + "_profile")
                profiles[label]["native_accounting_timings"] = {
                    key: value for key, value in json.loads(
                        (profile_dir / "checkpoint_performance.json").read_text()).items()
                    if key in {"accountant_state_read_seconds", "epsilon_calculation_seconds"}}
        report["models"][model_name] = {"runs": runs, "exact_equality": equality_checks,
            "speedup_baseline_to_25": runs["baseline_every_step"]["wall_seconds"] / runs["optimized_every_25"]["wall_seconds"],
            "interval1_wall_ratio_to_baseline": runs["optimized_every_step"]["wall_seconds"] / runs["baseline_every_step"]["wall_seconds"],
            "profiles": profiles}
        write_json(root / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("Report: " + str(root / "summary.json"), flush=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print("Committed summary: " + str(args.report), flush=True)


if __name__ == "__main__":
    main()
