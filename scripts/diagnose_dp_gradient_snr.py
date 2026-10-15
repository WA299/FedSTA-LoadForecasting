"""Read-only, research-only gradient/noise diagnostics for G3.2.

This script reads only the TRAIN window dataset and frozen G3.1 Best
Checkpoints. It never constructs an optimizer or writes gradients to model
parameters. Its output is non-DP and restricted; do not publish it as a
privacy-protected result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch

from src.fl.common import CLIENT_IDS
from src.data.loader import (StandardScaler, WindowDataset, discover_grids,
                             load_grid)
from src.models.forecasting import build_model
from src.privacy.per_sample import (per_sample_clipped_gradients,
                                    torch_func_clipped_gradients)
from src.training.engine import load_checkpoint, select_device, write_json
from src.fl.dp_server import read_g31_config


DEFAULT_RUNS = {
    "sigma0": Path("results/g3_1/dp_fedavg/stfednet_gated/sigma0p0_seed42_20261015_005322_562400"),
    "sigma0p5": Path("results/g3_1/dp_fedavg/stfednet_gated/sigma0p5_seed42_20261015_154732_627600"),
    "sigma1": Path("results/g3_1/dp_fedavg/stfednet_gated/sigma1p0_seed42_20261015_013727_775600"),
}
MODULE_PREFIXES = {
    "gat_spatial_attention": ("spatial.gat.",),
    "spatial_gate": ("spatial.gate.",),
    "spatial_output_projection": ("spatial.to_load_channel.",),
    "gru": ("temporal.gru.",),
    "temporal_attention": ("attention.",),
    "prediction_head": ("head.",),
}
DIAGNOSTIC_FIELDS = (
    "signal_l2_norm", "noise_l2_norm", "snr", "direction_cosine",
    "noise_std_mean_expected", "noise_std_mean_observed",
)


def fixed_train_indices(train_size: int, batch_size: int = 32,
                        repeats: int = 3) -> np.ndarray:
    """Return deterministic, stratified TRAIN indices shared by all checkpoints.

    Systematic positions spread the finite diagnostic records across the train
    interval instead of taking adjacent overlapping windows. The selected
    indices are never written to the result file.
    """
    if train_size < batch_size * repeats:
        raise ValueError("TRAIN split is too small for the requested diagnostic batches")
    count = batch_size * repeats
    indices = np.rint(np.linspace(0, train_size - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError("Could not construct unique fixed TRAIN indices")
    return indices.reshape(repeats, batch_size)


def parameter_groups(model: torch.nn.Module) -> dict[str, tuple[str, ...]]:
    """Partition every Gated ST-FedNet parameter into interpretable modules."""
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups: dict[str, list[str]] = {key: [] for key in MODULE_PREFIXES}
    for name in names:
        matches = [group for group, prefixes in MODULE_PREFIXES.items()
                   if any(name.startswith(prefix) for prefix in prefixes)]
        if len(matches) != 1:
            raise ValueError(f"G3.2 parameter has no unique module group: {name}")
        groups[matches[0]].append(name)
    if any(not values for values in groups.values()):
        raise ValueError("Expected all five Gated ST-FedNet parameter groups")
    if sum(map(len, groups.values())) != len(names):
        raise ValueError("G3.2 parameter groups do not cover the full model")
    return {key: tuple(values) for key, values in groups.items()}


def vector_diagnostics(signal: Mapping[str, torch.Tensor],
                       noise: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Compare an already aggregated clipped-gradient mean with its noise."""
    if set(signal) != set(noise) or not signal:
        raise ValueError("Signal and noise parameter keys must match and be non-empty")
    signal_sq = torch.zeros((), dtype=torch.float64, device=next(iter(signal.values())).device)
    noise_sq = torch.zeros_like(signal_sq)
    dot = torch.zeros_like(signal_sq)
    count = 0
    noise_parts = []
    for name, signal_tensor in signal.items():
        noise_tensor = noise[name]
        if signal_tensor.shape != noise_tensor.shape:
            raise ValueError("Signal/noise parameter shape mismatch for " + name)
        s = signal_tensor.detach().to(torch.float64)
        n = noise_tensor.detach().to(torch.float64)
        signal_sq += s.square().sum()
        noise_sq += n.square().sum()
        dot += (s * (s + n)).sum()
        noise_parts.append(noise_tensor.detach().reshape(-1))
        count += noise_tensor.numel()
    signal_norm = float(signal_sq.sqrt().cpu())
    noise_norm = float(noise_sq.sqrt().cpu())
    perturbed_sq = signal_sq + noise_sq + 2.0 * (dot - signal_sq)
    perturbed_norm = float(perturbed_sq.clamp_min(0).sqrt().cpu())
    cosine = (float(dot.cpu()) / (signal_norm * perturbed_norm)
              if signal_norm > 0 and perturbed_norm > 0 else None)
    joined_noise = torch.cat(noise_parts).to(torch.float64)
    observed_std = float(joined_noise.std(unbiased=False).cpu()) if count else 0.0
    return {
        "signal_l2_norm": signal_norm,
        "noise_l2_norm": noise_norm,
        "snr": (signal_norm / noise_norm if noise_norm > 0 else None),
        "direction_cosine": cosine,
        "noise_std_mean_observed": observed_std,
    }


def _parameter_subset(values: Mapping[str, torch.Tensor], names: Sequence[str]):
    return {name: values[name] for name in names}


def _prepare_train_only_clients(config):
    """Prepare only standardized training rows/windows; never build val/test views."""
    grid_paths = dict(discover_grids(config["data_root"]))
    clients = {}
    for client_id in CLIENT_IDS:
        grid = load_grid(grid_paths[client_id], client_id)
        train_end = int(grid.timestamps.size * float(config["train_ratio"]))
        scaler = StandardScaler.fit(grid.load_matrix, grid.timestamps, train_end)
        train_values = scaler.transform(grid.load_matrix[:train_end])
        train_starts = np.arange(
            max(0, train_end - int(config["input_hours"]) - int(config["horizon_hours"]) + 1),
            dtype=np.int64)
        train = WindowDataset(
            values=train_values,
            timestamps=grid.timestamps[:train_end],
            starts=train_starts,
            input_hours=int(config["input_hours"]),
            horizon_hours=int(config["horizon_hours"]),
            load_mask=grid.load_mask,
        )
        expected_windows = int(config.get("expected_train_records", 6084))
        if len(train) != expected_windows:
            raise ValueError(f"Unexpected TRAIN window count for {client_id}: {len(train)}")
        clients[client_id] = SimpleNamespace(grid=grid, train=train,
                                              scaler=scaler)
    return clients


def _mean_std(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {"n": len(finite), "mean": (float(np.mean(finite)) if finite else None),
            "std": (float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if finite else None)}


def summarize_repeats(rows: Sequence[Mapping[str, float | int | None]],
                      fields: Sequence[str] = DIAGNOSTIC_FIELDS) -> dict[str, dict[str, float | int | None]]:
    return {field: _mean_std([row.get(field) for row in rows]) for field in fields}


def _checkpoint_spec(label: str, directory: Path, expected_sigma: float):
    checkpoint_path = directory / "best_dp_global.pt"
    config_path = directory / "config.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing formal Best Checkpoint/config for {label}: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    checkpoint = load_checkpoint(checkpoint_path)
    if config.get("model_name") != "stfednet_gated" or checkpoint.get("model_name") != "stfednet_gated":
        raise ValueError(f"{label} must be a stfednet_gated run")
    if not math.isclose(float(config.get("noise_multiplier", -1)), expected_sigma, abs_tol=1e-12):
        raise ValueError(f"{label} config noise multiplier does not match its label")
    if not math.isclose(float(checkpoint.get("noise_multiplier", -1)), expected_sigma, abs_tol=1e-12):
        raise ValueError(f"{label} checkpoint noise multiplier does not match its label")
    if config.get("smoke") or checkpoint.get("smoke"):
        raise ValueError(f"Smoke Checkpoint is not allowed: {label}")
    if config.get("test_evaluated") or checkpoint.get("test_evaluated"):
        raise ValueError(f"Checkpoint reports test evaluation: {label}")
    if config.get("privacy_claim_allowed") or checkpoint.get("privacy_claim_allowed"):
        raise ValueError(f"Expected research-only non-claim checkpoint: {label}")
    required = {"rounds": 20, "local_epochs": 1, "seed": 42,
                "input_hours": 48, "horizon_hours": 1, "hidden_dim": 32,
                "batch_size": 32, "expected_batch_size": 32,
                "expected_train_records": 6084, "max_grad_norm": 1.0,
                "dropout": 0.1}
    for key, expected in required.items():
        if not math.isclose(float(config.get(key, float("nan"))), float(expected),
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} violates the shared protocol: {key}")
    for key, expected in (("train_ratio", 0.70), ("val_ratio", 0.15)):
        if not math.isclose(float(config.get(key, float("nan"))), expected,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} violates the shared chronological split: {key}")
    stored_config = checkpoint.get("config", {})
    for key in required.keys() | {"train_ratio", "val_ratio"}:
        if key in stored_config and key in config and stored_config[key] != config[key]:
            raise ValueError(f"{label} Best Checkpoint/config.json mismatch: {key}")
    if checkpoint.get("client_ids") != list(CLIENT_IDS):
        raise ValueError(f"{label} does not contain the canonical five clients")
    signatures = checkpoint.get("data_signatures")
    if not isinstance(signatures, dict) or tuple(signatures) != CLIENT_IDS:
        # Dict insertion order is written canonically by G3.1; compare sets too
        # for strict compatibility with otherwise equivalent checkpoint files.
        if not isinstance(signatures, dict) or set(signatures) != set(CLIENT_IDS):
            raise ValueError(f"{label} is missing the five client data signatures")
    return {
        "label": label, "directory": directory, "sigma": expected_sigma,
        "checkpoint": checkpoint, "checkpoint_path": checkpoint_path,
        "config": config,
    }


def diagnose(run_directories: Mapping[str, Path], output: Path,
             device_name: str = "auto", batch_size: int = 32, repeats: int = 3,
             clip: float = 1.0, sigmas: Sequence[float] = (0.5, 1.0),
             gradient_method: str = "auto") -> dict:
    """Run restricted diagnostics on frozen G3.1 Best states; never updates weights."""
    started = time.perf_counter()
    if batch_size != 32:
        raise ValueError("G3.2 diagnostics must use the G3.1 expected batch size 32")
    if clip != 1.0:
        raise ValueError("G3.2 diagnostics must preserve C=1.0")
    if repeats < 2:
        raise ValueError("At least two independent fixed TRAIN batches are required")
    runs = [
        _checkpoint_spec("sigma0", Path(run_directories["sigma0"]), 0.0),
        _checkpoint_spec("sigma0p5", Path(run_directories["sigma0p5"]), 0.5),
        _checkpoint_spec("sigma1", Path(run_directories["sigma1"]), 1.0),
    ]
    reference_signatures = runs[0]["checkpoint"]["data_signatures"]
    for run in runs[1:]:
        if run["checkpoint"]["data_signatures"] != reference_signatures:
            raise ValueError("The three Checkpoints do not share the same data signatures")

    # The G3.1 loader requires an explicit training sigma even though this
    # read-only diagnostic performs no DP update; pin it to the sigma=1 model
    # configuration as a neutral protocol reference.
    config = read_g31_config(overrides={"noise_multiplier": 1.0,
                                        "device": device_name})
    device = select_device(device_name)
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    prepared_clients = _prepare_train_only_clients(config)
    method = gradient_method
    if method == "auto":
        method = "torch_func" if device.type == "cuda" else "loop"
    if method not in {"loop", "torch_func"}:
        raise ValueError("gradient_method must be auto, loop, or torch_func")
    if method == "torch_func" and device.type != "cuda":
        raise ValueError("torch_func diagnostics are only enabled on the validated CUDA path")

    result = {
        "schema_version": 1,
        "phase": "G3.2",
        "visibility": "internal_restricted_non_dp",
        "privacy_claim_allowed": False,
        "smoke": True,
        "diagnostic_only": True,
        "test_evaluated": False,
        "parameter_updates": False,
        "model_name": "stfednet_gated",
        "device": str(device),
        "gradient_method": method,
        "record_unit": "one complete standardized TRAIN graph window across all nodes and 48 history steps plus 1-hour target",
        "preprocessing_condition": "fixed G3.1 train-fitted scaler, graph topology and load mask",
        "batch_size": batch_size,
        "repeat_batches_per_client": repeats,
        "clip_threshold": clip,
        "noise_multipliers_tested": [float(s) for s in sigmas],
        "noise_source": "ordinary_torch_prng_research_only; generator state not saved",
        "noise_std_mean_expected": {str(float(s)): float(s) * clip / batch_size for s in sigmas},
        "checkpoints": [],
        "notes": [
            "All gradient SNR numerators are the full-model norm of the mean clipped gradient after per-window clipping and fixed division by 32.",
            "Per-window unclipped norms are reported only for clipping statistics and are never used as SNR numerators.",
            "The diagnostic constructs standardized TRAIN rows and TRAIN windows only; it never constructs or indexes a validation/test window view.",
            "Reported measurements are descriptive finite-batch diagnostics, not DP outputs or causal proof of training failure.",
        ],
    }
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    for run in runs:
        payload = run["checkpoint"]
        run_entry = {
            "run_label": run["label"], "training_noise_multiplier": run["sigma"],
            "best_round": payload.get("best_round", payload.get("round")),
            "best_validation_wape_percent": payload.get("best_validation_wape"),
            "clients": {},
        }
        model = build_model("stfednet_gated", config["hidden_dim"], config["dropout"],
                            config["horizon_hours"]).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        groups = parameter_groups(model)
        original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.train()
        for client_id in CLIENT_IDS:
            prepared = prepared_clients[client_id]
            indices_by_batch = fixed_train_indices(len(prepared.train), batch_size, repeats)
            edges = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
            load_mask = torch.as_tensor(prepared.grid.load_mask, dtype=torch.bool, device=device)
            batch_rows = []
            for batch_id, indices in enumerate(indices_by_batch, start=1):
                records = [prepared.train[int(index)] for index in indices]
                x = torch.stack([torch.as_tensor(row[0], dtype=torch.float32) for row in records]).to(device)
                y = torch.stack([torch.as_tensor(row[1], dtype=torch.float32) for row in records]).to(device)
                if method == "torch_func":
                    measured = torch_func_clipped_gradients(
                        model, x, y, edges, load_mask, clip, noise_multiplier=0.0,
                        delta=float(config["huber_delta"]), normalizer=float(batch_size))
                else:
                    measured = per_sample_clipped_gradients(
                        model, x, y, edges, load_mask, clip, noise_multiplier=0.0,
                        delta=float(config["huber_delta"]), normalizer=float(batch_size))
                signal = measured["gradients"]
                preclip = [float(value) for value in measured["pre_clip_norms"]]
                batch_entry = {
                    "batch_ordinal": batch_id,
                    "train_record_count": len(preclip),
                    "clip_fraction": sum(value > clip for value in preclip) / len(preclip),
                    "mean_preclip_per_window_l2_norm": float(np.mean(preclip)),
                    "std_preclip_per_window_l2_norm": float(np.std(preclip, ddof=1)) if len(preclip) > 1 else 0.0,
                    "mean_clipped_per_window_l2_norm": float(np.mean(measured["clipped_norms"])),
                    "aggregate_gradient_mean_l2_norm": math.sqrt(sum(
                        float(value.detach().double().square().sum().cpu()) for value in signal.values())),
                    "noise_scenarios": {},
                }
                for sigma in sigmas:
                    noise_std = float(sigma) * clip / batch_size
                    noise_tensors = {
                        name: torch.randn(value.shape, dtype=value.dtype, device=value.device) * noise_std
                        for name, value in signal.items()
                    }
                    full = vector_diagnostics(signal, noise_tensors)
                    full["noise_std_mean_expected"] = noise_std
                    module_metrics = {}
                    for module_name, parameter_names in groups.items():
                        module_values = vector_diagnostics(
                            _parameter_subset(signal, parameter_names),
                            _parameter_subset(noise_tensors, parameter_names))
                        module_values["noise_std_mean_expected"] = noise_std
                        module_metrics[module_name] = module_values
                    batch_entry["noise_scenarios"][str(float(sigma))] = {
                        "model": full,
                        "modules": module_metrics,
                    }
                batch_rows.append(batch_entry)
            clipping_rows = {
                "clip_fraction": [row["clip_fraction"] for row in batch_rows],
                "mean_preclip_per_window_l2_norm": [row["mean_preclip_per_window_l2_norm"] for row in batch_rows],
                "mean_clipped_per_window_l2_norm": [row["mean_clipped_per_window_l2_norm"] for row in batch_rows],
                "aggregate_gradient_mean_l2_norm": [row["aggregate_gradient_mean_l2_norm"] for row in batch_rows],
            }
            scenario_summaries = {}
            for sigma in sigmas:
                key = str(float(sigma))
                scenario_summaries[key] = {
                    "model": summarize_repeats([row["noise_scenarios"][key]["model"] for row in batch_rows]),
                    "modules": {
                        module_name: summarize_repeats([
                            row["noise_scenarios"][key]["modules"][module_name]
                            for row in batch_rows])
                        for module_name in groups
                    },
                }
            run_entry["clients"][client_id] = {
                "num_nodes": int(prepared.grid.load_matrix.shape[1]),
                "train_window_count": len(prepared.train),
                "clipping_summary_across_repeated_batches": {
                    key: _mean_std(values) for key, values in clipping_rows.items()},
                "noise_scenario_summary_across_repeated_batches": scenario_summaries,
                "repeated_batch_measurements": batch_rows,
            }
        for name, value in model.state_dict().items():
            if not torch.equal(value, original_state[name]):
                raise RuntimeError("Read-only diagnostic changed model state: " + name)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Read-only diagnostic populated optimizer gradients")
        run_entry["model_state_unchanged"] = True
        result["checkpoints"].append(run_entry)
        del model
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    cross_checkpoint = {}
    for sigma in sigmas:
        sigma_key = str(float(sigma))
        per_checkpoint = {}
        for checkpoint_result in result["checkpoints"]:
            snr_values = [
                measurement["noise_scenarios"][sigma_key]["model"]["snr"]
                for client_result in checkpoint_result["clients"].values()
                for measurement in client_result["repeated_batch_measurements"]
            ]
            per_checkpoint[checkpoint_result["run_label"]] = _mean_std(snr_values)
        cross_checkpoint[sigma_key] = per_checkpoint
    noisy_means_below_one = all(
        cross_checkpoint[str(float(sigma))][label]["mean"] < 1.0
        for sigma in sigmas for label in ("sigma0p5", "sigma1"))
    control_stronger = all(
        cross_checkpoint[str(float(sigma))]["sigma0"]["mean"] >
        cross_checkpoint[str(float(sigma))][label]["mean"]
        for sigma in sigmas for label in ("sigma0p5", "sigma1"))
    result["restricted_interpretation"] = {
        "mean_full_model_snr_by_checkpoint_and_simulated_sigma": cross_checkpoint,
        "noise_dominance_hypothesis": (
            "supported_by_current_best-checkpoint_gradient_snapshots"
            if noisy_means_below_one and control_stronger
            else "not_decisively_supported_by_current_best-checkpoint_gradient_snapshots"),
        "causal_status": "not_established; final-checkpoint gradients do not measure accumulated training trajectory or AdamW interaction",
        "next_single_variable_experiment": (
            "try expected_batch_size=64 before changing optimizer; keep all other protocol fields fixed and explicitly account for the derived change in q and nominal updates per round"),
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        result["cuda_device_name"] = torch.cuda.get_device_name(device)
    result["runtime_seconds"] = time.perf_counter() - started
    result["output_policy"] = "Local-only restricted JSON. Contains aggregate non-DP training diagnostics; do not upload to public source control or release externally."
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma0-run", type=Path, default=DEFAULT_RUNS["sigma0"])
    parser.add_argument("--sigma05-run", type=Path, default=DEFAULT_RUNS["sigma0p5"])
    parser.add_argument("--sigma1-run", type=Path, default=DEFAULT_RUNS["sigma1"])
    parser.add_argument("--output", type=Path, default=Path("results/g3_2/gradient_diagnostics.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--gradient-method", choices=("auto", "loop", "torch_func"), default="auto")
    args = parser.parse_args()
    runs = {"sigma0": args.sigma0_run, "sigma0p5": args.sigma05_run, "sigma1": args.sigma1_run}
    result = diagnose(runs, args.output, args.device, args.batch_size, args.repeats,
                      args.clip, (0.5, 1.0), args.gradient_method)
    print("[G3.2] Restricted read-only diagnostics complete; smoke=true, test_evaluated=false")
    print("[G3.2] Internal result: " + str(args.output))
    print("[G3.2] CUDA peak allocated: " + str(result.get("cuda_peak_allocated_bytes", "not_applicable")))


if __name__ == "__main__":
    main()
