"""Isolated client-side window-level DP-SGD update used by G3.1."""

from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np
import torch

from src.models.forecasting import build_model
from src.privacy.per_sample import (per_sample_clipped_gradients,
                                    torch_func_clipped_gradients)
from src.privacy.poisson import nominal_steps_per_epoch, poisson_indices


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _stack_records(dataset, indices, device):
    records = [dataset[int(index)] for index in indices]
    if not records:
        nodes = int(dataset.num_nodes)
        return (torch.empty((0, dataset.input_hours, nodes, 1), dtype=torch.float32, device=device),
                torch.empty((0, dataset.horizon_hours, nodes), dtype=torch.float32, device=device))
    x = torch.stack([torch.as_tensor(row[0], dtype=torch.float32) for row in records]).to(device)
    y = torch.stack([torch.as_tensor(row[1], dtype=torch.float32) for row in records]).to(device)
    return x, y


def client_dp_update(global_state, prepared, config, model_name, device,
                     accountant, sampling_generator, noise_generator,
                     gradient_method="loop", max_steps_per_epoch=None,
                     resume_progress=None, on_step=None):
    """Run client-local Poisson DP-SGD and return its FedAvg model update.

    Each record is one complete graph window. The clipped gradient is summed,
    Gaussian noise of std ``sigma*C`` is added to that sum, and it is divided
    by the fixed expectation ``q*N``. No ordinary batch-gradient clipping is
    performed here.
    """
    if gradient_method not in {"loop", "torch_func"}:
        raise ValueError("gradient_method must be loop or torch_func")
    dataset = prepared.train
    record_count = len(dataset)
    expected = int(config["expected_batch_size"])
    if record_count <= 0 or expected > record_count:
        raise ValueError("Client training split must be non-empty and at least expected batch size")
    q = expected / record_count
    if not math.isclose(q, accountant.sample_rate, rel_tol=0, abs_tol=1e-15):
        raise ValueError("Client sampler q does not match its RDP accountant")
    steps_per_epoch = nominal_steps_per_epoch(record_count, expected)
    epoch_count = int(config["local_epochs"])
    steps_this_epoch = (steps_per_epoch if max_steps_per_epoch is None
                        else min(steps_per_epoch, int(max_steps_per_epoch)))
    if steps_this_epoch <= 0:
        raise ValueError("At least one Poisson update step is required")
    total_steps = epoch_count * steps_this_epoch

    model = build_model(model_name, config["hidden_dim"], config["dropout"],
                        config["horizon_hours"]).to(device)
    model.load_state_dict(global_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    next_step = 0
    selected_total = 0
    empty_steps = 0
    batch_sizes = []
    methods = Counter()
    loss_total, clipped_records, gradient_norm_total, gradient_record_count = 0.0, 0, 0.0, 0
    max_gradient_stack_bytes = 0
    if resume_progress is not None:
        if resume_progress.get("client_id") != prepared.client_id:
            raise ValueError("In-progress DP client does not match the configured client")
        model.load_state_dict(resume_progress["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_progress["optimizer_state_dict"])
        next_step = int(resume_progress["next_step"])
        selected_total = int(resume_progress.get("selected_records", 0))
        empty_steps = int(resume_progress.get("empty_steps", 0))
        batch_sizes = list(resume_progress.get("batch_sizes", []))
        methods.update(resume_progress.get("method_counts", {}))
        loss_total = float(resume_progress.get("loss_total", 0.0))
        clipped_records = int(resume_progress.get("clipped_records", 0))
        gradient_norm_total = float(resume_progress.get("gradient_norm_total", 0.0))
        gradient_record_count = int(resume_progress.get("gradient_record_count", 0))
        max_gradient_stack_bytes = int(resume_progress.get("max_gradient_stack_bytes", 0))
        if not 0 <= next_step <= total_steps:
            raise ValueError("Invalid in-progress DP step index")
        # Model construction above consumes the framework PRNG. Undo that
        # extra consumption when resuming a partially trained client so its
        # Dropout stream continues exactly at the committed step boundary.
        torch.set_rng_state(resume_progress["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_progress["cuda_rng_state"])

    edge_index = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
    load_mask = torch.as_tensor(prepared.grid.load_mask, dtype=torch.bool, device=device)
    sigma, clip = float(config["noise_multiplier"]), float(config["max_grad_norm"])
    fixed_normalizer = q * record_count
    model.train()
    for step_index in range(next_step, total_steps):
        indices = poisson_indices(record_count, q, sampling_generator)
        x, y = _stack_records(dataset, indices, device)
        actual = int(indices.numel())
        if actual == 0:
            # No per-record transform is required, but the normal DP mechanism
            # still draws Gaussian noise and performs an AdamW update.
            result = per_sample_clipped_gradients(
                model, x, y, edge_index, load_mask, clip, sigma,
                delta=float(config["huber_delta"]), normalizer=fixed_normalizer,
                generator=noise_generator)
            used_method = "empty_noise_only"
            empty_steps += 1
        elif gradient_method == "loop":
            result = per_sample_clipped_gradients(
                model, x, y, edge_index, load_mask, clip, sigma,
                delta=float(config["huber_delta"]), normalizer=fixed_normalizer,
                generator=noise_generator)
            used_method = "loop"
        else:
            # No implicit fallback: unsupported torch.func operators propagate
            # an error, and the selected implementation is recorded in outputs.
            result = torch_func_clipped_gradients(
                model, x, y, edge_index, load_mask, clip, sigma,
                delta=float(config["huber_delta"]), normalizer=fixed_normalizer,
                generator=noise_generator)
            used_method = "torch_func"
        optimizer.zero_grad(set_to_none=True)
        parameters = dict(model.named_parameters())
        for name, parameter in parameters.items():
            if parameter.requires_grad:
                if not torch.isfinite(result["gradients"][name]).all():
                    raise ValueError("Nonfinite per-window DP gradient for " + prepared.client_id + ": " + name)
                parameter.grad = result["gradients"][name].detach().clone()
        optimizer.step()
        accountant.step()
        selected_total += actual
        batch_sizes.append(actual)
        methods[used_method] += 1
        record_norms = [float(value) for value in result.get("pre_clip_norms", [])]
        loss_values = [float(value) for value in result.get("losses", [])]
        loss_total += sum(loss_values)
        gradient_norm_total += sum(record_norms)
        gradient_record_count += len(record_norms)
        clipped_records += sum(value > clip for value in record_norms)
        max_gradient_stack_bytes = max(max_gradient_stack_bytes,
                                       int(result.get("gradient_stack_bytes", 0)))
        completed = step_index + 1
        progress = {
            "client_id": prepared.client_id,
            "model_state_dict": _cpu_copy(model.state_dict()),
            "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
            "next_step": completed,
            "selected_records": selected_total,
            "empty_steps": empty_steps,
            "batch_sizes": list(batch_sizes),
            "method_counts": dict(methods),
            "loss_total": loss_total,
            "clipped_records": clipped_records,
            "gradient_norm_total": gradient_norm_total,
            "gradient_record_count": gradient_record_count,
            "max_gradient_stack_bytes": max_gradient_stack_bytes,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        if on_step is not None:
            on_step(progress, accountant.state_dict())

    return {
        "state_dict": _cpu_copy(model.state_dict()),
        # FedAvg weight uses unique available training windows, not Poisson draw
        # count (which is random and may visit a window more than once).
        "sample_count": record_count,
        "diagnostics": {
            "client_id": prepared.client_id,
            "expected_batch_size": expected,
            "sample_rate": q,
            "nominal_steps_per_epoch": steps_per_epoch,
            "actual_dp_steps": total_steps,
            "selected_record_draws": selected_total,
            "mean_actual_batch_size": (float(np.mean(batch_sizes)) if batch_sizes else 0.0),
            "empty_steps": empty_steps,
            "method_counts": dict(methods),
            "max_gradient_stack_bytes": max_gradient_stack_bytes,
            "mean_huber_loss_per_selected_record": (loss_total / selected_total
                                                       if selected_total else None),
            "mean_preclip_gradient_norm_per_selected_record": (
                gradient_norm_total / gradient_record_count if gradient_record_count else None),
            "clipped_record_fraction": (clipped_records / gradient_record_count
                                         if gradient_record_count else None),
            "theoretical_epsilon": accountant.epsilon(),
            "epsilon_claim_allowed": False,
            "diagnostics_visibility": "internal_restricted_non_dp",
        },
    }
