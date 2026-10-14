"""Bounded CUDA benchmark for the G3.0 reference gradient loop.

Only the training split is read. This is an engineering timing check and does
not produce a model checkpoint or a privacy claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from src.models.forecasting import build_model
from src.privacy.per_sample import (per_sample_clipped_gradients,
                                    torch_func_clipped_gradients)
from src.training.engine import loader, masked_huber, prepare_data, read_config, select_device


CLIENTS = {"lv_39": 39, "lv_80": 80, "industrial_mvlv": 76}


class _ForbiddenTestSplit:
    def __len__(self):
        raise AssertionError("G3.0 benchmark must not inspect test windows")

    def __getitem__(self, index):
        raise AssertionError("G3.0 benchmark must not inspect test windows")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bytes(model):
    return int(sum(p.numel() * p.element_size() for p in model.parameters()
                   if p.requires_grad))


def measure(model_name, client_id, prepared, config, device, batch_size):
    config = dict(config, batch_size=batch_size, dropout=0.0)
    batches = loader(prepared.train, config, shuffle=False, max_batches=1)
    x, y = next(iter(batches))
    x, y = x.to(device), y.to(device)
    edges = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
    mask = torch.as_tensor(prepared.grid.load_mask, dtype=torch.bool, device=device)
    torch.manual_seed(9042)
    initial = {key: value.detach().clone() for key, value in
               build_model(model_name, config["hidden_dim"], 0.0,
                           config["horizon_hours"]).state_dict().items()}

    def make_model_and_optimizer():
        current = build_model(model_name, config["hidden_dim"], 0.0,
                              config["horizon_hours"]).to(device)
        current.load_state_dict(initial, strict=True)
        current.train()
        opt = torch.optim.AdamW(current.parameters(), lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
        return current, opt

    model, optimizer = make_model_and_optimizer()

    def ordinary_step():
        optimizer.zero_grad(set_to_none=True)
        loss = masked_huber(model(x, edges), y, mask, config["huber_delta"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()

    # Warm up kernels/allocator, then take a median of three short repeats.
    ordinary_step()
    _sync(device)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ordinary_times = []
    for _ in range(3):
        start = time.perf_counter()
        ordinary_step()
        _sync(device)
        ordinary_times.append(time.perf_counter() - start)
    ordinary_seconds = statistics.median(ordinary_times)
    ordinary_peak = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    parameter_bytes = _bytes(model)
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, optimizer = make_model_and_optimizer()
    if device.type == "cuda":
        _sync(device)
        torch.cuda.reset_peak_memory_stats(device)

    def reference_step():
        optimizer.zero_grad(set_to_none=True)
        gradients = per_sample_clipped_gradients(model, x, y, edges, mask, 1.0)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.grad = gradients["gradients"][name]
        optimizer.step()
        return gradients

    result = reference_step()
    _sync(device)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sample_times = []
    for _ in range(3):
        start = time.perf_counter()
        result = reference_step()
        _sync(device)
        sample_times.append(time.perf_counter() - start)
    sample_seconds = statistics.median(sample_times)
    sample_peak = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    max_clipped_norm = max(result["clipped_norms"])
    del model, optimizer, result
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    func_error = None
    func_seconds = None
    func_peak = 0
    func_stack_bytes = None
    try:
        model, optimizer = make_model_and_optimizer()

        def func_step():
            optimizer.zero_grad(set_to_none=True)
            gradients = torch_func_clipped_gradients(model, x, y, edges, mask, 1.0)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.grad = gradients["gradients"][name]
            optimizer.step()
            return gradients

        if device.type == "cuda":
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)
        func_result = func_step()
        _sync(device)
        func_times = []
        for _ in range(3):
            start = time.perf_counter()
            func_result = func_step()
            _sync(device)
            func_times.append(time.perf_counter() - start)
        func_seconds = statistics.median(func_times)
        func_peak = (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        func_stack_bytes = func_result["gradient_stack_bytes"]
    except Exception as error:  # Backend/version-dependent feasibility result.
        func_error = f"{type(error).__name__}: {error}"
    finally:
        if "model" in locals():
            del model
        if "optimizer" in locals():
            del optimizer
        gc.collect()
    records_per_epoch = len(prepared.train)
    per_record = sample_seconds / max(1, x.shape[0])
    return {"model": model_name, "client": client_id, "nodes": prepared.grid.num_nodes,
            "batch_size": int(x.shape[0]), "train_windows": records_per_epoch,
            "ordinary_batch_seconds": ordinary_seconds,
            "per_sample_loop_batch_seconds": sample_seconds,
            "torch_func_batch_seconds": func_seconds,
            "torch_func_error": func_error,
            "slowdown": sample_seconds / max(ordinary_seconds, 1e-12),
            "ordinary_peak_cuda_mb": ordinary_peak / 2**20,
            "per_sample_peak_cuda_mb": sample_peak / 2**20,
            "torch_func_peak_cuda_mb": func_peak / 2**20,
            "one_record_gradient_bytes": parameter_bytes,
            "accumulator_gradient_bytes": parameter_bytes,
            "torch_func_gradient_stack_bytes": func_stack_bytes,
            "max_clipped_norm": max_clipped_norm,
            "rough_dp_loop_epoch_seconds": per_record * records_per_epoch,
            "rough_ordinary_epoch_seconds_at_measured_batch":
                ordinary_seconds * ((records_per_epoch + int(x.shape[0]) - 1) // int(x.shape[0])),
            "rough_torch_func_epoch_seconds_at_measured_batch":
                func_seconds * ((records_per_epoch + int(x.shape[0]) - 1) // int(x.shape[0]))
                if func_seconds is not None else None,
            "rough_epoch_estimate_assumption":
                "repeat one measured training batch; excludes data loading and validation",
            "test_split_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="results/g3_0/benchmark.json")
    args = parser.parse_args()
    device = select_device(args.device)
    config = read_config("configs/g1.yaml")
    rows = []
    for client_id in CLIENTS:
        prepared = prepare_data(config, client_id)
        prepared.test = _ForbiddenTestSplit()
        for model_name in ("gru_attention", "stfednet_gated"):
            rows.append(measure(model_name, client_id, prepared, config, device, args.batch_size))
            print(f"{model_name} {client_id}: ordinary={rows[-1]['ordinary_batch_seconds']:.3f}s "
                  f"per_sample={rows[-1]['per_sample_loop_batch_seconds']:.3f}s "
                  f"peak={rows[-1]['per_sample_peak_cuda_mb']:.1f}MB", flush=True)
    output = {"smoke": True, "device": str(device), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "rows": rows, "test_split_accessed": False}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[G3.0 Benchmark] smoke=true; wrote {path}", flush=True)


if __name__ == "__main__":
    main()
