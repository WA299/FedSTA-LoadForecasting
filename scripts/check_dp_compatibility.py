"""Check G3.0 per-sample-gradient feasibility without training a DP model.

The command is intentionally isolated from the G2 training entry points. It
checks Opacus validation/fixes, probes torch.func on CPU and CUDA, validates
the explicit per-window reference implementation, and writes a compact JSON
report. No validation or test dataset is accessed by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import torch

from src.models.forecasting import build_model
from src.privacy.per_sample import masked_huber_per_sample, per_sample_clipped_gradients


def _edge(nodes: int, device: torch.device) -> torch.Tensor:
    if nodes < 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    source = torch.arange(max(0, nodes - 1), device=device)
    target = source + 1
    return torch.stack((torch.cat((source, target)), torch.cat((target, source))))


def _synthetic_probe(model_name: str, nodes: int, device: torch.device) -> dict:
    torch.manual_seed(1701 + nodes)
    # GRU backward on cuDNN requires training mode even when dropout is zero.
    model = build_model(model_name, hidden_dim=32, dropout=0.0).to(device).train()
    x = torch.randn(2, 48, nodes, 1, device=device)
    y = torch.randn(2, 1, nodes, device=device)
    mask = torch.ones(nodes, dtype=torch.bool, device=device)
    edge = _edge(nodes, device)
    result = per_sample_clipped_gradients(model, x, y, edge, mask, 1.0)
    return {"model": model_name, "nodes": nodes, "device": str(device),
            "record_count": result["record_count"],
            "loss_mean": float(sum(result["losses"]) / len(result["losses"])),
            "max_clipped_norm": max(result["clipped_norms"]),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters())}


def _torch_func_probe(model_name: str, device: torch.device) -> dict:
    from torch.func import functional_call, grad, vmap

    nodes = 3
    model = build_model(model_name, hidden_dim=8, dropout=0.1).to(device).train()
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    x = torch.randn(2, 48, nodes, 1, device=device)
    y = torch.randn(2, 1, nodes, device=device)
    edge = _edge(nodes, device)
    mask = torch.ones(nodes, dtype=torch.bool, device=device)

    def loss_fn(params, sample_x, sample_y):
        prediction = functional_call(model, (params, buffers),
                                     (sample_x.unsqueeze(0), edge))
        return torch.nn.functional.huber_loss(
            prediction[..., mask], sample_y.unsqueeze(0)[..., mask])

    result = {"model": model_name, "device": str(device),
              "single_grad": False, "batched_grad": False, "error": None}
    try:
        grad(loss_fn)(parameters, x[0], y[0])
        result["single_grad"] = True
    except Exception as error:  # pragma: no cover - version/device dependent
        result["single_error"] = f"{type(error).__name__}: {error}"
    try:
        vmap(grad(loss_fn), in_dims=(None, 0, 0), randomness="different")(
            parameters, x, y)
        result["batched_grad"] = True
    except Exception as error:  # pragma: no cover - expected on CPU in this env
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _opacus_probe() -> dict:
    try:
        import opacus
        from opacus.grad_sample import GradSampleModule
        from opacus.optimizers import DPOptimizer
        from opacus.validators import ModuleValidator
    except Exception as error:  # pragma: no cover - dependency check
        return {"installed": False, "error": f"{type(error).__name__}: {error}"}

    output = {"installed": True, "version": opacus.__version__, "models": {}}
    edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    for model_name in ("gru_attention", "stfednet_gated"):
        raw = build_model(model_name, hidden_dim=8, dropout=0.0).train()
        raw_item = {}
        try:
            raw_wrapped = GradSampleModule(raw, batch_first=True,
                                           loss_reduction="mean", strict=True).train()
            raw_x = torch.randn(2, 48, 3, 1)
            raw_y = torch.randn(2, 1, 3)
            torch.nn.functional.huber_loss(raw_wrapped(raw_x, edge), raw_y).backward()
            raw_item["original_grad_sample_backward"] = True
        except Exception as error:
            raw_item["original_grad_sample_backward"] = False
            raw_item["original_grad_sample_error"] = f"{type(error).__name__}: {error}"
        original = build_model(model_name, hidden_dim=8, dropout=0.0).train()
        errors = ModuleValidator.validate(original, strict=False)
        fixed = ModuleValidator.fix(original).train()
        item = {"validator_errors": [str(error) for error in errors],
                "fixed_gru_type": type(fixed.temporal.gru).__name__,
                "strict_state_dict_compatible": False, "grad_sample_backward": False,
                "fixed_output_equivalent": False, **raw_item}
        try:
            fixed.load_state_dict(original.state_dict(), strict=True)
            item["strict_state_dict_compatible"] = True
        except Exception as error:
            item["strict_state_dict_error"] = f"{type(error).__name__}: {error}"
            # The converter adds DPGRU aliases to its state_dict. A non-strict
            # copy is used only to test the converted forward operation; the
            # official model and checkpoint are never modified.
            fixed.load_state_dict(original.state_dict(), strict=False)
        x = torch.randn(2, 48, 3, 1)
        y = torch.randn(2, 1, 3)
        with torch.no_grad():
            item["fixed_output_equivalent"] = bool(torch.allclose(
                original(x, edge), fixed(x, edge), atol=1e-6, rtol=1e-5))
        try:
            wrapped = GradSampleModule(fixed, batch_first=True,
                                       loss_reduction="mean", strict=True).train()
            prediction = wrapped(x, edge)
            torch.nn.functional.huber_loss(prediction, y).backward()
            item["grad_sample_backward"] = True
            item["grad_sample_leading_sizes"] = {
                name: int(parameter.grad_sample.shape[0])
                for name, parameter in wrapped.named_parameters()
                if hasattr(parameter, "grad_sample")}
            optimizer = DPOptimizer(torch.optim.SGD(wrapped.parameters(), lr=0.0),
                                    noise_multiplier=0.0, max_grad_norm=1.0,
                                    expected_batch_size=2, loss_reduction="mean")
            try:
                optimizer.step()
                item["dp_optimizer_step"] = True
            except Exception as error:
                item["dp_optimizer_step"] = False
                item["dp_optimizer_error"] = f"{type(error).__name__}: {error}"
        except Exception as error:
            item["grad_sample_error"] = f"{type(error).__name__}: {error}"
        output["models"][model_name] = item
    try:
        from opacus import PrivacyEngine
        PrivacyEngine(secure_mode=True)
        output["secure_rng"] = True
    except Exception as error:
        output["secure_rng"] = False
        output["secure_rng_error"] = f"{type(error).__name__}: {error}"
    output["torchcsprng_installed"] = importlib.util.find_spec("torchcsprng") is not None
    return output


def _rdp_examples() -> list[dict]:
    from opacus.accountants import RDPAccountant

    windows = 6084
    batch_size = 32
    rounds = 20
    steps_per_epoch = math.ceil(windows / batch_size)
    sample_rate = batch_size / windows
    delta = 1e-5
    rows = []
    for noise_multiplier in (0.5, 1.0, 2.0, 4.0):
        accountant = RDPAccountant()
        # Opacus' accountant composes Poisson-subsampled steps. Repeating a
        # history entry is equivalent to calling step once per local update.
        for _ in range(rounds * steps_per_epoch):
            accountant.step(noise_multiplier=noise_multiplier,
                            sample_rate=sample_rate)
        rows.append({"noise_multiplier": noise_multiplier,
                     "epsilon": float(accountant.get_epsilon(delta)),
                     "delta": delta, "sample_rate": sample_rate,
                     "steps": rounds * steps_per_epoch,
                     "assumption": "Poisson sampling; illustrative only"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/g3_0/compatibility.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    report = {"python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "opacus": _opacus_probe(),
              "torch_func": {"cpu": [_torch_func_probe(name, torch.device("cpu"))
                                     for name in ("gru_attention", "stfednet_gated")]},
              "reference": [_synthetic_probe(name, nodes, device)
                            for name in ("gru_attention", "stfednet_gated")
                            for nodes in (39, 80, 76)],
              "rdp_examples": _rdp_examples(),
              "test_split_accessed": False}
    if device.type == "cuda":
        # CUDA GRU's functorch path needs cuDNN disabled in this PyTorch build.
        with torch.backends.cudnn.flags(enabled=False):
            report["torch_func"]["cuda_cudnn_disabled"] = [
                _torch_func_probe(name, device) for name in ("gru_attention", "stfednet_gated")]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "device": str(device),
                      "test_split_accessed": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
