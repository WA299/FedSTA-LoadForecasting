"""Reference per-window gradient clipping for G3.0 feasibility checks.

This deliberately uses an explicit record loop. It is a correctness reference,
not a production DP trainer or an optimized replacement for the G2 client loop.
One record is one complete graph window, with the masked node errors reduced
within that record.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
from torch.nn import functional as F


def masked_huber_per_sample(prediction: torch.Tensor, target: torch.Tensor,
                            load_mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Return one mean masked Huber loss per graph window, shape ``[B]``.

    The reduction within each record is the ordinary mean over its horizon and
    effective load nodes. Thus ``losses.mean()`` exactly matches the existing
    ``masked_huber`` batch mean when every record has the same fixed mask.
    """
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have equal shapes [B,...,N]")
    mask = torch.as_tensor(load_mask, dtype=torch.bool, device=prediction.device)
    if mask.ndim != 1 or mask.numel() != prediction.shape[-1]:
        raise ValueError("load_mask must have one entry for every graph node")
    if not bool(mask.any()):
        raise ValueError("No effective load nodes")
    element_losses = F.huber_loss(prediction[..., mask], target[..., mask],
                                  delta=float(delta), reduction="none")
    return element_losses.reshape(prediction.shape[0], -1).mean(dim=1)


def per_sample_clipped_gradients(model, x: torch.Tensor, y: torch.Tensor,
                                 edge_index: torch.Tensor, load_mask: torch.Tensor,
                                 max_grad_norm: float, noise_multiplier: float = 0.0,
                                 delta: float = 1.0,
                                 normalizer: Optional[float] = None,
                                 generator: Optional[torch.Generator] = None) -> Dict[str, object]:
    """Compute mean clipped per-window gradients, optionally adding Gaussian noise.

    For each complete graph window, this computes a gradient of its masked mean
    Huber loss, clips the concatenated trainable-parameter gradient to
    ``max_grad_norm``, sums clipped gradients, adds noise with standard
    deviation ``noise_multiplier * max_grad_norm`` to that sum, then divides by
    ``normalizer`` (the actual batch size by default). This matches an add/remove
    record convention. A replacement-neighbour convention needs twice the
    sensitivity/noise scale.

    The caller should leave the model in training mode while computing
    gradients; cuDNN's native GRU does not permit backward in evaluation mode.
    Set dropout to zero for deterministic numerical comparisons.

    ``normalizer`` may be set to an expected Poisson batch size, including for
    an empty Poisson batch. For a fixed-size final partial batch, leave it unset
    so the sum is divided by the actual number of records. A deterministic
    ``generator`` exists for tests only; this utility is not a production DP
    optimizer and does not provide cryptographically secure randomness.
    """
    clip = float(max_grad_norm)
    sigma = float(noise_multiplier)
    if not clip > 0 or not torch.isfinite(torch.tensor(clip)):
        raise ValueError("max_grad_norm must be finite and positive")
    if sigma < 0 or not torch.isfinite(torch.tensor(sigma)):
        raise ValueError("noise_multiplier must be finite and non-negative")
    if x.ndim != 4 or y.ndim != 3 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected x=[B,T,N,F], y=[B,H,N] with matching B")
    if x.shape[2] != y.shape[2]:
        raise ValueError("Input and target node dimensions differ")

    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    if any(parameter.device != x.device for _, parameter in parameters):
        raise ValueError("Model parameters and batch must be on the same device")

    batch_size = int(x.shape[0])
    divisor = float(batch_size if normalizer is None else normalizer)
    if divisor <= 0 or not torch.isfinite(torch.tensor(divisor)):
        raise ValueError("normalizer must be finite and positive")
    if batch_size == 0 and normalizer is None:
        raise ValueError("An empty batch requires an explicit expected normalizer")

    sums = {name: torch.zeros_like(parameter) for name, parameter in parameters}
    losses = []
    pre_clip_norms = []
    clipped_norms = []
    for index in range(batch_size):
        prediction = model(x[index:index + 1], edge_index)
        loss = masked_huber_per_sample(prediction, y[index:index + 1],
                                       load_mask, delta)[0]
        gradients = torch.autograd.grad(loss, [parameter for _, parameter in parameters],
                                        allow_unused=True, retain_graph=False,
                                        create_graph=False)
        squared_norm = torch.zeros((), device=x.device, dtype=torch.float64)
        for gradient in gradients:
            if gradient is not None:
                squared_norm += gradient.detach().to(torch.float64).square().sum()
        norm = squared_norm.sqrt()
        factor = torch.clamp(torch.as_tensor(clip, device=x.device, dtype=torch.float64) /
                             (norm + 1e-12), max=1.0)
        for (name, _), gradient in zip(parameters, gradients):
            if gradient is not None:
                sums[name].add_(gradient.detach(), alpha=float(factor))
        pre_clip_norms.append(float(norm))
        clipped_norms.append(float(norm * factor))
        losses.append(float(loss.detach()))

    noise_std_on_sum = sigma * clip
    result = {}
    for name, parameter in parameters:
        total = sums[name]
        if sigma:
            noise = torch.randn(total.shape, dtype=total.dtype, device=total.device,
                                generator=generator)
            total = total + noise * noise_std_on_sum
        result[name] = total / divisor
    return {"gradients": result,
            "losses": losses,
            "pre_clip_norms": pre_clip_norms,
            "clipped_norms": clipped_norms,
            "record_count": batch_size,
            "normalizer": divisor,
            "noise_std_on_sum": noise_std_on_sum,
            "noise_std_on_mean": noise_std_on_sum / divisor,
            "max_grad_norm": clip,
            "noise_multiplier": sigma}


def torch_func_clipped_gradients(model, x: torch.Tensor, y: torch.Tensor,
                                 edge_index: torch.Tensor, load_mask: torch.Tensor,
                                 max_grad_norm: float, noise_multiplier: float = 0.0,
                                 delta: float = 1.0,
                                 normalizer: Optional[float] = None,
                                 generator: Optional[torch.Generator] = None) -> Dict[str, object]:
    """Experimental ``torch.func`` per-record gradient path.

    This path materializes a ``[B, ...parameter_shape]`` gradient stack and
    disables cuDNN for CUDA GRU transforms in the currently tested PyTorch
    build. It is kept for comparison only; the explicit loop is the supported
    numerical reference. Some PyTorch CPU/CUDA configurations cannot transform
    the native GRU, in which case the underlying exception is intentionally
    propagated to the caller.
    """
    from torch.func import functional_call, grad, vmap

    clip = float(max_grad_norm)
    sigma = float(noise_multiplier)
    if clip <= 0 or sigma < 0:
        raise ValueError("max_grad_norm must be positive and noise_multiplier non-negative")
    batch_size = int(x.shape[0])
    divisor = float(batch_size if normalizer is None else normalizer)
    if batch_size == 0 or divisor <= 0:
        raise ValueError("torch.func path requires a non-empty batch and positive normalizer")
    parameters = {name: parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad}
    buffers = dict(model.named_buffers())
    mask = torch.as_tensor(load_mask, dtype=torch.bool, device=x.device)

    def loss_fn(params, sample_x, sample_y):
        prediction = functional_call(model, (params, buffers),
                                     (sample_x.unsqueeze(0), edge_index))
        return masked_huber_per_sample(prediction, sample_y.unsqueeze(0),
                                       mask, delta)[0]

    transform_context = (torch.backends.cudnn.flags(enabled=False)
                         if x.device.type == "cuda" else nullcontext())
    with transform_context:
        gradient_stack = vmap(grad(loss_fn), in_dims=(None, 0, 0),
                              randomness="different")(parameters, x, y)
    # Optimizer gradients are values, not differentiable meta-gradients. Keep
    # no higher-order graph alive between benchmark steps or in parameter.grad.
    gradient_stack = {name: value.detach() for name, value in gradient_stack.items()}
    squared_norms = torch.zeros(batch_size, device=x.device, dtype=torch.float64)
    for gradient in gradient_stack.values():
        squared_norms += gradient.detach().to(torch.float64).reshape(batch_size, -1).square().sum(1)
    norms = squared_norms.sqrt()
    factors = torch.clamp(float(clip) / (norms + 1e-12), max=1.0)
    noise_std_on_sum = sigma * clip
    result = {}
    for name, gradient in gradient_stack.items():
        shape = (batch_size,) + (1,) * (gradient.ndim - 1)
        total = (gradient * factors.to(gradient.dtype).reshape(shape)).sum(dim=0)
        if sigma:
            noise = torch.randn(total.shape, dtype=total.dtype, device=total.device,
                                generator=generator)
            total = total + noise * noise_std_on_sum
        result[name] = total / divisor
    return {"gradients": result,
            "pre_clip_norms": norms.detach().cpu().tolist(),
            "clipped_norms": (norms * factors).detach().cpu().tolist(),
            "record_count": batch_size,
            "normalizer": divisor,
            "noise_std_on_sum": noise_std_on_sum,
            "noise_std_on_mean": noise_std_on_sum / divisor,
            "max_grad_norm": clip,
            "noise_multiplier": sigma,
            "gradient_stack_bytes": sum(value.numel() * value.element_size()
                                         for value in gradient_stack.values())}
