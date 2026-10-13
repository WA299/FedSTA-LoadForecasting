"""Model-parameter aggregation and FedProx objective helpers."""

from collections import OrderedDict

import torch


def fedavg_aggregate(client_states, sample_counts):
    """Return a sample-count weighted state dict without mutating inputs."""
    if not client_states or len(client_states) != len(sample_counts):
        raise ValueError("Provide one positive sample count for each client state")
    counts = [int(count) for count in sample_counts]
    if any(count <= 0 for count in counts):
        raise ValueError("All client sample counts must be positive")
    reference = client_states[0]
    keys = tuple(reference)
    for state in client_states[1:]:
        if tuple(state) != keys:
            raise ValueError("Client state dictionaries must have identical parameter keys")
        for key in keys:
            if state[key].shape != reference[key].shape or state[key].dtype != reference[key].dtype:
                raise ValueError("Client parameter shapes and dtypes must match")
    total = sum(counts)
    result = OrderedDict()
    for key in keys:
        first = reference[key]
        if torch.is_floating_point(first) or torch.is_complex(first):
            value = torch.zeros_like(first)
            for state, count in zip(client_states, counts):
                value.add_(state[key].to(device=value.device), alpha=count / total)
            result[key] = value
        else:
            # Current forecasting models have no integer buffers. Preserve a
            # shared integral buffer, and reject divergent counters explicitly.
            if any(not torch.equal(first, state[key]) for state in client_states[1:]):
                raise ValueError("Cannot average divergent non-floating model buffers")
            result[key] = first.clone()
    return result


def proximal_penalty(model, global_state):
    """Squared parameter distance from this round's starting global model."""
    terms = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name not in global_state or global_state[name].shape != parameter.shape:
                raise ValueError("Global parameter state does not match local model")
            anchor = global_state[name].to(device=parameter.device, dtype=parameter.dtype)
            terms.append(torch.sum((parameter - anchor) ** 2))
    if not terms:
        return next(model.parameters(), torch.zeros((), requires_grad=True)).new_zeros(())
    return torch.stack(terms).sum()


def fedprox_objective(huber_loss, model, global_state, mu, *, return_prox=False):
    """Huber plus ``mu/2 * ||w-w_start||^2`` over trainable parameters."""
    mu = float(mu)
    if mu < 0:
        raise ValueError("FedProx mu must be non-negative")
    if mu == 0:
        return (huber_loss, huber_loss.new_zeros(())) if return_prox else huber_loss
    prox = (mu / 2.0) * proximal_penalty(model, global_state)
    total = huber_loss + prox
    return (total, prox) if return_prox else total
