"""Single-client local update used by the single-machine FL simulator."""

import torch
from torch.utils.data import DataLoader, Subset

from src.models.forecasting import build_model
from src.training.engine import masked_huber, seed_everything
from src.fl.aggregation import fedprox_objective


def client_seed(base_seed, round_index, client_index):
    return int((int(base_seed) + 1009 * int(round_index) + 9176 * int(client_index)) % (2**31 - 1))


def client_update(global_state, prepared, config, model_name, round_index,
                  client_index, device, algorithm="fedavg", mu=0.01,
                  max_train_batches=None):
    """Train a private model copy; return only weights/count/mean local loss."""
    if algorithm not in {"fedavg", "fedprox"}:
        raise ValueError("Unsupported client algorithm")
    seed = client_seed(config["seed"], round_index, client_index)
    seed_everything(seed, config["cpu_threads"])
    model = build_model(model_name, config["hidden_dim"], config["dropout"],
                        config["horizon_hours"]).to(device)
    model.load_state_dict(global_state, strict=True)
    start_state = {key: value.detach().clone() for key, value in model.named_parameters()
                   if value.requires_grad}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    dataset = prepared.train
    if max_train_batches:
        dataset = Subset(dataset, range(min(len(dataset), int(max_train_batches) * config["batch_size"])))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                              generator=generator, num_workers=config.get("num_workers", 0))
    edges = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
    mask = torch.as_tensor(prepared.grid.load_mask, dtype=torch.bool, device=device)
    total_loss, observations = 0.0, 0
    huber_sum, prox_sum, grad_norm_sum, batches, clipped_batches = 0.0, 0.0, 0.0, 0, 0
    for _ in range(int(config["local_epochs"])):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, edges)
            huber = masked_huber(prediction, y, mask, config["huber_delta"])
            if algorithm == "fedprox":
                loss, prox = fedprox_objective(huber, model, start_state, mu, return_prox=True)
            else:
                loss, prox = huber, huber.new_zeros(())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite local objective for " + prepared.client_id)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            count = int(x.shape[0])
            observations += count
            total_loss += float(loss.detach()) * count
            huber_sum += float(huber.detach()) * count
            prox_sum += float(prox.detach()) * count
            # clip_grad_norm_ already returns the norm before clipping. Its
            # existing coefficient is max_norm / (norm + 1e-6), clamped at 1.
            norm = float(grad_norm.detach())
            grad_norm_sum += norm
            clipped_batches += int(norm + 1e-6 > config["gradient_clip"])
            batches += 1
    if observations <= 0:
        raise ValueError("Client " + prepared.client_id + " processed no train windows")
    # Move a CPU copy to the coordinator; no client data, topology, scaler or IDs
    # are included in this update payload.
    returned_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    # Formal n_k is the number of unique train windows, not node observations
    # or repeated epoch visits. Smoke n_k refers to the bounded subset.
    return {"state_dict": returned_state, "sample_count": len(dataset),
            "processed_windows": observations, "mean_loss": total_loss / observations,
            "diagnostics": {"huber_loss": huber_sum / observations,
                            "prox_loss": prox_sum / observations,
                            "total_loss": total_loss / observations,
                            "grad_norm_before_clip": grad_norm_sum / batches,
                            "clip_fraction": clipped_batches / batches,
                            "processed_windows": observations, "batch_count": batches,
                            "clipped_batch_count": clipped_batches}}
