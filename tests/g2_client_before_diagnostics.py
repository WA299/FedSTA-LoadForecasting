# Regression oracle: client loop from commit 3e64a6f, not production code.
"""Single-client local update used by the single-machine FL simulator."""

import torch
from torch.utils.data import DataLoader, Subset

from src.models.forecasting import build_model
from src.training.engine import masked_huber, seed_everything
# Frozen pre-G2.1 objective; used only as a regression oracle.
def fedprox_objective(huber_loss, model, global_state, mu):
    if float(mu) == 0:
        return huber_loss
    terms = [torch.sum((parameter - global_state[name].to(device=parameter.device, dtype=parameter.dtype)) ** 2)
             for name, parameter in model.named_parameters() if parameter.requires_grad]
    return huber_loss + (float(mu) / 2.0) * torch.stack(terms).sum()



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
    for _ in range(int(config["local_epochs"])):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, edges)
            huber = masked_huber(prediction, y, mask, config["huber_delta"])
            loss = (fedprox_objective(huber, model, start_state, mu)
                    if algorithm == "fedprox" else huber)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite local objective for " + prepared.client_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            count = int(x.shape[0])
            observations += count
            total_loss += float(loss.detach()) * count
    if observations <= 0:
        raise ValueError("Client " + prepared.client_id + " processed no train windows")
    # Move a CPU copy to the coordinator; no client data, topology, scaler or IDs
    # are included in this update payload.
    returned_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    # Formal n_k is the number of unique train windows, not node observations
    # or repeated epoch visits. Smoke n_k refers to the bounded subset.
    return {"state_dict": returned_state, "sample_count": len(dataset),
            "processed_windows": observations, "mean_loss": total_loss / observations}
