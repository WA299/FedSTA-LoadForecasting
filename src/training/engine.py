"""Shared training core; test evaluation is deliberately a separate entry point."""

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import yaml

from src.data.loader import StandardScaler, discover_grids, load_grid, prepare_client
from src.models.forecasting import build_model
from src.training.metrics import MetricAccumulator


def read_config(path="configs/g1.yaml"):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if config["horizon_hours"] != 1 or config["input_hours"] != 48:
        raise ValueError("G1 comparison protocol requires input_hours=48, horizon_hours=1")
    for key in ("batch_size", "epochs", "patience", "cpu_threads"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("max_train_batches", "max_val_batches"):
        if config.get(key) is not None and config[key] <= 0:
            raise ValueError(f"{key} must be positive or null")
    return config


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(requested)


def seed_everything(seed, cpu_threads=4):
    # Required by deterministic CUDA matrix operations; set before CUDA work.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def prepare_data(config, client_id):
    paths = dict(discover_grids(config["data_root"]))
    if client_id not in paths:
        raise ValueError(f"Unknown client {client_id}; choose from {tuple(paths)}")
    grid = load_grid(paths[client_id], client_id)
    return prepare_client(grid, config["input_hours"], config["horizon_hours"],
                          config["train_ratio"], config["val_ratio"])


def data_signature(prepared):
    """Bind checkpoints to aligned source data, topology, IDs and split boundaries."""
    digest = hashlib.sha256()
    for array in (prepared.grid.load_matrix, prepared.grid.edge_index,
                  prepared.grid.load_mask, prepared.grid.timestamps):
        digest.update(array.tobytes())
    digest.update(json.dumps(prepared.grid.node_ids).encode())
    digest.update(str(prepared.split_indices).encode())
    return digest.hexdigest()


def loader(dataset, config, shuffle=False, generator=None, max_batches=None):
    # A short smoke run uses the same deterministic leading subset in every model.
    if max_batches:
        dataset = Subset(dataset, range(min(len(dataset), max_batches * config["batch_size"])))
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      generator=generator, num_workers=config["num_workers"])


def masked_huber(prediction, target, mask, delta=1.0):
    # DP extension: the independent record will be one training window.
    # Masked node/time elements are reduced within that record, not DP records.
    if not mask.any():
        raise ValueError("No effective load nodes")
    return nn.functional.huber_loss(prediction[..., mask], target[..., mask], delta=delta)


@torch.no_grad()
def validate(model, dataset, prepared, config, device, max_batches=None, collect=False):
    model.eval()
    edges = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
    metrics = MetricAccumulator(config["mape_threshold_kw"])
    predictions, targets = [], []
    for x, y in loader(dataset, config, max_batches=max_batches):
        prediction = model(x.to(device), edges).cpu().numpy()
        truth_kw = prepared.scaler.inverse_transform(y.numpy())
        prediction_kw = prepared.scaler.inverse_transform(prediction)
        metrics.update(truth_kw, prediction_kw, prepared.grid.load_mask)
        if collect:
            predictions.append(prediction_kw)
            targets.append(truth_kw)
    arrays = (np.concatenate(targets), np.concatenate(predictions)) if collect else None
    return metrics.compute(), metrics.statistics(), arrays


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Payload contains only tensors and primitive Python values. Avoid pickle
    # loading of arbitrary objects; scaler arrays and NumPy RNG use tensors.
    return torch.load(path, map_location="cpu", weights_only=True)


def random_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {"algorithm": numpy_state[0],
                      "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
            "loader_generator": generator.get_state()}


def restore_random_state(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    generator.set_state(state["loader_generator"])


def restore_scaler(prepared, saved):
    if not np.array_equal(saved["mean"].numpy(), prepared.scaler.mean) or not np.array_equal(saved["scale"].numpy(), prepared.scaler.scale):
        raise ValueError("Checkpoint scaler differs from the current training data")
    prepared.scaler = StandardScaler(saved["mean"].numpy(), saved["scale"].numpy(),
                                     np.datetime64(saved["fit_start"], "h"), np.datetime64(saved["fit_end"], "h"))


def train(config, model_name, client_id, run_dir=None, resume=None, prepared=None):
    """Train with train/validation only. Test dataset is never accessed here.

    Epoch-boundary recovery restores optimizer, early stopping and all RNGs.
    Interrupted epochs restart from the previous completed latest checkpoint.
    """
    config = dict(config)
    seed_everything(config["seed"], config["cpu_threads"])
    device = select_device(config["device"])
    prepared = prepared or prepare_data(config, client_id)
    signature = data_signature(prepared)
    model = build_model(model_name, config["hidden_dim"], config["dropout"], config["horizon_hours"]).to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]) if parameters else None
    generator = torch.Generator().manual_seed(config["seed"])
    if resume:
        checkpoint = load_checkpoint(resume)
        if checkpoint["kind"] != "latest":
            raise ValueError("Resume requires latest.pt, not best.pt")
        if checkpoint["model_name"] != model_name or checkpoint["client_id"] != client_id or checkpoint["data_signature"] != signature:
            raise ValueError("Resume checkpoint model/client/data mismatch")
        mutable = {"epochs", "device", "cpu_threads", "output_root"}
        for key, value in checkpoint["config"].items():
            if key not in mutable and config.get(key) != value:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_scaler(prepared, checkpoint["scaler"])
        restore_random_state(checkpoint["rng_state"], generator)
        start_epoch = checkpoint["epoch"] + 1
        best_wape, best_epoch, stale = checkpoint["best_validation_wape"], checkpoint["best_epoch"], checkpoint["stale_epochs"]
        run_dir = Path(resume).resolve().parent
        if not (run_dir / "best.pt").exists():
            raise ValueError("Resume directory must retain the associated best.pt")
    else:
        start_epoch, best_wape, best_epoch, stale = 1, float("inf"), 0, 0
        suffix = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
        run_dir = Path(run_dir) if run_dir else Path(config["output_root"]) / model_name / client_id / f"seed{config['seed']}_{suffix}"
        if (run_dir / "latest.pt").exists():
            raise ValueError("Run directory already contains checkpoints; use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)

    def log(message):
        print(message, flush=True)
        with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"Model={model_name} | Client={client_id} | Device={device} | BatchSize={config['batch_size']} | LearningRate={config['learning_rate']} | Input={config['input_hours']} | Horizon={config['horizon_hours']} | Epochs={config['epochs']} | Seed={config['seed']}")
    smoke = bool(config.get("max_train_batches") or config.get("max_val_batches"))
    if smoke:
        log("[Smoke] Limited batches; metrics are engineering checks, not experiment results")
    train_loader = loader(prepared.train, config, True, generator, config.get("max_train_batches"))
    edges = torch.as_tensor(prepared.grid.edge_index, dtype=torch.long, device=device)
    mask = torch.as_tensor(prepared.grid.load_mask, dtype=torch.bool, device=device)
    csv_path = run_dir / "epochs.csv"
    fields = ["epoch", "train_loss", "val_mae_kw", "val_rmse_kw", "val_wape_percent", "val_mape_percent", "mape_valid_fraction", "best_wape_percent", "seconds"]
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    end_epoch = config["epochs"] if optimizer else 1
    if not optimizer:
        log("[Baseline] Parameter-free forecast; one validation pass, no optimizer updates")
    if resume and stale >= config["patience"]:
        log(f"[EarlyStopping] Already stopped; Best epoch: {best_epoch}")
        return run_dir
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            begin = time.perf_counter()
            model.train()
            total_loss, observations = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if optimizer:
                    optimizer.zero_grad(set_to_none=True)
                prediction = model(x, edges)
                loss = masked_huber(prediction, y, mask, config["huber_delta"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                if optimizer:
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
                    optimizer.step()
                total_loss += float(loss.detach()) * x.shape[0]
                observations += x.shape[0]
            metrics, _, _ = validate(model, prepared.val, prepared, config, device, config.get("max_val_batches"))
            wape = metrics["wape_percent"]
            if wape is None or not np.isfinite(wape):
                raise ValueError("Validation WAPE undefined; cannot select best checkpoint")
            improved = wape < best_wape - config["min_delta"]
            if improved:
                best_wape, best_epoch, stale = wape, epoch, 0
            else:
                stale += 1
            payload = {"format_version": 1, "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                       "epoch": epoch, "best_validation_wape": best_wape, "best_epoch": best_epoch,
                       "stale_epochs": stale, "model_name": model_name, "client_id": client_id,
                       "config": config, "seed": config["seed"], "validation_metrics": metrics,
                       "smoke": smoke, "data_signature": signature, "node_ids": prepared.grid.node_ids,
                       "scaler": {"mean": torch.from_numpy(prepared.scaler.mean.copy()),
                                  "scale": torch.from_numpy(prepared.scaler.scale.copy()),
                                  "fit_start": str(prepared.scaler.fit_start), "fit_end": str(prepared.scaler.fit_end)},
                       "rng_state": random_state(generator)}
            if improved:
                atomic_save(dict(payload, kind="best"), run_dir / "best.pt")
            atomic_save(dict(payload, kind="latest"), run_dir / "latest.pt")
            seconds = time.perf_counter() - begin
            mape = metrics["mape_percent"]
            mape_text = f"{mape:.4f}" if mape is not None else "n/a"
            log(f"Epoch {epoch:02d}/{end_epoch} | TrainLoss={total_loss/observations:.6f} | ValMAE={metrics['mae_kw']:.4f} | ValRMSE={metrics['rmse_kw']:.4f} | ValWAPE={wape:.4f} | ValMAPE={mape_text} | Best={best_wape:.4f} | Time={seconds:.2f}s")
            if improved:
                log("[Checkpoint] New best model saved")
            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(dict(zip(fields, [epoch, total_loss/observations,
                    metrics["mae_kw"], metrics["rmse_kw"], wape, mape, metrics["mape_valid_fraction"], best_wape, seconds])))
            if stale >= config["patience"]:
                log(f"[EarlyStopping] Best epoch: {best_epoch}")
                break
    except KeyboardInterrupt:
        log("[Interrupted] Resume from latest.pt; the incomplete epoch will restart")
    log(f"[Run] Checkpoints and logs: {run_dir}")
    return run_dir


def evaluate(checkpoint_path, output_dir=None, device="auto", max_batches=None, prepared=None):
    """Independent testing of a best checkpoint; never performs optimization."""
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["kind"] != "best":
        raise ValueError("Formal evaluation must load best.pt")
    if checkpoint["smoke"] and max_batches is None:
        raise ValueError("Smoke checkpoints cannot be used for formal testing; pass --max-batches for a smoke check")
    config = checkpoint["config"]
    seed_everything(config["seed"], config["cpu_threads"])
    selected_device = select_device(device)
    prepared = prepared or prepare_data(config, checkpoint["client_id"])
    if data_signature(prepared) != checkpoint["data_signature"]:
        raise ValueError("Evaluation data differs from checkpoint data")
    restore_scaler(prepared, checkpoint["scaler"])
    model = build_model(checkpoint["model_name"], config["hidden_dim"], config["dropout"], config["horizon_hours"]).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    output = Path(output_dir) if output_dir else Path(checkpoint_path).resolve().parent / ("smoke_test" if max_batches else "test")
    output.mkdir(parents=True, exist_ok=True)
    metrics, statistics, (truth, prediction) = validate(model, prepared.test, prepared, config, selected_device, max_batches, collect=True)
    result = {"model_name": checkpoint["model_name"], "client_id": checkpoint["client_id"],
              "seed": checkpoint["seed"], "checkpoint": str(Path(checkpoint_path).resolve()),
              "epoch": checkpoint["epoch"], "best_validation_wape": checkpoint["best_validation_wape"],
              "input_hours": config["input_hours"], "horizon_hours": config["horizon_hours"],
              "mape_threshold_kw": config["mape_threshold_kw"], "unit": "kW",
              "smoke": bool(checkpoint["smoke"] or max_batches), "metrics": metrics, "statistics": statistics}
    write_json(output / "metrics.json", result)
    timestamps = prepared.test.target_timestamps[:truth.shape[0]]
    np.savez_compressed(output / "predictions.npz", y_true_kw=truth, y_pred_kw=prediction,
                        timestamps=timestamps.astype(str), node_ids=np.asarray(prepared.grid.node_ids),
                        load_mask=prepared.grid.load_mask)
    message = f"[{'Smoke Test' if result['smoke'] else 'Test'}] MAE={metrics['mae_kw']:.4f} kW | RMSE={metrics['rmse_kw']:.4f} kW | WAPE={metrics['wape_percent']}% | MAPE={metrics['mape_percent']}% | MAPE valid={metrics['mape_valid_fraction']:.2%}\n[Results] {output}"
    print(message, flush=True)
    (output / "test.log").write_text(message + "\n", encoding="utf-8")
    return result
