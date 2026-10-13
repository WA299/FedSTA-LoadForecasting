from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.loader import GridData, prepare_client
from src.training.engine import evaluate, load_checkpoint, read_config, train


@pytest.fixture
def prepared():
    timestamps = np.arange(np.datetime64("2021-01-01T00", "h"),
                           np.datetime64("2021-01-21T00", "h"), np.timedelta64(1, "h"))
    values = np.zeros((len(timestamps), 3), dtype=np.float32)
    values[:, 0] = 3 + np.sin(np.arange(len(timestamps)) / 8)
    values[:, 1] = 2 + np.cos(np.arange(len(timestamps)) / 4)
    grid = GridData("synthetic", "synthetic", ["0", "1", "2"],
                    np.array([[0, 1, 1, 2], [1, 0, 2, 1]]), values, timestamps,
                    np.array([True, True, False]))
    return prepare_client(grid, horizon_hours=1)


def config():
    return dict(read_config(), batch_size=2, device="cpu", cpu_threads=2,
                epochs=2, max_train_batches=2, max_val_batches=2)


@pytest.mark.parametrize("model_name", ["stfednet", "stfednet_residual"])
def test_checkpoint_resume_matches_uninterrupted_run(tmp_path, prepared, model_name):
    settings = config()
    continuous = train(settings, model_name, "synthetic", tmp_path / "continuous", prepared=prepared)
    interrupted = train(dict(settings, epochs=1), model_name, "synthetic", tmp_path / "interrupted", prepared=prepared)
    latest = load_checkpoint(interrupted / "latest.pt")
    required = {"model_state_dict", "optimizer_state_dict", "epoch", "best_validation_wape",
                "model_name", "client_id", "config", "seed", "scaler", "rng_state", "stale_epochs"}
    assert required <= latest.keys()
    train(settings, model_name, "synthetic", resume=interrupted / "latest.pt", prepared=prepared)
    full, resumed = load_checkpoint(continuous / "latest.pt"), load_checkpoint(interrupted / "latest.pt")
    assert full["epoch"] == resumed["epoch"] == 2
    for key in full["model_state_dict"]:
        assert torch.equal(full["model_state_dict"][key], resumed["model_state_dict"][key])
    assert full["best_validation_wape"] == resumed["best_validation_wape"]
    with pytest.raises(ValueError, match="Smoke checkpoints"):
        evaluate(interrupted / "best.pt", prepared=prepared)
    result = evaluate(interrupted / "best.pt", device="cpu", max_batches=2, prepared=prepared)
    assert result["smoke"] and result["metrics"]["effective_count"] == 8
    saved = np.load(interrupted / "smoke_test" / "predictions.npz", allow_pickle=False)
    assert saved["y_true_kw"].shape == saved["y_pred_kw"].shape == (4, 1, 3)
    assert saved["timestamps"].shape == (4, 1)
    assert "[Checkpoint] New best model saved" in (interrupted / "train.log").read_text()
    assert len((interrupted / "epochs.csv").read_text().splitlines()) == 3
    if model_name == "stfednet_residual":
        import csv
        rows = list(csv.DictReader((interrupted / "epochs.csv").open(encoding="utf-8")))
        assert "SpaceScale=" in (interrupted / "train.log").read_text()
        assert float(rows[-1]["space_scale"]) == resumed["space_scale"]
        assert resumed["space_scale"] == resumed["model_state_dict"]["spatial.space_scale"].item()


@pytest.mark.parametrize("model_name", ["gru", "stfednet_residual"])
def test_early_stopping_and_training_does_not_read_test(tmp_path, prepared, monkeypatch, model_name):
    import src.training.engine as engine

    class ForbiddenTest:
        def __len__(self):
            raise AssertionError("Training accessed test data")

        def __getitem__(self, item):
            raise AssertionError("Training accessed test data")

    prepared.test = ForbiddenTest()
    actual_validate = engine.validate

    def constant_wape(*args, **kwargs):
        metrics, statistics, arrays = actual_validate(*args, **kwargs)
        metrics["wape_percent"] = 50.0
        return metrics, statistics, arrays

    monkeypatch.setattr(engine, "validate", constant_wape)
    settings = dict(config(), epochs=5, patience=1)
    directory = train(settings, model_name, "synthetic", tmp_path / "early", prepared=prepared)
    assert load_checkpoint(directory / "latest.pt")["epoch"] == 2
    assert load_checkpoint(directory / "best.pt")["epoch"] == 1
    assert "[EarlyStopping] Best epoch: 1" in (directory / "train.log").read_text()


def test_parameter_free_baseline_checkpoint(tmp_path, prepared):
    directory = train(config(), "seasonal_naive", "synthetic", tmp_path / "naive", prepared=prepared)
    checkpoint = load_checkpoint(directory / "best.pt")
    assert checkpoint["optimizer_state_dict"] is None
    assert checkpoint["model_state_dict"] == {}
