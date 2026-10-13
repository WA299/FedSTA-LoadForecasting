"""Check five-client orchestration without starting any real long-running jobs."""

import json

import torch

from scripts import compare_validation, train_clients
from src.training.engine import read_config


def test_batch_training_preserves_settings_and_never_evaluates(tmp_path, monkeypatch):
    calls = []

    def bounded_stub(config, model, client):
        calls.append((dict(config), model, client))
        return tmp_path / client

    monkeypatch.setattr(train_clients, "train", bounded_stub)
    monkeypatch.setattr(train_clients, "load_checkpoint", lambda path: {
        "epoch": 3, "validation_metrics": {"wape_percent": 12.0},
        "space_scale": 0.06, "smoke": False})
    summary = tmp_path / "manifest.json"
    monkeypatch.setattr("sys.argv", ["train_clients", "--summary", str(summary),
                                     "--output-root", str(tmp_path / "runs")])
    train_clients.main()
    assert [client for _, _, client in calls] == list(train_clients.CLIENT_IDS)
    for config, model, client in calls:
        assert model == "stfednet_residual"
        assert config == dict(read_config(), output_root=str(tmp_path / "runs"))
    saved = json.loads(summary.read_text())
    assert saved["split"] == "validation" and saved["test_evaluated"] is False
    assert len(saved["results"]) == 5


def test_validation_comparison_end_to_end_reads_only_metadata(tmp_path, monkeypatch):
    original, residual = tmp_path / "g1", tmp_path / "g11"
    for model, wape in [("gru_attention", 10.0), ("stfednet", 20.0), ("stfednet_residual", 9.0)]:
        root = residual if model == "stfednet_residual" else original
        for client in compare_validation.CLIENTS:
            directory = root / model / client / "run"
            directory.mkdir(parents=True)
            # No weights or data are present: this tool only needs validation metadata.
            torch.save({"kind": "best", "smoke": False, "model_name": model,
                        "client_id": client, "seed": 42, "epoch": 5,
                        "config": read_config(), "data_signature": client,
                        "validation_metrics": {"wape_percent": wape}}, directory / "best.pt")
    output = tmp_path / "comparison.json"
    monkeypatch.setattr("sys.argv", ["compare_validation", "--g1-root", str(original),
                                     "--g11-root", str(residual), "--output", str(output)])
    compare_validation.main()
    saved = json.loads(output.read_text())
    assert saved["test_evaluated"] is False and saved["client_count"] == 5
    assert saved["residual_wins"] == {"gru_attention": 5, "stfednet": 5}
    assert saved["client_equal_validation_wape_percent"]["stfednet_residual"] == 9.0
