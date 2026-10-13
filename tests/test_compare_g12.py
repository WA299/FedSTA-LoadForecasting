import json

import pytest
import torch

from scripts import compare_g12, train_clients
from src.training.engine import read_config


def save_comparison(tmp_path):
    roots = {model: tmp_path / model for model in compare_g12.MODELS}
    for model in compare_g12.MODELS:
        for client in compare_g12.CLIENTS:
            run = roots[model] / model / client / "run"
            run.mkdir(parents=True)
            torch.save({"kind": "best", "model_name": model, "client_id": client,
                        "seed": 42, "config": read_config(), "data_signature": client,
                        "epoch": 3, "smoke": False, "validation_metrics": {
                            "mae_kw": 1.0, "rmse_kw": 2.0, "wape_percent":
                            8.0 if model == "stfednet_gated" else 10.0, "mape_percent": 20.0}}, run / "best.pt")
    return roots


def test_four_models_comparison_without_dataset(tmp_path):
    roots = save_comparison(tmp_path)
    result = compare_g12.compare(roots, compare_g12.CLIENTS, 42)
    assert result["test_evaluated"] is False and result["complete_five_clients"]
    assert result["gated_wins"] == {model: 5 for model in compare_g12.MODELS[:-1]}
    assert result["client_equal_validation_wape_percent"]["stfednet_gated"] == 8.0
    assert set(result["comparisons"][0]["validation_metrics"]["stfednet_gated"]) == set(compare_g12.METRICS)
    # Reject mismatched data even when all other checkpoint fields agree.
    path = roots["stfednet"] / "stfednet" / "lv_39" / "run" / "best.pt"
    saved = torch.load(path, weights_only=True)
    saved["data_signature"] = "wrong"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="data/signature"):
        compare_g12.compare(roots, compare_g12.CLIENTS, 42)


def test_gated_batch_defaults_to_g12_preserves_config(tmp_path, monkeypatch):
    settings = read_config()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_clients, "read_config", lambda path: settings)
    calls = []
    monkeypatch.setattr(train_clients, "train", lambda conf, model, client:
                        calls.append((dict(conf), model, client)) or tmp_path)
    monkeypatch.setattr(train_clients, "load_checkpoint", lambda path: {
        "epoch": 1, "validation_metrics": {}, "validation_gate_statistics": {"gate_mean": 0.05}, "smoke": False})
    monkeypatch.setattr("sys.argv", ["train_clients", "--model", "stfednet_gated"])
    train_clients.main()
    assert len(calls) == 5 and all(conf["output_root"] == "results/g1_2" for conf, _, _ in calls)
    assert all(conf == dict(settings, output_root="results/g1_2") for conf, _, _ in calls)
    saved = json.loads((tmp_path / "results/g1_2/batch_summary.json").read_text())
    assert saved["test_evaluated"] is False
    assert saved["results"][0]["validation_gate_statistics"]["gate_mean"] == 0.05
