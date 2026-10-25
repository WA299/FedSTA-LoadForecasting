"""Final-suite tests use synthetic data, never the formal held-out dataset."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.loader import StandardScaler
from src.fl.common import CLIENT_IDS
from src.models.forecasting import build_model
from src.training import final_evaluation as final
from src.training.engine import data_signature
from src.training.metrics import MetricAccumulator


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    timestamps = np.arange(np.datetime64("2021-01-01T00"), np.datetime64("2021-01-06T00"), dtype="datetime64[h]")
    values = np.arange(len(timestamps), dtype=np.float32)[:, None] * np.array([[1, 2, 0]], np.float32)
    grid = SimpleNamespace(client_id="lv_39", timestamps=timestamps, load_matrix=values,
                           edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
                           load_mask=np.array([True, True, False]), node_ids=[1, 2, 3])
    config = {"input_hours": 48, "horizon_hours": 1, "train_ratio": .7, "val_ratio": .15,
              "hidden_dim": 32, "dropout": .1, "batch_size": 32, "learning_rate": .001,
              "weight_decay": .0001, "seed": 42, "mape_threshold_kw": .1,
              "huber_delta": 1., "data_root": "Datasets", "epochs": 30, "patience": 8,
              "max_train_batches": None, "max_val_batches": None}
    signature = data_signature(SimpleNamespace(grid=grid, split_indices=(84, 102)))
    saved = {"mean": torch.tensor([40., 80., 0.]), "scale": torch.tensor([20., 40., 1.]),
             "fit_start": str(timestamps[0]), "fit_end": str(timestamps[83])}
    path = tmp_path / "results/g1/gru_attention/lv_39/seed42_fixture/best.pt"
    path.parent.mkdir(parents=True)
    metrics = {"mae_kw": 1., "rmse_kw": 2., "wape_percent": 10., "mape_percent": 12.,
               "mape_valid_fraction": 1., "mape_threshold_kw": .1}
    best = {"kind": "best", "model_name": "gru_attention", "client_id": "lv_39", "config": config,
            "seed": 42, "smoke": False, "epoch": 22, "best_epoch": 22, "stale_epochs": 0,
            "best_validation_wape": 10., "validation_metrics": metrics, "data_signature": signature,
            "scaler": saved, "model_state_dict": build_model("gru_attention").state_dict()}
    torch.save(best, path)
    torch.save(dict(best, kind="latest", epoch=30), path.with_name("latest.pt"))
    monkeypatch.setattr(final, "discover_grids", lambda root: [("lv_39", "dummy")])
    monkeypatch.setattr(final, "load_grid", lambda path, cid: grid)
    return tmp_path, path, best, grid


def test_plan_does_not_load_data_or_fit_scaler(setup_run, monkeypatch):
    root, _, _, _ = setup_run
    def forbidden(*args, **kwargs):
        pytest.fail("Planning loaded data or fitted a scaler")
    monkeypatch.setattr(final, "load_grid", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    manifest = final.build_manifest(root)
    final.verify_manifest(manifest, root)
    assert len(manifest["entries"]) == 1
    assert manifest["plan_reads_source_data"] is False
    assert len(manifest["missing"]) == 20  # 14 missing local + 6 global groups


@pytest.mark.parametrize("change", ["config", "weights"])
def test_frozen_manifest_and_checkpoint_tamper_rejected(setup_run, change):
    root, path, best, _ = setup_run
    manifest = final.build_manifest(root)
    if change == "config":
        manifest["entries"][0]["config"]["input_hours"] = 24
    else:
        torch.save(dict(best, epoch=21), path)
    with pytest.raises(ValueError, match="changed"):
        final.verify_manifest(manifest, root)


@pytest.mark.parametrize("reason", ["smoke", "incomplete", "previous_test", "different_seed", "limited_batches"])
def test_ineligible_runs_excluded(setup_run, reason):
    root, path, best, _ = setup_run
    if reason == "smoke":
        best["smoke"] = True
    elif reason == "incomplete":
        latest = dict(best, kind="latest", epoch=23)
        torch.save(latest, path.with_name("latest.pt"))
    elif reason == "previous_test":
        (path.parent / "test").mkdir()
        final.atomic_json(path.parent / "test/metrics.json", {"checkpoint": str(path), "smoke": False})
    elif reason == "different_seed":
        best["seed"] = 2026
    else:
        best["config"]["max_train_batches"] = 1
    torch.save(best, path)
    manifest = final.build_manifest(root)
    assert not manifest["entries"]
    assert manifest["skipped"]


def test_newest_completed_repeat_not_lowest_validation(setup_run):
    root, path, best, _ = setup_run
    new_path = path.parent.with_name("seed42_new") / "best.pt"
    new_path.parent.mkdir()
    best["best_validation_wape"] = 20.
    best["validation_metrics"]["wape_percent"] = 20.
    torch.save(best, new_path)
    torch.save(dict(best, kind="latest", epoch=30), new_path.with_name("latest.pt"))
    import os
    newer = path.with_name("latest.pt").stat().st_mtime_ns + 1_000_000_000
    os.utime(new_path.with_name("latest.pt"), ns=(newer, newer))
    manifest = final.build_manifest(root)
    assert "seed42_new" in manifest["entries"][0]["checkpoint"]["path"]
    assert manifest["entries"][0]["validation_metrics"]["wape_percent"] == 20.


def test_test_only_recovers_saved_scaler_without_fitting(setup_run, monkeypatch):
    root, _, _, grid = setup_run
    entry = final.build_manifest(root)["entries"][0]
    monkeypatch.setattr(StandardScaler, "fit", lambda *a, **k: pytest.fail("refit"))
    prepared = final.prepare_test_only(entry, root)
    assert not hasattr(prepared, "train") and not hasattr(prepared, "val")
    assert len(prepared.test) == 18
    assert prepared.test.target_timestamps[0, 0] == grid.timestamps[102]
    assert prepared.test.target_timestamps[-1, -1] == grid.timestamps[-1]
    x, y = prepared.test[0]
    np.testing.assert_array_equal(prepared.scaler.inverse_transform(y.numpy()), grid.load_matrix[102:103])
    assert x.shape == (48, 3, 1)
    grid.load_matrix[0, 0] += 1
    with pytest.raises(ValueError, match="changed"):
        final.prepare_test_only(entry, root)


@pytest.mark.parametrize("model_name,kind", [("gru_attention", "best"),
                                            ("stfednet_gated", "best_global"),
                                            ("stfednet_gated", "best_dp_global")])
def test_inference_supports_checkpoint_kinds_no_updates(setup_run, monkeypatch, model_name, kind):
    root, path, best, _ = setup_run
    entry = final.build_manifest(root)["entries"][0]
    best["kind"] = kind
    best["model_name"] = model_name
    best["model_state_dict"] = build_model(model_name).state_dict()
    torch.save(best, path)
    entry["kind"], entry["model_name"] = kind, model_name
    before = final.sha256(path)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer created"))
    monkeypatch.setattr(StandardScaler, "fit", lambda *a, **k: pytest.fail("refit"))
    result = final.evaluate_entry(entry, root, torch.device("cpu"))
    assert result["metrics"]["effective_count"] == 18 * 2
    assert result["model_state_unchanged"]
    assert result["split"] == "test" and not result["smoke"]
    assert final.sha256(path) == before


def test_pooled_wape_and_equal_wape_are_distinct():
    rows = []
    for cid, truth, prediction in [("a", 1., 2.), ("b", 100., 110.)]:
        a = MetricAccumulator()
        a.update(np.array([[[truth, 0.]]]), np.array([[[prediction, 999.]]]), [True, False])
        rows.append({"method": "one", "client_id": cid, "metrics": a.compute(), "statistics": a.statistics(), "privacy": None})
    summary = final.summarize(rows)["one"]
    assert summary["client_equal_wape_percent"] == 55.
    assert summary["overall"]["wape_percent"] == pytest.approx(100 * 11 / 101)
    assert not summary["complete_five_clients"]


def test_execution_ledger_skips_completed_evaluations(setup_run, monkeypatch):
    root, _, _, _ = setup_run
    manifest = final.build_manifest(root)
    result = final.execute_manifest(manifest, root, device="cpu")
    monkeypatch.setattr(final, "evaluate_entry", lambda *a, **k: pytest.fail("repeated formal TEST"))
    repeat = final.execute_manifest(manifest, root, device="cpu")
    assert repeat == result
    assert (root / "results/final_test/FINAL_TEST_SUMMARY.md").exists()
    state_path = root / "results/final_test/execution_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["active"] = "ambiguous_interrupted_entry"
    final.atomic_json(state_path, state)
    with pytest.raises(RuntimeError, match="interrupted"):
        final.execute_manifest(manifest, root, device="cpu")


def test_best_round_accounting_not_latest_and_sigma_zero():
    from src.privacy.accountant import ClientRDPAccountant
    payload = {"kind": "best_dp_global", "noise_multiplier": 1., "best_round": 1}
    accountant = ClientRDPAccountant(1., 32 / 6084, 1., 1e-5)
    for _ in range(191):
        accountant.step()
    payload["accountant_states"] = {cid: accountant.state_dict() for cid in CLIENT_IDS}
    summary = final.privacy_summary(payload)
    assert summary["best_checkpoint_accounting"]["lv_39"]["steps"] == 191
    assert not summary["privacy_claim_allowed"]
    payload["noise_multiplier"] = 0.
    zero = ClientRDPAccountant(0., 32 / 6084, 1., 1e-5)
    for _ in range(191):
        zero.step()
    payload["accountant_states"] = {cid: zero.state_dict() for cid in CLIENT_IDS}
    summary = final.privacy_summary(payload)
    assert summary["status"] == "non_dp_control"
    assert all(s["theoretical_epsilon"] is None for s in summary["best_checkpoint_accounting"].values())


@pytest.mark.parametrize("kind", ["best_global", "best_dp_global"])
def test_global_complete_pair_and_round_validation(setup_run, kind):
    _, path, local, _ = setup_run
    config = dict(local["config"], rounds=20, local_epochs=1)
    if kind == "best_dp_global":
        config.update(expected_batch_size=32, expected_train_records=6084, max_grad_norm=1.,
                      delta=1e-5, privacy_mode="research")
    validation = {"round": 18, "clients": {c: local["validation_metrics"] for c in CLIENT_IDS}}
    payload = {"kind": kind, "model_name": "stfednet_gated", "config": config, "seed": 42,
               "smoke": False, "round": 18, "best_round": 18, "best_validation_wape": 10.,
               "client_ids": list(CLIENT_IDS), "data_signatures": {c: local["data_signature"] for c in CLIENT_IDS},
               "algorithm": "dp_fedavg" if kind == "best_dp_global" else "fedavg",
               "best_validation": validation, "current_validation": validation}
    filename = "best_dp_global.pt" if kind == "best_dp_global" else "best_global.pt"
    path = path.with_name(filename)
    latest_path = path.with_name(filename.replace("best", "latest"))
    torch.save(payload, path)
    torch.save(dict(payload, kind=kind.replace("best", "latest"), round=20), latest_path)
    assert final.inspect_run(path)["best"]["round"] == 18
    torch.save(dict(payload, kind=kind.replace("best", "latest"), round=19), latest_path)
    with pytest.raises(ValueError, match="incomplete"):
        final.inspect_run(path)


def test_missing_saved_scaler_rejected_not_refitted(setup_run):
    root, path, best, _ = setup_run
    best.pop("scaler")
    torch.save(best, path)
    torch.save(dict(best, kind="latest", epoch=30), path.with_name("latest.pt"))
    with pytest.raises(ValueError, match="refitting is prohibited"):
        final.build_manifest(root)
