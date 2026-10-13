"""G2.1 diagnostics are observational and analyses read validation files only."""

import csv
import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import Subset

from g2_client_before_diagnostics import client_update as legacy_update
from test_g2 import synthetic_clients, config
from scripts.compare_fedprox_mu import compare_mu, save_comparison
from scripts.analyze_fedavg_convergence import analyze_convergence
from src.fl.client import client_update
from src.fl.aggregation import fedprox_objective
from src.fl.common import CLIENT_IDS
from src.fl.diagnostics import DIAGNOSTIC_FIELDS, summarize_diagnostics
from src.fl.server import run_federated
from src.models.forecasting import build_model
from src.training.engine import load_checkpoint, seed_everything


def test_prox_diagnostic_is_actual_mu_scaled_parameter_distance():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.)
    anchor = {"weight": torch.zeros_like(model.weight)}
    huber = torch.tensor(3., requires_grad=True)
    total, prox = fedprox_objective(huber, model, anchor, .01, return_prox=True)
    assert prox.item() == pytest.approx(.01 / 2 * 4)
    assert total.item() == pytest.approx(3.02)
    assert total.item() == fedprox_objective(huber, model, anchor, .01).item()


@pytest.mark.parametrize("model_name", ["gru_attention", "stfednet_gated"])
@pytest.mark.parametrize("algorithm", ["fedavg", "fedprox"])
def test_diagnostics_preserve_pre_g21_parameters_and_loss(tmp_path, model_name, algorithm):
    settings, clients = config(tmp_path), synthetic_clients()
    settings.update(dropout=0.1, local_epochs=2)
    seed_everything(42, 1)
    state = build_model(model_name, dropout=0.1).state_dict()
    args = (state, clients["lv_39"], settings, model_name, 2, 0, torch.device("cpu"))
    before = legacy_update(*args, algorithm=algorithm, mu=0.01, max_train_batches=3)
    after = client_update(*args, algorithm=algorithm, mu=0.01, max_train_batches=3)
    assert before["mean_loss"] == after["mean_loss"]
    assert all(torch.equal(before["state_dict"][key], after["state_dict"][key]) for key in state)
    diag = after["diagnostics"]
    assert diag["total_loss"] == after["mean_loss"]
    assert diag["huber_loss"] + diag["prox_loss"] == pytest.approx(diag["total_loss"], abs=1e-7)
    assert (diag["prox_loss"] == 0) if algorithm == "fedavg" else (diag["prox_loss"] > 0)


def test_loss_window_weighting_and_batch_gradient_clip_statistics(tmp_path, monkeypatch):
    prepared, settings = synthetic_clients()["lv_39"], config(tmp_path)
    prepared.train = Subset(prepared.train, range(5))  # uneven final batch
    settings.update(local_epochs=2, gradient_clip=0.0001)
    original_clip = torch.nn.utils.clip_grad_norm_
    observed_norms, batch_losses = [], []
    import src.fl.client as client_module
    original_huber = client_module.masked_huber

    def record_clip(*args, **kwargs):
        norm = original_clip(*args, **kwargs)
        observed_norms.append(float(norm))
        return norm

    def record_huber(pred, y, mask, delta):
        loss = original_huber(pred, y, mask, delta)
        batch_losses.append((float(loss.detach()), y.shape[0]))
        return loss

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    monkeypatch.setattr(client_module, "masked_huber", record_huber)
    state = build_model("gru_attention", dropout=0).state_dict()
    update = client_update(state, prepared, settings, "gru_attention", 1, 0, torch.device("cpu"))
    diag = update["diagnostics"]
    assert diag["processed_windows"] == 10 and diag["batch_count"] == 6
    assert diag["huber_loss"] == sum(loss * n for loss, n in batch_losses) / 10
    assert diag["grad_norm_before_clip"] == sum(observed_norms) / 6
    assert diag["clipped_batch_count"] == 6 and diag["clip_fraction"] == 1


def test_server_diagnostics_use_distinct_window_and_batch_weights():
    clients = {"a": {"huber_loss": 1., "prox_loss": .1, "total_loss": 1.1,
                     "processed_windows": 10, "batch_count": 5, "grad_norm_before_clip": 2., "clipped_batch_count": 2},
               "b": {"huber_loss": 4., "prox_loss": .4, "total_loss": 4.4,
                     "processed_windows": 30, "batch_count": 10, "grad_norm_before_clip": 5., "clipped_batch_count": 8}}
    summary = summarize_diagnostics(clients)
    assert summary["mean_huber_loss"] == 3.25
    assert summary["mean_prox_loss"] == pytest.approx(.325)
    assert summary["mean_grad_norm_before_clip"] == 4.
    assert summary["clip_fraction"] == 10 / 15


@pytest.mark.parametrize("algorithm", ["fedavg", "fedprox"])
def test_old_checkpoint_schema_resume_and_new_diagnostics_repair(tmp_path, algorithm):
    clients, settings = synthetic_clients(), config(tmp_path)
    settings["dropout"] = 0.1
    full = run_federated(settings, "gru_attention", algorithm, run_dir=tmp_path / "full",
                         prepared_clients=clients, max_train_batches=2, max_val_batches=1)
    partial = run_federated(dict(settings, rounds=1), "gru_attention", algorithm, run_dir=tmp_path / "old",
                            prepared_clients=clients, max_train_batches=2, max_val_batches=1)
    old = load_checkpoint(partial / "latest_global.pt")
    for key in ("current_diagnostics", "best_diagnostics"):
        old.pop(key)
    for row in old["round_history"]:
        for key in (*DIAGNOSTIC_FIELDS, "client_diagnostics_json"):
            row.pop(key)
    torch.save(old, partial / "latest_global.pt")
    run_federated(settings, "gru_attention", algorithm, resume=partial / "latest_global.pt",
                  prepared_clients=clients, max_train_batches=2, max_val_batches=1)
    expected, actual = load_checkpoint(full / "latest_global.pt"), load_checkpoint(partial / "latest_global.pt")
    assert all(torch.equal(expected["model_state_dict"][key], actual["model_state_dict"][key]) for key in expected["model_state_dict"])
    assert expected["current_diagnostics"] == actual["current_diagnostics"]
    with (partial / "rounds.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["mean_prox_loss"] == "" and rows[1]["mean_prox_loss"] != ""
    result = json.loads((partial / "round_diagnostics.json").read_text(encoding="utf-8"))
    assert result["rounds"][0]["clients"] is None
    assert result["rounds"][0]["summary"]["mean_huber_loss"] is None
    assert len(result["rounds"][1]["clients"]) == 5
    # Another resume of a completed checkpoint repairs mixed old/new CSV schema.
    run_federated(settings, "gru_attention", algorithm, resume=partial / "latest_global.pt",
                  prepared_clients=clients, max_train_batches=2, max_val_batches=1)
    assert "GradNorm=" in (partial / "train.log").read_text(encoding="utf-8")


def fake_run(root, algorithm, model="gru_attention", mu=None, name="seed42_old", wape=20., completed=20):
    path = Path(root) / algorithm / model / name / "best_global.pt"
    path.parent.mkdir(parents=True)
    cfg = config(root, rounds=20)
    cfg["mu"] = mu if mu is not None else .01
    clients = {cid: {"wape_percent": wape} for cid in CLIENT_IDS}
    payload = {"kind": "best_global", "algorithm": algorithm, "model_name": model,
               "seed": 42, "mu": mu, "smoke": False, "test_evaluated": False,
               "client_ids": list(CLIENT_IDS), "data_signatures": {cid: "sig-" + cid for cid in CLIENT_IDS},
               "config": cfg, "round": 20, "best_round": 20, "best_validation_wape": wape,
               "current_validation": {"round": 20, "clients": clients, "client_equal_wape_percent": wape,
                                      "overall": {"wape_percent": wape + 1}, "test_evaluated": False}}
    torch.save(payload, path)
    torch.save(dict(payload, kind="latest_global", round=completed), path.with_name("latest_global.pt"))
    return path


def test_mu_grouping_latest_rule_smoke_exclusion_and_unknown_diagnostics(tmp_path):
    fake_run(tmp_path, "fedavg")
    old = fake_run(tmp_path, "fedprox", mu=.01, wape=10.)
    new = fake_run(tmp_path, "fedprox", mu=.01, name="seed42_new", wape=30.)
    os.utime(old.with_name("latest_global.pt"), ns=(100, 100))
    os.utime(new.with_name("latest_global.pt"), ns=(200, 200))
    for mu in (.001, .0001):
        fake_run(tmp_path, "fedprox", mu=mu, name=f"seed42_mu{mu}")
    smoke = fake_run(tmp_path, "fedprox", mu=.01, name="seed42_smoke", wape=1.)
    p = load_checkpoint(smoke)
    torch.save(dict(p, smoke=True), smoke)
    fake_run(tmp_path, "fedprox", mu=.00001, name="seed42_incomplete", completed=15)
    result = compare_mu(tmp_path)
    assert len(result["runs"]) == 4 and len(result["all_runs"]) == 5
    selected = next(r for r in result["runs"] if r["mu"] == .01)
    assert selected["client_equal_wape_percent"] == 30. and selected["diagnostics_status"] == "not recorded"
    assert selected["delta_vs_fedavg_percentage_points"] == 10.
    assert all(value is None for value in selected["diagnostics"].values())
    assert not result["missing_planned_experiments"]
    save_comparison(result, tmp_path / "comparison.json")
    assert (tmp_path / "comparison.csv").exists()


@pytest.mark.parametrize("mismatch", ["data_signatures", "gradient_clip", "local_epochs", "batch_size"])
def test_mu_comparison_rejects_data_or_training_protocol_mismatch(tmp_path, mismatch):
    fake_run(tmp_path, "fedavg")
    path = fake_run(tmp_path, "fedprox", mu=.001)
    p = load_checkpoint(path)
    if mismatch == "data_signatures":
        p[mismatch]["lv_39"] = "different-data"
    else:
        p["config"][mismatch] *= 2
    torch.save(p, path)
    torch.save(dict(p, kind="latest_global"), path.with_name("latest_global.pt"))
    with pytest.raises(ValueError, match=mismatch):
        compare_mu(tmp_path)


def test_final_round_best_alone_does_not_imply_non_convergence(tmp_path):
    path = fake_run(tmp_path, "fedavg")
    with path.with_name("rounds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "model_name", "algorithm", "smoke",
                                                   "client_equal_wape_percent", "client_wapes_json"])
        writer.writeheader()
        for round_index in range(1, 21):
            value = 20. + (20 - round_index) * .001  # tiny downward plateau
            writer.writerow({"round": round_index, "model_name": "gru_attention", "algorithm": "fedavg", "smoke": False,
                             "client_equal_wape_percent": value,
                             "client_wapes_json": json.dumps({cid: value for cid in CLIENT_IDS})})
    result = analyze_convergence(tmp_path)
    row = result["runs"][0]
    assert row["last_round_is_best"]
    assert row["judgment"] == "near plateau in this five-round window"
    assert len(row["rounds_15_to_20"]) == 6
    assert len(row["client_last_five"]) == 5
