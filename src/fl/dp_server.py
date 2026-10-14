"""Isolated G3.1 window-level DP-FedAvg coordinator; validation only."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import torch
import yaml

from src.fl.aggregation import fedavg_aggregate
from src.fl.common import (CLIENT_IDS, evaluate_clients, prepare_clients,
                           signature_map, validate_config)
from src.models.forecasting import build_model
from src.privacy.accountant import (ClientRDPAccountant,
                                    require_privacy_claim_backend)
from src.privacy.poisson import nominal_steps_per_epoch
from src.privacy.training import _cpu_copy, client_dp_update
from src.training.engine import (atomic_save, load_checkpoint,
                                 random_state, restore_random_state,
                                 seed_everything, select_device, write_json)


def read_g31_config(path="configs/g3_1.yaml", overrides=None):
    """Merge G3.1 settings over the frozen G2 data/model protocol."""
    from src.fl.common import read_g2_config

    g2 = read_g2_config("configs/g2.yaml")
    dp = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
    config = {**g2, **dp, **(overrides or {})}
    config["mu"] = 0.0  # present only because the shared G2 validator expects it
    return validate_dp_config(config)


def validate_dp_config(config):
    validate_config(config)
    require_privacy_claim_backend(str(config.get("privacy_mode", "research")))
    if config.get("noise_multiplier") is None or not math.isfinite(float(config["noise_multiplier"])) or float(config["noise_multiplier"]) < 0:
        raise ValueError("noise_multiplier must be supplied, finite and non-negative")
    if not math.isfinite(float(config["max_grad_norm"])) or float(config["max_grad_norm"]) <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if not 0 < float(config["delta"]) < 1:
        raise ValueError("delta must be in (0, 1)")
    if int(config["expected_batch_size"]) <= 0:
        raise ValueError("expected_batch_size must be positive")
    if config.get("max_train_batches") is not None:
        raise ValueError("DP training uses max_dp_steps_per_epoch, not G2 max_train_batches")
    if config.get("expected_train_records") is not None and int(config["expected_train_records"]) <= 0:
        raise ValueError("expected_train_records must be positive or null")
    if config.get("gradient_method") not in {"auto", "loop", "torch_func"}:
        raise ValueError("gradient_method must be auto, loop, or torch_func")
    if config.get("privacy_mode") not in {"research", "privacy_claim"}:
        raise ValueError("privacy_mode must be research or privacy_claim")
    if config.get("max_dp_steps_per_epoch") is not None and int(config["max_dp_steps_per_epoch"]) <= 0:
        raise ValueError("max_dp_steps_per_epoch must be positive or null")
    return config


def _seed_generators(client_ids, seed, device):
    samplers, noises = {}, {}
    for index, client_id in enumerate(client_ids):
        client_seed = int(seed) + 100_003 * (index + 1)
        sampler = torch.Generator(device="cpu").manual_seed(client_seed)
        noise = torch.Generator(device=device).manual_seed(client_seed + 71_003_003)
        samplers[client_id], noises[client_id] = sampler, noise
    return samplers, noises


def _capture_random_state(samplers, noises):
    dummy = torch.Generator(device="cpu")
    return {
        "framework": random_state(dummy),
        "samplers": {key: gen.get_state() for key, gen in samplers.items()},
        "noise_generators": {key: gen.get_state() for key, gen in noises.items()},
    }


def _restore_random_state(saved, samplers, noises):
    if set(saved["samplers"]) != set(samplers) or set(saved["noise_generators"]) != set(noises):
        raise ValueError("Checkpoint random-generator client keys do not match")
    dummy = torch.Generator(device="cpu")
    restore_random_state(saved["framework"], dummy)
    for key, gen in samplers.items():
        gen.set_state(saved["samplers"][key])
    for key, gen in noises.items():
        gen.set_state(saved["noise_generators"][key])


def _new_run_dir(config, model_name):
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns() % 1_000_000).zfill(6)
    sigma_tag = str(config["noise_multiplier"]).replace(".", "p")
    return (Path(config["output_root"]) / "dp_fedavg" / model_name /
            ("sigma" + sigma_tag + "_seed" + str(config["seed"]) + "_" + stamp))


def _csv_write(path, history):
    fields = ["round", "model_name", "train_loss", "train_selected_records", "client_equal_wape_percent",
              "overall_wape_percent", "client_metrics_json", "client_privacy_json",
              "client_sampling_json", "seconds", "is_best", "smoke", "test_evaluated"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in history)


def _config_compatible(saved, current):
    mutable = {"rounds", "cpu_threads", "output_root"}
    for key, value in saved.items():
        if key not in mutable and current.get(key) != value:
            raise ValueError("DP-FedAvg resume configuration mismatch: " + key)


def _write_diagnostics(run_dir, history):
    write_json(Path(run_dir) / "round_diagnostics.json", {
        "visibility": "internal_restricted_non_dp",
        "note": "Training diagnostics and validation results are not protected by DP-SGD publication guarantees.",
        "rounds": history})


def _accountant_map(config, prepared_clients, saved=None):
    output = {}
    for client_id, prepared in prepared_clients.items():
        expected = int(config["expected_batch_size"])
        if expected > len(prepared.train):
            raise ValueError("Expected Poisson batch is larger than client training data")
        q = expected / len(prepared.train)
        output[client_id] = ClientRDPAccountant(
            config["noise_multiplier"], q, config["max_grad_norm"], config["delta"],
            None if saved is None else saved.get(client_id))
        if saved is not None and client_id not in saved:
            raise ValueError("Missing per-client accountant checkpoint: " + client_id)
    return output


def run_dp_federated(config, model_name="stfednet_gated", run_dir=None,
                     resume=None, prepared_clients=None):
    """Run DP-SGD clients and FedAvg; the test split is never read."""
    if model_name not in {"stfednet_gated", "gru_attention"}:
        raise ValueError("G3.1 supports stfednet_gated and gru_attention")
    config = dict(config)
    validate_dp_config(config)
    seed_everything(int(config["seed"]), int(config["cpu_threads"]))
    device = select_device(config["device"])
    prepared_clients = prepared_clients or prepare_clients(config)
    if tuple(prepared_clients) != tuple(config["clients"]) or tuple(config["clients"]) != CLIENT_IDS:
        raise ValueError("G3.1 requires all five clients in the canonical order")
    expected_records = config.get("expected_train_records")
    if expected_records is not None:
        mismatches = {cid: len(prepared_clients[cid].train) for cid in prepared_clients
                      if len(prepared_clients[cid].train) != int(expected_records)}
        if mismatches:
            raise ValueError("Training-window count differs from the configured protocol: " +
                             json.dumps(mismatches, sort_keys=True))
    signatures = signature_map(prepared_clients)
    actual_method = config["gradient_method"]
    if actual_method == "auto":
        actual_method = "torch_func" if device.type == "cuda" else "loop"
    smoke = bool(config.get("max_dp_steps_per_epoch") is not None or
                 config.get("max_val_batches") is not None)
    samplers, noises = _seed_generators(config["clients"], config["seed"], device)
    model = build_model(model_name, config["hidden_dim"], config["dropout"],
                        config["horizon_hours"]).to(device)
    global_state = _cpu_copy(model.state_dict())
    run_dir = Path(resume).resolve().parent if resume else (Path(run_dir) if run_dir else _new_run_dir(config, model_name))
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest_dp_global.pt"
    if not resume and latest_path.exists():
        raise ValueError("Run directory already contains a DP checkpoint; use --resume")
    history, pending_updates = [], {}
    best_state, best_validation = None, None
    best_wape, best_round, completed_round = math.inf, 0, 0
    active_round, next_client_index, client_progress = None, 0, None
    saved_accounts = None
    best_accounts = None
    if resume:
        latest = load_checkpoint(resume)
        if latest.get("kind") != "latest_dp_global":
            raise ValueError("Resume requires latest_dp_global.pt")
        if latest.get("model_name") != model_name:
            raise ValueError("DP-FedAvg resume model mismatch")
        if latest.get("client_ids") != list(config["clients"]):
            raise ValueError("DP-FedAvg resume client mismatch")
        if latest.get("data_signatures") != signatures:
            raise ValueError("DP-FedAvg resume data signature mismatch")
        _config_compatible(latest["config"], config)
        if latest["gradient_method"] != actual_method:
            raise ValueError("DP resume requires the same actual gradient backend")
        global_state = latest["global_model_state_dict"]
        best_state = latest.get("best_model_state_dict")
        best_validation = latest.get("best_validation")
        best_accounts = latest.get("best_accountant_states")
        best_wape = float(latest["best_validation_wape"])
        best_round = int(latest["best_round"])
        completed_round = int(latest["completed_round"])
        active_round = latest.get("active_round")
        next_client_index = int(latest.get("next_client_index", 0))
        pending_updates = dict(latest.get("pending_round_updates", {}))
        client_progress = latest.get("client_progress")
        history = list(latest.get("round_history", []))
        saved_accounts = latest["accountant_states"]
        _restore_random_state(latest["rng_state"], samplers, noises)
        run_dir = Path(resume).resolve().parent
        _csv_write(run_dir / "rounds.csv", history)
        _write_diagnostics(run_dir, history)
    accounts = _accountant_map(config, prepared_clients, saved_accounts)

    if resume:
        client_keys = list(config["clients"])
        if not 0 <= next_client_index <= len(client_keys):
            raise ValueError("Invalid next DP client index")
        if set(pending_updates) != set(client_keys[:next_client_index]):
            raise ValueError("Pending DP updates do not match committed client order")
        if active_round is not None and int(active_round) != completed_round + 1:
            raise ValueError("Invalid active DP round")
        if completed_round > int(config["rounds"]):
            raise ValueError("Cannot resume with fewer rounds than already completed")
        for index, cid in enumerate(client_keys):
            nominal = nominal_steps_per_epoch(len(prepared_clients[cid].train), config["expected_batch_size"])
            limit = config.get("max_dp_steps_per_epoch")
            per_round = int(config["local_epochs"]) * (nominal if limit is None else min(nominal, int(limit)))
            expected_steps = completed_round * per_round
            if active_round is not None and index < next_client_index:
                expected_steps += per_round
            if client_progress is not None and index == next_client_index:
                if client_progress["client_id"] != cid:
                    raise ValueError("DP progress belongs to the wrong client")
                expected_steps += int(client_progress["next_step"])
            if accounts[cid].steps != expected_steps:
                raise ValueError("Accountant steps disagree with committed training progress for " + cid)

    config_out = dict(config, model_name=model_name, actual_gradient_method=actual_method,
                      data_signatures=signatures, client_ids=list(prepared_clients),
                      q_by_client={cid: acc.sample_rate for cid, acc in accounts.items()},
                      steps_per_epoch_by_client={cid: nominal_steps_per_epoch(
                          len(prepared_clients[cid].train), config["expected_batch_size"])
                          for cid in prepared_clients},
                      noise_source="ordinary_torch_prng_research_only",
                      privacy_claim_allowed=False, smoke=smoke, test_evaluated=False)
    write_json(run_dir / "config.json", config_out)

    def persist_checkpoint():
        payload = {
            "format_version": 1, "kind": "latest_dp_global", "model_name": model_name,
            "algorithm": "dp_fedavg", "global_model_state_dict": _cpu_copy(global_state),
            "best_model_state_dict": _cpu_copy(best_state) if best_state is not None else None,
            "round": completed_round, "completed_round": completed_round,
            "active_round": active_round, "next_client_index": next_client_index,
            "pending_round_updates": _cpu_copy(pending_updates),
            "client_progress": _cpu_copy(client_progress) if client_progress is not None else None,
            "best_round": best_round, "best_validation_wape": best_wape,
            "best_validation": best_validation,
            "best_accountant_states": best_accounts,
            "client_ids": list(prepared_clients), "data_signatures": signatures,
            "config": config, "seed": config["seed"], "noise_multiplier": config["noise_multiplier"],
            "max_grad_norm": config["max_grad_norm"], "delta": config["delta"],
            "gradient_method": actual_method, "privacy_mode": config["privacy_mode"],
            "randomness_security": "research_prng_not_secure",
            "rng_state": _capture_random_state(samplers, noises),
            "accountant_states": {cid: acc.state_dict() for cid, acc in accounts.items()},
            "round_history": history, "smoke": smoke, "test_evaluated": False,
            "privacy_claim_allowed": False,
            "optimizer_policy": "AdamW reinitialized for each client-round; step state checkpointed during an active client",
        }
        atomic_save(payload, latest_path)
        # Shareable protocol/accounting metadata contains no private diagnostics,
        # model-selection statistics, RNG seeds or reconstructible RNG states.
        write_json(run_dir / "privacy_accounting.json", {
            "visibility": "research_protocol_metadata_no_privacy_claim",
            "privacy_mode": config["privacy_mode"],
            "randomness_security": "ordinary_torch_prng_not_secure",
            "privacy_claim_allowed": False,
            "completed_round": completed_round,
            "active_round": active_round,
            "smoke": smoke,
            "clients": {cid: {
                "noise_multiplier": acc.noise_multiplier,
                "max_grad_norm": acc.max_grad_norm,
                "sample_rate": acc.sample_rate,
                "delta": acc.delta,
                "committed_update_steps": acc.steps,
                "theoretical_epsilon": acc.epsilon(),
                "status": acc.state_dict()["status"],
                "accountant_name": "opacus.RDPAccountant",
                "accountant_version": acc.state_dict()["accountant_version"],
            } for cid, acc in accounts.items()},
        })

    def repair_best_outputs():
        if best_state is None:
            return
        best_payload = {
            "format_version": 1, "kind": "best_dp_global", "model_name": model_name,
            "algorithm": "dp_fedavg", "model_state_dict": _cpu_copy(best_state),
            "round": best_round, "best_round": best_round,
            "best_validation_wape": best_wape, "best_validation": best_validation,
            "client_ids": list(prepared_clients), "data_signatures": signatures,
            "config": config, "seed": config["seed"],
            "noise_multiplier": config["noise_multiplier"],
            "max_grad_norm": config["max_grad_norm"], "delta": config["delta"],
            "gradient_method": actual_method, "privacy_mode": config["privacy_mode"],
            "accountant_states": best_accounts,
            "randomness_security": "research_prng_not_secure",
            "privacy_claim_allowed": False, "smoke": smoke, "test_evaluated": False,
        }
        atomic_save(best_payload, run_dir / "best_dp_global.pt")
        write_json(run_dir / "best_clients_validation.json", {
            "algorithm": "dp_fedavg", "model_name": model_name,
            "best_round": best_round, "best_validation_wape": best_wape,
            "data_signatures": signatures, "client_ids": list(prepared_clients),
            "privacy_claim_allowed": False, "smoke": smoke,
            "test_evaluated": False, "visibility": "internal_restricted_non_dp",
            **best_validation})

    def log(message):
        print(message, flush=True)
        with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log("Algorithm=DP-FedAvg | Model=" + model_name + " | Clients=5/5 | Device=" + str(device) +
        " | ExpectedBatch=" + str(config["expected_batch_size"]) +
        " | q range=" + str([round(acc.sample_rate, 8) for acc in accounts.values()]) +
        " | C=" + str(config["max_grad_norm"]) + " | sigma=" + str(config["noise_multiplier"]) +
        " | delta=" + str(config["delta"]) + " | GradientMethod=" + actual_method +
        " | Mode=" + str(config["privacy_mode"]) + " | Rounds=" + str(config["rounds"]))
    log("[Privacy] Ordinary PyTorch PRNG; research only; no privacy claim is enabled")
    if smoke:
        log("[Smoke] Limited Poisson steps and/or validation batches; smoke=true; test split not accessed")
    if resume:
        repair_best_outputs()

    if active_round is None and completed_round < int(config["rounds"]):
        active_round = completed_round + 1
        next_client_index = 0
        pending_updates = {}
        client_progress = None
        persist_checkpoint()

    while active_round is not None and int(active_round) <= int(config["rounds"]):
        round_started = time.perf_counter()
        round_start_state = _cpu_copy(global_state)
        for client_index in range(next_client_index, len(config["clients"])):
            client_id = config["clients"][client_index]
            current_progress = client_progress if client_id == (client_progress or {}).get("client_id") else None

            def save_step_progress(progress, account_state):
                nonlocal client_progress
                client_progress = progress
                # The atomic latest checkpoint is the commit point for a local
                # DP optimizer step and its matching accountant increment.
                persist_checkpoint()

            update = client_dp_update(
                round_start_state, prepared_clients[client_id], config, model_name, device,
                accounts[client_id], samplers[client_id], noises[client_id], actual_method,
                config.get("max_dp_steps_per_epoch"), current_progress, save_step_progress)
            pending_updates[client_id] = update
            client_progress = None
            next_client_index = client_index + 1
            persist_checkpoint()
        if set(pending_updates) != set(config["clients"]):
            raise RuntimeError("Incomplete five-client DP round cannot be aggregated")
        global_state = fedavg_aggregate(
            [pending_updates[cid]["state_dict"] for cid in config["clients"]],
            [pending_updates[cid]["sample_count"] for cid in config["clients"]])
        model.load_state_dict(global_state, strict=True)
        validation = evaluate_clients(model, prepared_clients, config, device,
                                      config.get("max_val_batches"))
        equal_wape = validation["client_equal_wape_percent"]
        if equal_wape is None:
            raise ValueError("Undefined five-client equal validation WAPE")
        improved = equal_wape < best_wape
        if improved:
            best_wape, best_round = float(equal_wape), int(active_round)
            best_state = _cpu_copy(global_state)
            best_validation = {"round": active_round, **validation, "smoke": smoke,
                               "test_evaluated": False}
            best_accounts = {cid: acc.state_dict() for cid, acc in accounts.items()}
        sampling = {cid: pending_updates[cid]["diagnostics"] for cid in config["clients"]}
        selected_count = sum(row["selected_record_draws"] for row in sampling.values())
        train_loss = (sum((row["mean_huber_loss_per_selected_record"] or 0.0) *
                          row["selected_record_draws"] for row in sampling.values()) / selected_count
                      if selected_count else None)
        history.append({
            "round": int(active_round), "model_name": model_name,
            "train_loss": train_loss,
            "train_selected_records": sum(row["diagnostics"]["selected_record_draws"]
                                           for row in pending_updates.values()),
            "client_equal_wape_percent": float(equal_wape),
            "overall_wape_percent": validation["overall"]["wape_percent"],
            "client_metrics_json": json.dumps(validation["clients"], sort_keys=True),
            "client_privacy_json": json.dumps({cid: accounts[cid].state_dict()
                                                for cid in config["clients"]}, sort_keys=True),
            "client_sampling_json": json.dumps(sampling, sort_keys=True),
            "seconds": time.perf_counter() - round_started,
            "is_best": improved, "smoke": smoke, "test_evaluated": False,
        })
        completed_round = int(active_round)
        active_round, next_client_index, pending_updates, client_progress = None, 0, {}, None
        _csv_write(run_dir / "rounds.csv", history)
        _write_diagnostics(run_dir, history)
        persist_checkpoint()
        if improved:
            repair_best_outputs()
            log("[Checkpoint] New best DP-FedAvg model saved")
        epsilons = {cid: accounts[cid].epsilon() for cid in config["clients"]}
        log("Round " + str(completed_round).zfill(2) + "/" + str(config["rounds"]) +
            " | TrainLoss=" + (format(train_loss, ".6f") if train_loss is not None else "not_observed_empty_draws") +
            " | AvgValWAPE=" + format(float(equal_wape), ".4f") +
            "% | OverallValWAPE=" + format(float(validation["overall"]["wape_percent"]), ".4f") +
            "% | Best=" + format(best_wape, ".4f") + "% | ClientEpsilon=" +
            json.dumps(epsilons, sort_keys=True) + " | Time=" +
            format(history[-1]["seconds"], ".2f") + "s")
        if completed_round < int(config["rounds"]):
            active_round = completed_round + 1
            next_client_index, pending_updates, client_progress = 0, {}, None
            persist_checkpoint()
    log("[Run] DP-FedAvg validation checkpoints and restricted diagnostics: " + str(run_dir))
    return run_dir
