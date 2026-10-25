"""Frozen, inference-only final suite. Planning never loads source data.

Global checkpoints deliberately omit scalers. Recover an existing scaler from
a local checkpoint with the identical data signature and preprocessing protocol;
never call StandardScaler.fit or construct train/validation window datasets.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loader import StandardScaler, WindowDataset, discover_grids, load_grid
from src.fl.common import CLIENT_IDS, pooled_and_equal_metrics
from src.models.forecasting import build_model
from src.training.engine import data_signature, load_checkpoint, seed_everything, select_device
from src.training.metrics import MetricAccumulator


PROJECT_NAME = "基于时空注意力机制与联邦学习的配电网负荷预测"
PREPROCESS = ("input_hours", "horizon_hours", "train_ratio", "val_ratio", "data_root")
SCIENTIFIC = ("input_hours", "horizon_hours", "train_ratio", "val_ratio", "hidden_dim",
              "dropout", "batch_size", "learning_rate", "weight_decay", "huber_delta",
              "mape_threshold_kw", "seed")
METRICS = ("mae_kw", "rmse_kw", "wape_percent", "mape_percent", "mape_valid_fraction")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def scaler_digest(saved):
    digest = hashlib.sha256()
    for key in ("mean", "scale"):
        array = saved[key].cpu().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    digest.update(canonical({k: saved[k] for k in ("fit_start", "fit_end")}))
    return digest.hexdigest()


def previously_tested(path, checkpoint):
    if checkpoint.get("test_evaluated"):
        return True
    # Old local evaluation writes a sidecar without changing the checkpoint.
    for record_path in path.parent.rglob("metrics.json"):
        if "smoke_test" in record_path.parts:
            continue
        record = json.loads(record_path.read_text(encoding="utf-8-sig"))
        if not record.get("smoke", False) and record.get("checkpoint"):
            if Path(record["checkpoint"]).resolve() == path.resolve():
                return True
    return False


def check_protocol(payload):
    c = payload["config"]
    expected = {"input_hours": 48, "horizon_hours": 1, "train_ratio": .7,
                "val_ratio": .15, "hidden_dim": 32, "dropout": .1, "batch_size": 32,
                "learning_rate": .001, "weight_decay": .0001, "seed": 42,
                "mape_threshold_kw": .1, "huber_delta": 1.0}
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError("outside frozen main protocol: " + key)
    if payload.get("seed") != 42 or payload.get("smoke"):
        raise ValueError("wrong seed or smoke")
    for key in ("max_train_batches", "max_val_batches", "max_dp_steps_per_epoch"):
        if c.get(key) is not None:
            raise ValueError("limited batch/step run: " + key)


def inspect_run(path):
    """Check complete Best/Latest metadata, without loading any dataset."""
    best = load_checkpoint(path)
    check_protocol(best)
    if previously_tested(path, best):
        raise ValueError("already formally tested")
    kind = best["kind"]
    latest_path = path.with_name(path.name.replace("best", "latest", 1))
    if not latest_path.exists():
        raise ValueError("missing Latest completion evidence")
    latest = load_checkpoint(latest_path)
    check_protocol(latest)
    if latest.get("kind") != kind.replace("best", "latest", 1):
        raise ValueError("Latest checkpoint kind mismatch")
    for key in ("model_name", "seed", "config", "best_validation_wape"):
        if best[key] != latest[key]:
            raise ValueError("Best/Latest mismatch: " + key)
    if kind == "best":
        for key in ("client_id", "data_signature", "best_epoch"):
            if best[key] != latest[key]:
                raise ValueError("Best/Latest mismatch: " + key)
        c = best["config"]
        if c["epochs"] != 30 or c["patience"] != 8:
            raise ValueError("outside the frozen local epoch/early-stop budget")
        complete = (latest["epoch"] == c["epochs"] or
                    latest["stale_epochs"] >= c["patience"] or
                    (best["model_name"] == "persistence" and latest["epoch"] == 1))
        if not complete or best["epoch"] != best["best_epoch"]:
            raise ValueError("incomplete local training")
        if best["validation_metrics"]["wape_percent"] != best["best_validation_wape"]:
            raise ValueError("local validation WAPE mismatch")
    elif kind in {"best_global", "best_dp_global"}:
        c = best["config"]
        if c["rounds"] != 20 or c["local_epochs"] != 1:
            raise ValueError("not the 20-round/one-local-epoch main experiment")
        if latest.get("completed_round", latest["round"]) != 20 or latest["round"] != 20:
            raise ValueError("incomplete global training")
        for key in ("client_ids", "data_signatures", "best_round", "algorithm"):
            if best[key] != latest[key]:
                raise ValueError("Best/Latest mismatch: " + key)
        if best["client_ids"] != list(CLIENT_IDS) or set(best["data_signatures"]) != set(CLIENT_IDS):
            raise ValueError("missing five-client signatures")
        val = best.get("current_validation", best["best_validation"])
        if val.get("smoke") or val.get("test_evaluated"):
            raise ValueError("not a formal validation summary")
        equal = np.mean([val["clients"][cid]["wape_percent"] for cid in CLIENT_IDS])
        if (best["round"] != best["best_round"] or val["round"] != best["best_round"]
                or not math.isclose(equal, best["best_validation_wape"], rel_tol=1e-10)):
            raise ValueError("global validation/Best round mismatch")
        if kind == "best_dp_global":
            for key, value in {"expected_batch_size": 32, "expected_train_records": 6084,
                               "max_grad_norm": 1., "delta": 1e-5,
                               "privacy_mode": "research"}.items():
                if c.get(key) != value:
                    raise ValueError("not the frozen DP main protocol: " + key)
            if latest.get("active_round") is not None or latest.get("client_progress"):
                raise ValueError("unfinished DP round")
    else:
        raise ValueError("unsupported checkpoint kind")
    return {"path": path, "best": best, "latest_path": latest_path,
            "completion_key": (latest_path.stat().st_mtime_ns, str(path.resolve()))}


def privacy_summary(payload):
    if payload["kind"] != "best_dp_global":
        return None
    result = {}
    sigma = payload["noise_multiplier"]
    for cid in CLIENT_IDS:
        saved = payload["accountant_states"][cid]
        if (saved["steps"] != payload["best_round"] * 191
                or saved["noise_multiplier"] != sigma
                or saved["max_grad_norm"] != 1 or saved["delta"] != 1e-5
                or not math.isclose(saved["sample_rate"], 32 / 6084)):
            raise ValueError("Best accountant metadata inconsistent")
        from src.privacy.accountant import ClientRDPAccountant
        accountant = ClientRDPAccountant(sigma, 32 / 6084, 1., 1e-5, state=saved)
        epsilon = accountant.epsilon()
        recorded = saved["theoretical_epsilon"]
        if (epsilon is None) != (recorded is None) or (epsilon is not None and
                not math.isclose(epsilon, recorded, rel_tol=1e-10)):
            raise ValueError("Best accountant epsilon inconsistent")
        result[cid] = {k: saved[k] for k in ("steps", "sample_rate", "delta",
                      "max_grad_norm", "noise_multiplier", "accountant_name", "accountant_version")}
        result[cid]["theoretical_epsilon"] = epsilon
    return {"mode": "research", "privacy_claim_allowed": False,
            "status": "non_dp_control" if sigma == 0 else "conditional_theoretical_window_rdp",
            "best_checkpoint_accounting": result,
            "boundary": "Fixed preprocessing/topology/mask; ordinary PRNG; non-DP validation selection and test publication. No end-to-end/user-level guarantee."}


def build_manifest(root="."):
    root = Path(root).resolve()
    groups, skipped, missing = {}, [], []
    specs = [("local_gru_attention", "results/g1/gru_attention", "best.pt"),
             ("local_stfednet_gated", "results/g1_2/stfednet_gated", "best.pt"),
             ("persistence", "results/g1/persistence", "best.pt")]
    specs += [(a, f"results/g2/{a}/stfednet_gated", "best_global.pt")
              for a in ("centralized", "fedavg", "fedprox")]
    specs += [("dp", "results/g3_1/dp_fedavg/stfednet_gated", "best_dp_global.pt")]
    for label, folder, filename in specs:
        for path in sorted((root / folder).rglob(filename)):
            try:
                run = inspect_run(path)
                b = run["best"]
                if label.startswith("local_") or label == "persistence":
                    if b["model_name"] != (label.removeprefix("local_")) or b["client_id"] not in CLIENT_IDS:
                        raise ValueError("model/client mismatch")
                    group = (label, b["client_id"])
                else:
                    if b["model_name"] != "stfednet_gated":
                        raise ValueError("global model mismatch")
                    if label == "fedprox" and b.get("mu") != .001:
                        raise ValueError("excluded mu: G2.1 validation selected gated mu=0.001")
                    if label == "dp":
                        sigma = b["noise_multiplier"]
                        if sigma not in (0., .5, 1.):
                            raise ValueError("outside final noise multiplier list")
                        label_for_run = f"dp_sigma{sigma:g}"
                    else:
                        if b.get("algorithm") != label:
                            raise ValueError("algorithm mismatch")
                        label_for_run = label
                    group = (label_for_run, "all")
                groups.setdefault(group, []).append(run)
            except (ValueError, KeyError, FileNotFoundError) as error:
                skipped.append({"checkpoint": relative(path, root), "reason": str(error)})
    selected = {group: max(runs, key=lambda r: r["completion_key"]) for group, runs in groups.items()}
    for group, runs in groups.items():
        for run in runs:
            if run is not selected[group]:
                skipped.append({"checkpoint": relative(run["path"], root),
                                "reason": "older completed repeat; not selected by validation WAPE"})
    for label, _, _ in specs[:3]:
        for cid in CLIENT_IDS:
            if (label, cid) not in selected:
                missing.append({"method": label, "client_id": cid, "reason": "no eligible complete untested checkpoint"})
    for label in ("centralized", "fedavg", "fedprox", "dp_sigma0", "dp_sigma0.5", "dp_sigma1"):
        if (label, "all") not in selected:
            missing.append({"method": label, "client_id": "all", "reason": "no eligible complete untested checkpoint"})
    entries = []
    hashes = {}
    def file_ref(path):
        key = relative(path, root)
        hashes.setdefault(key, sha256(path))
        return {"path": key, "sha256": hashes[key]}
    for (method, client), run in sorted(selected.items()):
        b = run["best"]
        for cid in (CLIENT_IDS if client == "all" else (client,)):
            signature = b["data_signature"] if client != "all" else b["data_signatures"][cid]
            donors = [v for (_, donor_client), v in selected.items()
                      if donor_client == cid and "scaler" in v["best"]
                      and v["best"]["data_signature"] == signature
                      and all(v["best"]["config"].get(k) == b["config"].get(k) for k in PREPROCESS)]
            if not donors:
                raise ValueError(f"No saved matching scaler for {method}/{cid}; refitting is prohibited")
            if len({scaler_digest(d["best"]["scaler"]) for d in donors}) != 1:
                raise ValueError("Matching saved scalers disagree: " + cid)
            donor = run if "scaler" in b else max(donors, key=lambda r: r["completion_key"])
            config = b["config"]
            # All final methods must share the same scientific settings and client signature.
            reference = donors[0]["best"]
            if any(config.get(k) != reference["config"].get(k) for k in SCIENTIFIC):
                raise ValueError("Cross-method scientific protocol mismatch")
            validation = (b["validation_metrics"] if client != "all" else
                          b.get("current_validation", b["best_validation"])["clients"][cid])
            entries.append({"entry_id": f"{method}__{cid}", "method": method,
                            "model_name": b["model_name"], "client_id": cid, "seed": b["seed"],
                            "kind": b["kind"], "checkpoint": file_ref(run["path"]),
                            "completion_evidence": file_ref(run["latest_path"]),
                            "completed_epochs_or_rounds": (load_checkpoint(run["latest_path"]).get("epoch")
                                                          if client != "all" else 20),
                            "data_signature": signature, "config": config,
                            "best_epoch_or_round": b.get("epoch", b.get("best_round")),
                            "validation_metrics": validation, "mu": b.get("mu"),
                            "privacy": privacy_summary(b),
                            "scaler_source": {**file_ref(donor["path"]),
                                              "scaler_sha256": scaler_digest(donor["best"]["scaler"]),
                                              "fit_start": donor["best"]["scaler"]["fit_start"],
                                              "fit_end": donor["best"]["scaler"]["fit_end"]}})
    for cid in CLIENT_IDS:
        if len({e["data_signature"] for e in entries if e["client_id"] == cid}) > 1:
            raise ValueError("Cross-method client data signature mismatch")
    manifest = {"schema_version": 1, "project_name": PROJECT_NAME,
                "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_test",
                "selection": "seed42; latest completed eligible formal repeat by Latest mtime_ns then path; no test-based selection",
                "fedprox_selection": "mu=0.001 fixed by completed G2.1 gated validation sensitivity selection",
                "excluded_roots": ["results/g3_3_batch64", "results/g3_4_lr_pilot", "results/g3_pilot"],
                "plan_reads_source_data": False, "entries": entries, "missing": missing, "skipped": skipped}
    manifest["freeze_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    return manifest


def verify_manifest(manifest, root):
    content = {k: v for k, v in manifest.items() if k != "freeze_sha256"}
    if hashlib.sha256(canonical(content)).hexdigest() != manifest["freeze_sha256"]:
        raise ValueError("Frozen manifest was changed")
    checked = set()
    for entry in manifest["entries"]:
        for key in ("checkpoint", "completion_evidence", "scaler_source"):
            ref = entry[key]
            path = (Path(root) / ref["path"]).resolve()
            path.relative_to(Path(root).resolve())
            if path not in checked:
                if sha256(path) != ref["sha256"]:
                    raise ValueError("Frozen checkpoint changed: " + str(path))
                checked.add(path)
    if len({e["entry_id"] for e in manifest["entries"]}) != len(manifest["entries"]):
        raise ValueError("Duplicate evaluation entry")


def prepare_test_only(entry, root):
    """Recover saved scaler; load raw arrays for signature, and build TEST only."""
    config, cid = entry["config"], entry["client_id"]
    donor = load_checkpoint(Path(root) / entry["scaler_source"]["path"])
    if donor["client_id"] != cid or donor["data_signature"] != entry["data_signature"]:
        raise ValueError("Scaler donor client/data mismatch")
    if any(donor["config"].get(k) != config.get(k) for k in PREPROCESS):
        raise ValueError("Scaler donor preprocessing mismatch")
    saved = donor["scaler"]
    if scaler_digest(saved) != entry["scaler_source"]["scaler_sha256"]:
        raise ValueError("Scaler digest mismatch")
    paths = dict(discover_grids(Path(root) / config["data_root"]))
    grid = load_grid(paths[cid], cid)
    total = len(grid.timestamps)
    train_end, val_end = int(total * config["train_ratio"]), int(total * (config["train_ratio"] + config["val_ratio"]))
    prepared = SimpleNamespace(grid=grid, split_indices=(train_end, val_end))
    if data_signature(prepared) != entry["data_signature"]:
        raise ValueError("Source data/topology/IDs/split changed")
    mean, scale = saved["mean"].numpy().copy(), saved["scale"].numpy().copy()
    if (mean.shape != (grid.load_matrix.shape[1],) or scale.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid saved scaler")
    scaler = StandardScaler(mean, scale, np.datetime64(saved["fit_start"]), np.datetime64(saved["fit_end"]))
    if scaler.fit_start != grid.timestamps[0] or scaler.fit_end != grid.timestamps[train_end - 1]:
        raise ValueError("Saved scaler is not training-period-only")
    context_start = val_end - config["input_hours"]
    values = scaler.transform(grid.load_matrix[context_start:])
    starts = np.arange(total - val_end - config["horizon_hours"] + 1, dtype=np.int64)
    test = WindowDataset(values, grid.timestamps[context_start:], starts,
                         config["input_hours"], config["horizon_hours"], grid.load_mask).as_torch()
    return SimpleNamespace(grid=grid, scaler=scaler, test=test)


@torch.inference_mode()
def evaluate_entry(entry, root, device):
    checkpoint = load_checkpoint(Path(root) / entry["checkpoint"]["path"])
    if previously_tested(Path(root) / entry["checkpoint"]["path"], checkpoint):
        raise ValueError("Checkpoint was formally tested outside this frozen suite")
    config = entry["config"]
    if checkpoint["config"] != config:
        raise ValueError("Frozen config mismatch")
    prepared = prepare_test_only(entry, root)
    model = build_model(entry["model_name"], config["hidden_dim"], config["dropout"], config["horizon_hours"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    metrics = MetricAccumulator(config["mape_threshold_kw"])
    edge_index = torch.as_tensor(prepared.grid.edge_index, device=device, dtype=torch.long)
    for x, y in DataLoader(prepared.test, batch_size=config["batch_size"], shuffle=False, num_workers=0):
        prediction = model(x.to(device), edge_index).cpu().numpy()
        metrics.update(prepared.scaler.inverse_transform(y.numpy()),
                       prepared.scaler.inverse_transform(prediction), prepared.grid.load_mask)
    if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise RuntimeError("Inference changed model state")
    return {"entry_id": entry["entry_id"], "method": entry["method"],
            "model_name": entry["model_name"], "client_id": entry["client_id"],
            "seed": entry["seed"], "checkpoint": entry["checkpoint"],
            "best_epoch_or_round": entry["best_epoch_or_round"], "config": config,
            "data_signature": entry["data_signature"], "scaler_source": entry["scaler_source"],
            "validation_metrics": entry["validation_metrics"], "privacy": entry["privacy"],
            "metrics": metrics.compute(), "statistics": metrics.statistics(),
            "test_windows": len(prepared.test), "test_target_start": str(prepared.test.target_timestamps[0, 0]),
            "test_target_end": str(prepared.test.target_timestamps[-1, -1]),
            "model_state_unchanged": True, "smoke": False, "split": "test"}


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record["method"], []).append(record)
    summaries = {}
    for method, rows in groups.items():
        equal, pooled = pooled_and_equal_metrics(rows)
        summaries[method] = {"client_count": len(rows), "complete_five_clients": len(rows) == 5,
                             "client_equal_wape_percent": equal, "overall": pooled,
                             "clients": {r["client_id"]: r["metrics"] for r in rows},
                             "privacy": rows[0]["privacy"]}
    return summaries


def save_results(output, manifest, records):
    output = Path(output)
    result = {"project_name": PROJECT_NAME, "manifest_sha256": manifest["freeze_sha256"],
              "split": "test", "smoke": False, "selection_frozen_before_test": True,
              "privacy_claim_allowed": False, "records": records, "methods": summarize(records),
              "missing": manifest["missing"], "prediction_arrays_saved": False,
              "warning": "Published test aggregates are non-DP. Research RDP is conditional and excludes preprocessing/model selection/public metric releases."}
    atomic_json(output / "final_test_metrics.json", result)
    fields = ["method", "model_name", "client_id", "seed", "best_epoch_or_round", "test_windows", *METRICS,
              "theoretical_epsilon", "delta", "noise_multiplier", "checkpoint"]
    with (output / "final_test_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            private = (row["privacy"] or {}).get("best_checkpoint_accounting", {}).get(row["client_id"], {})
            writer.writerow({**{k: row.get(k) for k in fields}, **{k: row["metrics"][k] for k in METRICS},
                             **{k: private.get(k) for k in ("theoretical_epsilon", "delta", "noise_multiplier")},
                             "checkpoint": row["checkpoint"]["path"]})
    lines = ["# 正式测试汇总", "", PROJECT_NAME, "", f"冻结清单 SHA256：`{manifest['freeze_sha256']}`。",
             "", "所有模型在读取测试切分前已冻结；本次只做推理，无训练、无Scaler重拟合、无测试后调参。",
             "", "| 方法 | 客户端数 | 等权WAPE (%) | Overall WAPE (%) |",
             "|---|---:|---:|---:|"]
    for method, group in result["methods"].items():
        lines.append(f"| {method} | {group['client_count']} | {group['client_equal_wape_percent']:.6f} | {group['overall']['wape_percent']:.6f} |")
    lines += ["", "Persistence只有lv_39，不能将其单客户端结果当作五网汇总。完整逐客户端指标、MAPE有效比例、配置和Checkpoint路径见同目录CSV/JSON。",
              "", "等权WAPE是客户端WAPE的算术平均；Overall WAPE按全部有效观测的绝对误差与真实负荷绝对值求比。",
              "", "DP组均为research模式；σ=0为非DP控制。ε来自Best轮次的各客户端会计状态，不是Latest20轮状态。",
              "训练集Scaler、固定拓扑/mask、重叠窗口、重复曲线、非DP验证选优与公开测试指标不包含在端到端隐私保证中。",
              "", "## 各客户端正式测试", "", "| 方法 | 客户端 | MAE (kW) | RMSE (kW) | WAPE (%) | MAPE (%) | MAPE有效比例 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in records:
        m = row["metrics"]
        lines.append(f"| {row['method']} | {row['client_id']} | {m['mae_kw']:.6f} | {m['rmse_kw']:.6f} | {m['wape_percent']:.6f} | {m['mape_percent']:.6f} | {m['mape_valid_fraction']:.6f} |")
    lines += ["", "## Best轮次理论窗口级会计", "",
              "以下ε为research条件化理论值，不涵盖预处理、验证选优、公开测试统计或完整20轮发布轨迹。",
              "", "| 方法 | Best轮次 | 每客户端步数 | 理论ε | δ |", "|---|---:|---:|---:|---:|"]
    for method in result["methods"]:
        row = next(r for r in records if r["method"] == method)
        if row["privacy"] is not None:
            account = row["privacy"]["best_checkpoint_accounting"][row["client_id"]]
            eps = account["theoretical_epsilon"]
            eps_text = f"{eps:.6f}" if eps is not None else "无有限DP预算（σ=0）"
            lines.append(f"| {method} | {row['best_epoch_or_round']} | {account['steps']} | {eps_text} | {account['delta']:g} |")
    (output / "FINAL_TEST_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def execute_manifest(manifest, root=".", output="results/final_test", device="auto"):
    root = Path(root).resolve()
    output = root / output
    output.mkdir(parents=True, exist_ok=True)
    verify_manifest(manifest, root)  # Validate all files before the first TEST read.
    if not manifest["entries"]:
        raise ValueError("No eligible frozen evaluation entries")
    lock_path = output / "execution.lock"
    state_path = output / "execution_state.json"
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    try:
        state = (json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else
                 {"manifest_sha256": manifest["freeze_sha256"], "completed": {}, "active": None})
        if state["manifest_sha256"] != manifest["freeze_sha256"]:
            raise ValueError("Execution ledger belongs to a different frozen manifest")
        if state["active"]:
            raise RuntimeError("An evaluation was interrupted before commit; do not automatically repeat: " + state["active"])
        selected_device = select_device(device)
        seed_everything(42)
        for entry in manifest["entries"]:
            key = entry["entry_id"]
            if key in state["completed"]:
                continue
            state["active"] = key
            atomic_json(state_path, state)
            record = evaluate_entry(entry, root, selected_device)
            # Confirm the source weights were never modified before committing the result.
            if sha256(root / entry["checkpoint"]["path"]) != entry["checkpoint"]["sha256"]:
                raise RuntimeError("Checkpoint changed during evaluation")
            state["completed"][key] = record
            state["active"] = None
            atomic_json(state_path, state)
            print(f"[FinalTest] {key} | windows={record['test_windows']} | WAPE={record['metrics']['wape_percent']:.6f}%", flush=True)
        verify_manifest(manifest, root)
        records = [state["completed"][e["entry_id"]] for e in manifest["entries"]]
        return save_results(output, manifest, records)
    finally:
        lock_path.unlink()
