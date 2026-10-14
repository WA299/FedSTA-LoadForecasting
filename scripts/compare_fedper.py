"""Compare validation-only FedPer runs with existing G2 baselines."""

import argparse
import csv
import math
from pathlib import Path

from scripts.compare_g2 import add_local_only
from scripts.g21_results import check_matched_protocol, scan_runs, select_latest
from src.fl.common import CLIENT_IDS, MODEL_NAMES
from src.fl.fedper import combine_fedper_state
from src.models.forecasting import build_model
from src.training.engine import load_checkpoint, write_json


FEDPROX_MU = {"stfednet_gated": 0.001, "gru_attention": 0.0001}
SELECTION = "latest completed formal FedPer run by Latest checkpoint mtime_ns then path; never by validation score"


def _scan_fedper(root, model_name, seed, rounds=20):
    runs, skipped = [], []
    for path in sorted(Path(root).glob(f"{model_name}/seed*/best_fedper.pt")):
        best = load_checkpoint(path)
        latest_path = path.with_name("latest_fedper.pt")
        reason = None
        if best.get("kind") != "best_fedper" or best.get("algorithm") != "fedper":
            reason = "checkpoint kind/algorithm mismatch"
        elif best.get("model_name") != model_name or best.get("seed") != seed:
            reason = "model/seed mismatch"
        elif best.get("smoke") or best.get("config", {}).get("max_train_batches") or best.get("config", {}).get("max_val_batches"):
            reason = "smoke or limited batches"
        elif best["config"].get("rounds") != rounds:
            reason = "different configured round budget"
        elif not latest_path.exists():
            reason = "missing latest checkpoint"
        if reason:
            skipped.append({"checkpoint": str(path.resolve()), "reason": reason})
            continue
        latest = load_checkpoint(latest_path)
        if latest.get("kind") != "latest_fedper" or latest.get("algorithm") != "fedper":
            raise ValueError("FedPer Latest kind/algorithm mismatch")
        check_matched_protocol(latest, best)
        if latest.get("round") != rounds:
            skipped.append({"checkpoint": str(path.resolve()), "reason": "incomplete round budget"})
            continue
        if latest.get("data_signatures") != best.get("data_signatures") or latest.get("client_ids") != list(CLIENT_IDS):
            raise ValueError("FedPer Best/Latest client or data signature mismatch")
        for payload in (best, latest):
            if payload.get("test_evaluated") or payload["current_validation"].get("test_evaluated"):
                raise ValueError("FedPer checkpoint indicates test evaluation")
            if payload.get("smoke") or payload["current_validation"].get("smoke"):
                raise ValueError("FedPer Best/Latest smoke flag mismatch")
            if set(payload.get("head_state_dicts", {})) != set(CLIENT_IDS):
                raise ValueError("FedPer checkpoint is missing personalized heads")
            if set(payload["data_signatures"]) != set(CLIENT_IDS) or not all(
                    isinstance(value, str) and value for value in payload["data_signatures"].values()):
                raise ValueError("Missing FedPer data signatures")
            model = build_model(model_name, payload["config"]["hidden_dim"], payload["config"]["dropout"])
            for head in payload["head_state_dicts"].values():
                model.load_state_dict(combine_fedper_state(payload["shared_state_dict"], head), strict=True)
        validation = best["current_validation"]
        if validation.get("round") != best.get("best_round") or best["round"] != best["best_round"]:
            raise ValueError("FedPer validation does not match its selected round")
        if set(validation["clients"]) != set(CLIENT_IDS):
            raise ValueError("Incomplete FedPer validation clients")
        equal = sum(validation["clients"][cid]["wape_percent"] for cid in CLIENT_IDS) / 5
        if not math.isclose(equal, validation["client_equal_wape_percent"], rel_tol=1e-10) or not math.isclose(
                equal, best["best_validation_wape"], rel_tol=1e-10):
            raise ValueError("Inconsistent FedPer equal-client validation WAPE")
        runs.append({"path": path, "best": best, "latest": latest,
                     "selection_key": (latest_path.stat().st_mtime_ns, str(path.resolve()))})
    return (max(runs, key=lambda row: row["selection_key"]) if runs else None), skipped


def _append_record(records, algorithm, model_name, path, payload, latest=None, **extra):
    validation = payload["current_validation"]
    records.append({"algorithm": algorithm, "model_name": model_name,
                    "checkpoint": str(path.resolve()) if path else None,
                    "best_round": validation.get("round"),
                    "client_equal_wape_percent": validation["client_equal_wape_percent"],
                    "overall": validation["overall"], "clients": validation["clients"],
                    "training_budget": {"configured_rounds": payload["config"]["rounds"],
                                        "completed_rounds": latest.get("round") if latest else payload["config"]["rounds"],
                                        "local_epochs_per_round": payload["config"]["local_epochs"]},
                    "config": payload["config"], "best_validation_wape": payload.get("best_validation_wape"),
                    "data_signatures": payload["data_signatures"],
                    "round_sample_counts": payload.get("round_sample_counts"),
                    "smoke": False, "test_evaluated": False, **extra})


def compare(g2_root="results/g2", fedper_root="results/g2_2/fedper", seed=42,
            rounds=20, output="results/g2_2/fedper_comparison.json"):
    records, skipped = [], []
    scanned = {}
    for algorithm in ("fedavg", "centralized", "fedprox"):
        scanned[algorithm], ignored = scan_runs(g2_root, algorithm, seed, rounds)
        skipped.extend(ignored)
    for model_name in MODEL_NAMES:
        candidates = [run for run in scanned["fedavg"] if run["best"]["model_name"] == model_name]
        if not candidates:
            skipped.append({"model_name": model_name, "method": "fedavg", "reason": "missing baseline"})
            continue
        avg_run = select_latest(candidates)
        avg_path, avg = avg_run["path"], avg_run["best"]

        # Local-only records below are tied to this G2 data-signature/config reference.
        selected = {"fedavg": (avg_path, avg, avg_run["latest"])}
        for algorithm in ("centralized",):
            candidates = [run for run in scanned[algorithm] if run["best"]["model_name"] == model_name]
            if candidates:
                candidate = select_latest(candidates)
                check_matched_protocol(candidate["best"], avg)
                selected[algorithm] = (candidate["path"], candidate["best"], candidate["latest"])
            else:
                skipped.append({"model_name": model_name, "method": algorithm, "reason": "missing baseline"})

        prox_runs = scanned["fedprox"]
        prox_candidates = [run for run in prox_runs if run["best"].get("model_name") == model_name
                           and run["best"].get("mu") == FEDPROX_MU[model_name]]
        if prox_candidates:
            prox = select_latest(prox_candidates)
            check_matched_protocol(prox["best"], avg)
            selected["fedprox"] = (prox["path"], prox["best"], prox["latest"])
        else:
            skipped.append({"model_name": model_name, "method": "fedprox",
                            "reason": f"no complete formal run for G2.1 selected mu={FEDPROX_MU[model_name]}"})

        fedper_run, fedper_skipped = _scan_fedper(fedper_root, model_name, seed, rounds)
        skipped.extend(fedper_skipped)
        if fedper_run:
            path, payload, latest = fedper_run["path"], fedper_run["best"], fedper_run["latest"]
            check_matched_protocol(payload, avg)
            selected["fedper"] = (path, payload, latest)
        else:
            skipped.append({"model_name": model_name, "method": "fedper",
                            "reason": "no completed formal FedPer run; user training pending"})

        for algorithm, (path, payload, latest) in selected.items():
            extras = {}
            if algorithm == "fedprox":
                extras.update(mu=payload["mu"],
                              mu_selection_source="completed G2.1 validation sensitivity selection")
            if algorithm == "fedper":
                extras.update(personalized_model_count=5,
                              parameter_semantics="one shared backbone plus five client-private heads")
            _append_record(records, algorithm, model_name, path, payload, latest, **extras)

        for algorithm in ("centralized", "fedper"):
            if algorithm in selected:
                # Ensure same seed, split, model, data and budget against FedAvg.
                check_matched_protocol(selected[algorithm][1], avg)

    if not records:
        raise FileNotFoundError("No matching formal G2 FedAvg baselines found")
    reference = next(row for row in records if row["algorithm"] == "fedavg")
    local_rows = []
    # Existing G1 local checkpoints use G2 data signatures; no forward passes are performed.
    reference_payload_path = Path(reference["checkpoint"])
    reference_payload = load_checkpoint(reference_payload_path)
    add_local_only(local_rows, reference_payload, seed)
    for local in local_rows:
        local_configs = {}
        for cid, path in local["checkpoint_by_client"].items():
            saved = load_checkpoint(path)
            local_configs[cid] = saved["config"]
            for field in ("gradient_clip", "num_workers"):
                if saved["config"].get(field) != reference_payload["config"].get(field):
                    raise ValueError("Local-only protocol mismatch: " + field)
        _append_record(records, "local_only", local["model_name"], None,
                       {"config": reference_payload["config"],
                        "data_signatures": reference_payload["data_signatures"],
                        "current_validation": {"round": None,
                                               "client_equal_wape_percent": local["client_equal_wape_percent"],
                                               "overall": local["overall"], "clients": local["clients"]}},
                       None, checkpoint_by_client=local["checkpoint_by_client"],
                       config_by_client=local_configs,
                       training_budget={"per_client": local["training_budget_by_client"]},
                       note="Five independent Local-only models; not one shared global model.")
        records[-1]["training_budget"] = {"per_client": local["training_budget_by_client"]}

    by_key = {(row["algorithm"], row["model_name"]): row for row in records}
    for row in records:
        avg = by_key.get(("fedavg", row["model_name"]))
        local = by_key.get(("local_only", row["model_name"]))
        row["delta_client_equal_wape_vs_fedavg_pp"] = (
            row["client_equal_wape_percent"] - avg["client_equal_wape_percent"] if avg else None)
        row["delta_client_equal_wape_vs_local_only_pp"] = (
            row["client_equal_wape_percent"] - local["client_equal_wape_percent"] if local else None)
        row["client_delta_wape_vs_fedavg_pp"] = ({
            cid: row["clients"][cid]["wape_percent"] - avg["clients"][cid]["wape_percent"]
            for cid in CLIENT_IDS} if avg else None)
        row["client_delta_wape_vs_local_only_pp"] = ({
            cid: row["clients"][cid]["wape_percent"] - local["clients"][cid]["wape_percent"]
            for cid in CLIENT_IDS} if local else None)
        row["negative_transfer_diagnostics"] = ({
            cid: {"excess_wape_vs_local_only_pp": row["client_delta_wape_vs_local_only_pp"][cid] if local else None,
                  "change_vs_fedavg_pp": row["client_delta_wape_vs_fedavg_pp"][cid]}
            for cid in ("lv_80", "industrial_mvlv")} if avg else None)

    result = {"protocol": {"seed": seed, "rounds": rounds, "clients": list(CLIENT_IDS),
                           "split": "validation", "input_hours": 48, "horizon_hours": 1,
                           "test_evaluated": False},
              "selection": {"g2": "latest completed formal run per method/model/seed; Best by equal-client validation WAPE",
                            "fedprox_mu": FEDPROX_MU,
                            "fedprox_mu_source": "G2.1 completed validation sensitivity selection",
                            "fedper": SELECTION},
              "fedper_interpretation": "FedPer is five personalized prediction models: one shared backbone plus a separate head per client.",
              "runs": records, "skipped_or_missing": skipped,
              "training_budget_note": "Local-only comprises independent per-client training; other methods use recorded round and local-epoch budgets.",
              "validation_only": True, "test_evaluated": False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    fields = ["algorithm", "model_name", "seed", "best_round", "client_id", "mae_kw", "rmse_kw",
              "wape_percent", "mape_percent", "mape_valid_fraction", "client_equal_wape_percent",
              "overall_wape_percent", "delta_client_equal_wape_vs_fedavg_pp",
              "delta_client_equal_wape_vs_local_only_pp", "fedprox_mu", "personalized_model_count",
              "configured_rounds", "completed_rounds", "client_delta_wape_vs_fedavg_pp",
              "client_delta_wape_vs_local_only_pp",
              "test_evaluated"]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            for cid in CLIENT_IDS:
                metrics = row["clients"][cid]
                writer.writerow({"algorithm": row["algorithm"], "model_name": row["model_name"],
                                 "seed": seed, "best_round": row["best_round"], "client_id": cid,
                                 **{key: metrics.get(key) for key in ("mae_kw", "rmse_kw", "wape_percent",
                                                                     "mape_percent", "mape_valid_fraction")},
                                 "client_equal_wape_percent": row["client_equal_wape_percent"],
                                 "overall_wape_percent": row["overall"]["wape_percent"],
                                 "delta_client_equal_wape_vs_fedavg_pp": row["delta_client_equal_wape_vs_fedavg_pp"],
                                 "delta_client_equal_wape_vs_local_only_pp": row.get("delta_client_equal_wape_vs_local_only_pp"),
                                 "client_delta_wape_vs_fedavg_pp": (row["client_delta_wape_vs_fedavg_pp"] or {}).get(cid),
                                 "client_delta_wape_vs_local_only_pp": (row["client_delta_wape_vs_local_only_pp"] or {}).get(cid),
                                 "fedprox_mu": row.get("mu"),
                                 "personalized_model_count": row.get("personalized_model_count"),
                                 "configured_rounds": row["training_budget"].get("configured_rounds"),
                                 "completed_rounds": row["training_budget"].get("completed_rounds"),
                                 "test_evaluated": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2-root", default="results/g2")
    parser.add_argument("--fedper-root", default="results/g2_2/fedper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", default="results/g2_2/fedper_comparison.json")
    args = parser.parse_args()
    result = compare(args.g2_root, args.fedper_root, args.seed, args.rounds, args.output)
    for row in result["runs"]:
        print(f"{row['model_name']} | {row['algorithm']} | BestRound={row['best_round']} | "
              f"AvgValWAPE={row['client_equal_wape_percent']:.4f}%")
    print("[FedPer Compare] Validation-only summaries written to " + str(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
