"""Compare completed FedProx mu groups against matched FedAvg; validation only."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from scripts.g21_results import (SELECTION, check_matched_protocol, diagnostic_summary,
                                scan_runs, select_latest)
from src.fl.common import CLIENT_IDS
from src.fl.diagnostics import DIAGNOSTIC_FIELDS
from src.training.engine import write_json


def compare_mu(root="results/g2", seed=None, rounds=20):
    groups, skipped, inventory = defaultdict(list), [], []
    for algorithm in ("fedavg", "fedprox"):
        runs, ignored = scan_runs(root, algorithm, seed, rounds)
        skipped.extend(ignored)
        for run in runs:
            p = run["best"]
            groups[(algorithm, p["model_name"], p["seed"], p.get("mu"))].append(run)
    selected = {key: select_latest(runs) for key, runs in groups.items()}
    records = []
    for key, run in sorted(selected.items(), key=lambda pair: str(pair[0])):
        algorithm, model_name, run_seed, mu = key
        baseline = selected.get(("fedavg", model_name, run_seed, None))
        if baseline is None:
            raise ValueError(f"Missing matched FedAvg for {model_name}/seed{run_seed}")
        # Check ALL replicates in the group, not only the selected one.
        for candidate in groups[key]:
            check_matched_protocol(candidate["best"], baseline["best"])
            inventory.append({"algorithm": algorithm, "model_name": model_name, "seed": run_seed,
                              "mu": mu, "checkpoint": str(candidate["path"].resolve()),
                              "best_wape_percent": candidate["best"]["best_validation_wape"],
                              "selected": candidate is run})
        p, latest = run["best"], run["latest"]
        val = p["current_validation"]
        diagnostics = diagnostic_summary(p)
        records.append({"algorithm": algorithm, "model_name": model_name, "seed": run_seed, "mu": mu,
                        "checkpoint": str(run["path"].resolve()), "best_round": p["best_round"],
                        "client_equal_wape_percent": val["client_equal_wape_percent"],
                        "overall_wape_percent": val["overall"]["wape_percent"], "clients": val["clients"],
                        "delta_vs_fedavg_percentage_points": val["client_equal_wape_percent"] - baseline["best"]["best_validation_wape"],
                        "training_budget": {"configured_rounds": p["config"]["rounds"],
                                            "completed_rounds": latest["round"],
                                            "local_epochs": p["config"]["local_epochs"]},
                        "config": p["config"], "data_signatures": p["data_signatures"],
                        "diagnostics": diagnostics,
                        "diagnostics_status": "recorded" if all(value is not None for value in diagnostics.values()) else "not recorded",
                        "smoke": False, "test_evaluated": False})
    if not records:
        raise FileNotFoundError("No complete formal FedAvg/FedProx checkpoints found")
    missing = [{"model_name": model, "seed": run_seed, "mu": mu}
               for model, run_seed in sorted({(key[1], key[2]) for key in selected})
               for mu in (0.01, 0.001, 0.0001)
               if ("fedprox", model, run_seed, mu) not in selected]
    return {"selection": SELECTION, "split": "validation", "test_evaluated": False,
            "configured_rounds_filter": rounds, "runs": records, "all_runs": inventory,
            "missing_planned_experiments": missing, "skipped": skipped,
            "missing_diagnostics": "null / not recorded; no historical diagnostics inferred"}


def save_comparison(result, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    fields = ["algorithm", "model_name", "seed", "mu", "best_round", "client_id", "wape_percent",
              "client_equal_wape_percent", "overall_wape_percent", "delta_vs_fedavg_percentage_points",
              "diagnostics_status"] + list(DIAGNOSTIC_FIELDS)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in result["runs"]:
            for cid in CLIENT_IDS:
                writer.writerow({**{key: run.get(key) for key in fields}, "client_id": cid,
                                 "wape_percent": run["clients"][cid]["wape_percent"], **run["diagnostics"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/g2")
    parser.add_argument("--seed", type=int, help="Omit to group all seeds independently")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", default="results/g2_1/fedprox_mu_comparison.json")
    args = parser.parse_args()
    result = compare_mu(args.root, args.seed, args.rounds)
    save_comparison(result, args.output)
    for run in result["runs"]:
        print(f"{run['model_name']} | seed={run['seed']} | {run['algorithm']} mu={run['mu']} | BestRound={run['best_round']} | AvgValWAPE={run['client_equal_wape_percent']:.4f}%")
    print("[G2.1 Compare] Validation summaries only: " + str(Path(args.output).resolve()))


if __name__ == "__main__":
    main()
