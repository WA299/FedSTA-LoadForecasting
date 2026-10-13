"""Read stored FedAvg round curves; report trends without running training."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.g21_results import SELECTION, scan_runs, select_latest
from src.fl.common import CLIENT_IDS
from src.training.engine import write_json


def trend(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return {"delta_percentage_points": None, "slope_pp_per_round": None, "range_pp": None}
    return {"delta_percentage_points": float(values[-1] - values[0]),
            "slope_pp_per_round": float(np.polyfit(np.arange(values.size), values, 1)[0]),
            "range_pp": float(np.ptp(values))}


def analyze_run(run):
    path, best, latest = run["path"], run["best"], run["latest"]
    with path.with_name("rounds.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["round"]) for row in rows] != list(range(1, latest["round"] + 1)):
        raise ValueError("CSV must contain every committed round exactly once: " + str(path))
    curve = []
    for row in rows:
        if row.get("smoke", "").lower() not in {"false", "0"} or row["algorithm"] != "fedavg" or row["model_name"] != best["model_name"]:
            raise ValueError("Expected formal FedAvg CSV rows")
        clients = json.loads(row["client_wapes_json"])
        equal = float(row["client_equal_wape_percent"])
        if set(clients) != set(CLIENT_IDS) or not np.isclose(equal, np.mean(list(clients.values())), rtol=1e-10):
            raise ValueError("CSV client WAPEs inconsistent with equal-client WAPE")
        curve.append({"round": int(row["round"]), "client_equal_wape_percent": equal, "clients": clients})
    # First minimum matches the original strictly-improving model selection.
    observed_best = min(curve, key=lambda row: row["client_equal_wape_percent"])
    if observed_best["round"] != best["best_round"] or not np.isclose(observed_best["client_equal_wape_percent"], best["best_validation_wape"], rtol=1e-10):
        raise ValueError("CSV and Best checkpoint disagree")
    late = [row for row in curve if 15 <= row["round"] <= 20]
    tail = curve[-5:]
    statistics = trend([row["client_equal_wape_percent"] for row in tail])
    recent = trend([row["client_equal_wape_percent"] for row in curve[-3:]])
    # Descriptive heuristics, not a statistical convergence proof. The flag
    # alone that the final round is Best is deliberately not used to decide.
    slope, spread = statistics["slope_pp_per_round"], statistics["range_pp"]
    if slope is None:
        judgment, suggestion = "insufficient late-round data", "Collect more validation rounds only if authorized."
    elif abs(slope) <= 0.05 and spread <= 0.5:
        judgment = "near plateau in this five-round window"
        suggestion = "No strong late-window trend; extending is optional, not evidence of non-convergence."
    elif slope < -0.05:
        judgment = "late-window downward trend with possible fluctuations"
        if recent["slope_pp_per_round"] is not None and abs(recent["slope_pp_per_round"]) <= 0.05 and recent["range_pp"] <= 0.5:
            judgment += "; most recent three rounds are flatter"
        suggestion = "A separate 40-round continuation may clarify the trend; first preserve the original 20-round directory/Best/Latest."
    else:
        judgment = "fluctuating or worsening late window"
        suggestion = "Inspect client-specific curves; final-round Best alone does not establish non-convergence."
    return {"model_name": best["model_name"], "seed": best["seed"], "checkpoint": str(path.resolve()),
            "best_round": best["best_round"], "best_wape_percent": best["best_validation_wape"],
            "completed_rounds": latest["round"], "last_round_is_best": curve[-1]["round"] == best["best_round"],
            "round_curve": curve, "rounds_15_to_20": late,
            "rounds_15_to_20_trend": trend([row["client_equal_wape_percent"] for row in late]),
            "last_five_rounds": [row["round"] for row in tail], "last_five_equal_wape_trend": statistics,
            "last_three_equal_wape_trend": recent,
            "client_last_five": {cid: {"wapes_percent": [row["clients"][cid] for row in tail],
                                       **trend([row["clients"][cid] for row in tail])} for cid in CLIENT_IDS},
            "judgment": judgment, "suggestion": suggestion, "test_evaluated": False}


def analyze_convergence(root="results/g2", seed=42, rounds=20):
    runs, skipped = scan_runs(root, "fedavg", seed, rounds)
    groups = defaultdict(list)
    for run in runs:
        groups[(run["best"]["model_name"], run["best"]["seed"])].append(run)
    if not groups:
        raise FileNotFoundError("No completed formal FedAvg curves found")
    return {"selection": SELECTION, "configured_rounds_filter": rounds, "split": "validation",
            "test_evaluated": False, "automatic_training_started": False,
            "heuristic": {"window_rounds": 5, "plateau_max_abs_slope_pp": 0.05,
                          "plateau_max_range_pp": 0.5, "note": "descriptive heuristic, no statistical proof of convergence"},
            "runs": [analyze_run(select_latest(group)) for _, group in sorted(groups.items())], "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/g2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", default="results/g2_1/fedavg_convergence.json")
    args = parser.parse_args()
    result = analyze_convergence(args.root, args.seed, args.rounds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["model_name", "seed", "round", "client_equal_wape_percent"] + list(CLIENT_IDS)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in result["runs"]:
            for row in run["round_curve"]:
                writer.writerow({"model_name": run["model_name"], "seed": run["seed"],
                                 "round": row["round"], "client_equal_wape_percent": row["client_equal_wape_percent"],
                                 **row["clients"]})
            print(f"{run['model_name']} | BestRound={run['best_round']} | LastRoundBest={run['last_round_is_best']} | {run['judgment']}")
    print("[G2.1 Convergence] Read-only analysis: " + str(output.resolve()))


if __name__ == "__main__":
    main()
