"""Metrics and mergeable sufficient statistics in physical kW units."""

import math

import numpy as np


class MetricAccumulator:
    def __init__(self, mape_threshold=0.1):
        self.threshold = float(mape_threshold)
        self.count = self.mape_count = 0
        self.absolute_error = self.squared_error = self.absolute_true = self.mape_sum = 0.0

    def update(self, true, prediction, load_mask):
        true, prediction = np.asarray(true), np.asarray(prediction)
        mask = np.asarray(load_mask, dtype=bool)
        if true.shape != prediction.shape or true.shape[-1] != mask.size:
            raise ValueError("Prediction, target and node mask dimensions must match")
        true, prediction = true[..., mask].astype(np.float64), prediction[..., mask].astype(np.float64)
        if not np.isfinite(true).all() or not np.isfinite(prediction).all():
            raise ValueError("Nonfinite values in physical-unit metrics")
        error = np.abs(prediction - true)
        valid = np.abs(true) >= self.threshold
        self.count += true.size
        self.mape_count += int(valid.sum())
        self.absolute_error += float(error.sum())
        self.squared_error += float(np.square(error).sum())
        self.absolute_true += float(np.abs(true).sum())
        self.mape_sum += float((error[valid] / np.abs(true[valid])).sum())

    def statistics(self):
        return {"count": self.count, "mape_count": self.mape_count,
                "absolute_error": self.absolute_error, "squared_error": self.squared_error,
                "absolute_true": self.absolute_true, "mape_sum": self.mape_sum}

    def compute(self):
        if not self.count:
            raise ValueError("No active load observations available for metrics")
        return {"mae_kw": self.absolute_error / self.count,
                "rmse_kw": math.sqrt(self.squared_error / self.count),
                "wape_percent": 100 * self.absolute_error / self.absolute_true if self.absolute_true else None,
                "mape_percent": 100 * self.mape_sum / self.mape_count if self.mape_count else None,
                "mape_valid_fraction": self.mape_count / self.count,
                "mape_valid_count": self.mape_count, "effective_count": self.count,
                "mape_threshold_kw": self.threshold}


def aggregate_results(results):
    """Pool sample statistics and also expose client-equal WAPE."""
    if not results:
        raise ValueError("No result files supplied")
    signatures = {(r["model_name"], r["seed"], r["input_hours"], r["horizon_hours"],
                   r["smoke"], r["mape_threshold_kw"]) for r in results}
    if len(signatures) != 1:
        raise ValueError("Aggregate only the same model, seed, task and evaluation protocol")
    client_ids = [r["client_id"] for r in results]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("Each client may appear only once in a summary")
    accumulator = MetricAccumulator(results[0]["mape_threshold_kw"])
    for key in accumulator.statistics():
        setattr(accumulator, key, sum(r["statistics"][key] for r in results))
    wapes = [r["metrics"]["wape_percent"] for r in results]
    return {"model_name": results[0]["model_name"], "seed": results[0]["seed"],
            "client_count": len(results), "complete_five_clients": len(results) == 5,
            "smoke": results[0]["smoke"], "overall": accumulator.compute(),
            "client_equal_wape_percent": float(np.mean(wapes)) if all(v is not None for v in wapes) else None,
            "clients": {r["client_id"]: r["metrics"] for r in results}}
