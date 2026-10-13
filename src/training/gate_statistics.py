"""Streaming gate moments over validation windows, history steps and all nodes."""

import math

import torch


class GateStatistics:
    """Population standard deviation; merge batches by their element counts."""

    def __init__(self):
        self.count = self.windows = 0
        self.mean = self.m2 = 0.0

    def update(self, gate):
        values = gate.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Gate values must be finite and in [0,1]")
        count = values.numel()
        if not count:
            return
        mean = float(values.mean())
        m2 = float(torch.square(values - mean).sum())
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total
        self.windows += gate.shape[0]

    def compute(self, total_windows):
        if not self.count:
            raise ValueError("No validation gates collected")
        return {"gate_mean": self.mean, "gate_std": math.sqrt(max(self.m2 / self.count, 0.0)),
                "gate_count": self.count, "validation_windows": self.windows,
                "total_validation_windows": total_windows,
                "complete_validation": self.windows == total_windows,
                "scope": "evaluated_validation_windows_all_history_steps_all_nodes",
                "std_definition": "population (ddof=0)",
                "node_scope": "all nodes including non-load intermediate buses",
                "overlap_handling": "each window occurrence is counted"}
