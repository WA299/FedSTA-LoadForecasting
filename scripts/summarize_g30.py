"""Combine G3.0 environment probes and CUDA timing into one audit JSON."""

import json
from pathlib import Path


def main():
    root = Path("results/g3_0")
    compatibility = json.loads((root / "compatibility.json").read_text(encoding="utf-8"))
    benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    rows = benchmark["rows"]
    report = {
        "phase": "G3.0 feasibility only",
        "formal_dp_training_executed": False,
        "test_split_accessed": False,
        "smoke": bool(benchmark["smoke"]),
        "environment": {key: compatibility[key] for key in
                        ("python", "platform", "torch", "cuda_runtime",
                         "cuda_available", "gpu", "opacus")},
        "per_sample_gradient_methods": {
            "explicit_window_loop": {
                "status": "numerically validated on both models; reference implementation",
                "supports_variable_graph_nodes": True,
                "noise_added_to": "sum of individually clipped per-window gradients",
                "normalization": "actual batch size, or caller-supplied expected Poisson batch size",
            },
            "opacus": compatibility["opacus"],
            "torch_func": compatibility["torch_func"],
        },
        "cuda_benchmark": {
            "device": benchmark["device"], "torch": benchmark["torch"],
            "gpu": benchmark["gpu"], "rows": rows,
            "maximum_per_sample_cuda_peak_mb": max(row["per_sample_peak_cuda_mb"] for row in rows),
            "maximum_torch_func_cuda_peak_mb": max(row["torch_func_peak_cuda_mb"] for row in rows),
            "maximum_torch_func_gradient_stack_bytes": max(
                row["torch_func_gradient_stack_bytes"] for row in rows),
            "epoch_estimate_scope": "batch32 extrapolation from one limited training batch; excludes data loading and validation",
        },
        "accounting_examples": {
            "status": "theoretical Poisson-sampling examples; no model was trained",
            "mechanism": "per-client Poisson subsampled Gaussian mechanism, composed with RDP",
            "record": "one complete graph time window",
            "adjacency": "add/remove one transformed training window, conditional on fixed public topology, load mask, and scaler",
            "clients": 5,
            "client_participation": "5/5 each communication round",
            "windows_per_client": 6084,
            "batch_size_expected": 32,
            "sample_rate": 32 / 6084,
            "nominal_steps_per_epoch": 191,
            "rounds": 20,
            "local_epochs_per_round": 1,
            "steps_per_client": 3820,
            "clipping_norm_example": 1.0,
            "delta": 1e-5,
            "rdp_results": compatibility["rdp_examples"],
            "current_g2_sampler_accounted_by_these_values": False,
            "current_g2_sampler": "fixed-size shuffled DataLoader; Poisson amplification assumptions do not apply without a matching proof",
        },
        "privacy_limits": [
            "Window-level protection does not imply timepoint-, household-, user-, or client-level protection.",
            "Overlapping 48-to-1 windows reuse an hourly value in up to 49 records; a raw-timepoint change can affect multiple records.",
            "A training-fitted scaler and physical topology/load mask are private-derived side information unless public or separately protected.",
            "Published validation metrics and validation-based model selection are not protected by DP-SGD on training windows.",
            "Repeated/correlated raw curves across clients require cross-client composition analysis.",
            "DP-SGD does not add secure aggregation, encrypted transport, client identity hiding, or protect all metadata.",
        ],
    }
    path = root / "feasibility.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
