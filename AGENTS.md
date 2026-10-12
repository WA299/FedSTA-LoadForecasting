# FedSTA project instructions

## Fixed project identity

The project name is **基于时空注意力机制与联邦学习的配电网负荷预测**. The local root is `E:\Dachuang_FedSTA`; raw data is kept under `E:\Dachuang_FedSTA\Datasets` and is never rewritten by preprocessing.

## Technical route

The approved G1 ST-FedNet is GAT (2 heads and self-loops) -> single-layer GRU -> single-head Additive Temporal Attention -> Linear. Hidden dimension is 32, dropout 0.1; no Transformer or extra gates. G1 uses 48 history hours to predict the next 1 hour. Preserve configurable G0 48-to-24 preprocessing. Later phases add FedAvg, FedProx, DP-SGD and epsilon/utility analysis.

## Federated design

There are five clients: the four LV reference grids and the industrial MV/LV grid. Each client keeps its own variable-size graph and load mask. Central coordination receives model updates only; client data and topology remain local. The common model interface must support variable `num_nodes` and a per-client `edge_index`.

## Data interface

`src.data.loader` is the G0 source of truth. `GridData` exposes `load_matrix [8760, num_nodes]`, `edge_index [2, 2*num_physical_branches]`, `load_mask [num_nodes]`, hourly `timestamps`, `node_ids`, and `client_id`. `PreparedClient` exposes a training-fitted per-node `StandardScaler` and chronological `train`, `val`, and `test` window datasets. Windows have `X [48, num_nodes, 1]` and `Y [horizon_hours, num_nodes]` (G0 default 24, G1 configuration 1). Source physical values are in kW; window tensors are standardized. Use `inverse_transform` before reporting physical-unit metrics.

The four MATPOWER-style LV `p_load.csv` files are interpreted as MW and converted by `*1000` to kW. This unit is an explicit source-data assumption because the files do not carry a unit field. Industrial `Load_kWh` is hourly energy; numerically it equals average kW over a one-hour interval. Duplicate LV columns are summed by bus. Duplicate industrial timestamp rows are kept once when identical; conflicting repeats and missing hours are rejected.

## Python organization

Keep loading and preprocessing in `src/data/`, forecasting models in `src/models/`, federated algorithms in `src/fl/`, and privacy code in `src/privacy/`. Configuration belongs in `configs/`, runnable entry points in `scripts/`, tests in `tests/`, documentation in `docs/`, and generated summaries in `results/`. Avoid committing caches, environments, or model checkpoints.

## Git workflow

Use the existing `main` branch and remote `origin` unless a task explicitly requests a feature branch. Inspect `git status` before edits, preserve existing files and history, never force-push, and review the staged file list before committing. The supplied raw snapshot is about 74 MB and is versioned with this project by explicit project authorization. Use Git LFS for future large files as needed.

## Reproducibility

Pin or record Python dependencies, use deterministic chronological splits, fit scalers on training rows only, keep configuration in YAML, and write machine-readable validation inventories. Any imputation, unit conversion, duplicate handling, or topology filtering must be recorded in metadata and the G0 report.

## Progress

G0 data engineering remains available. G1 implements Persistence, Seasonal Naive, GRU, GRU+Temporal Attention, GCN+GRU, GAT+GRU and ST-FedNet in `src/models/forecasting.py`. `src/training` provides train/validation, masked Huber Loss, AdamW, early stopping (patience 8), best/latest checkpoints, epoch-boundary recovery and independent testing. Use validation WAPE for selection; never access the test split during training. Restore physical kW before metrics, exclude non-load nodes, and use |true| >= 0.1 kW for MAPE. Summary must include client-equal WAPE alongside pooled metrics.

## Runtime and commands

Always use `E:\Dachuang_FedSTA\.venv\Scripts\python.exe` (Python 3.13.7). The incorrect `venv` directory created during setup was deleted; `.venv` is the actual PyCharm interpreter. The verified runtime is PyTorch 2.10.0+cu126 on the RTX 4060 Laptop GPU, driver 560.94, CUDA runtime 12.6. Native PyTorch implements graph layers; no PyTorch Geometric is required.

Train: `python -m scripts.train_model --model stfednet --client lv_39 --epochs 30`. Resume: `python -m scripts.train_model --resume <latest.pt> --epochs 50`. Evaluate independently: `python -m scripts.evaluate_model --checkpoint <best.pt>`. Activate `.venv` first or use the absolute interpreter. Run `python -m pytest -q` for regression tests. See `docs/G1_RUN_GUIDE.md` for complete commands.

Formal runs are user-owned in PyCharm. Codex runs only small smoke batches unless explicitly asked to train longer. A checkpoint trained/evaluated with limited batches is marked `smoke`; do not report it as a formal result. Checkpoints and prediction NPZ remain local; source, configs, logs and compact JSON results may be committed. Checkpoints bind to a hash of data/topology/IDs and save scaler, optimizer and RNG state. Resume restores the last completed epoch, not a partial epoch.

G1 validation completed: 47 automated tests passed, including the six G0 tests; 35 actual-data CUDA model/client smoke combinations passed, including 25 neural-model resume checks. See `docs/G1_ACCEPTANCE_REPORT.md`. Await overall acceptance before user-owned formal comparison experiments; G2 follows with federated algorithms.

## G2 and G3 extension boundary

Model parameters must stay independent of node count so five clients can share them. Reuse `build_model`, `masked_huber`, `MetricAccumulator` and client `PreparedClient` in G2. G3 privacy units must be stated explicitly: one sliding training window is a record containing all nodes and hours; nodes are not independent DP records. Overlapping windows do not automatically provide user-level privacy. Do not claim epsilon before defining adjacency, sampling and accountant assumptions.


