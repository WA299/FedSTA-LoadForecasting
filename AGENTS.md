# FedSTA project instructions

## Fixed project identity

The project name is **基于时空注意力机制与联邦学习的配电网负荷预测**. The local root is `E:\Dachuang_FedSTA`; raw data is kept under `E:\Dachuang_FedSTA\Datasets` and is never rewritten by preprocessing.

## Technical route

The planned ST-FedNet combines physical-topology graph attention (GAT), temporal attention, and an adaptive gated fusion. Later phases add GRU/TCN/GCN-GRU baselines, FedAvg, FedProx, DP-SGD, and an epsilon/utility analysis. The first forecasting task uses 48 hourly observations to predict the following 24 hourly node loads.

## Federated design

There are five clients: the four LV reference grids and the industrial MV/LV grid. Each client keeps its own variable-size graph and load mask. Central coordination receives model updates only; client data and topology remain local. The common model interface must support variable `num_nodes` and a per-client `edge_index`.

## Data interface

`src.data.loader` is the G0 source of truth. `GridData` exposes `load_matrix [8760, num_nodes]`, `edge_index [2, 2*num_physical_branches]`, `load_mask [num_nodes]`, hourly `timestamps`, `node_ids`, and `client_id`. `PreparedClient` exposes a training-fitted per-node `StandardScaler` and chronological `train`, `val`, and `test` window datasets. Windows have `X [48, num_nodes, 1]` and `Y [24, num_nodes]`; values are in kW. Use `inverse_transform` before reporting MAE, RMSE, or WAPE.

The four MATPOWER-style LV `p_load.csv` files are interpreted as MW and converted by `*1000` to kW. This unit is an explicit source-data assumption because the files do not carry a unit field. Industrial `Load_kWh` is hourly energy; numerically it equals average kW over a one-hour interval. Duplicate LV columns are summed by bus. Duplicate industrial timestamp rows are kept once when identical; conflicting repeats and missing hours are rejected.

## Python organization

Keep loading and preprocessing in `src/data/`, forecasting models in `src/models/`, federated algorithms in `src/fl/`, and privacy code in `src/privacy/`. Configuration belongs in `configs/`, runnable entry points in `scripts/`, tests in `tests/`, documentation in `docs/`, and generated summaries in `results/`. Avoid committing caches, environments, or model checkpoints.

## Git workflow

Use the existing `main` branch and remote `origin` unless a task explicitly requests a feature branch. Inspect `git status` before edits, preserve existing files and history, never force-push, and review the staged file list before committing. The supplied raw snapshot is about 74 MB and is versioned with this project by explicit project authorization. Use Git LFS for future large files as needed.

## Reproducibility

Pin or record Python dependencies, use deterministic chronological splits, fit scalers on training rows only, keep configuration in YAML, and write machine-readable validation inventories. Any imputation, unit conversion, duplicate handling, or topology filtering must be recorded in metadata and the G0 report.

## Progress

G0 data engineering is implemented: all five grids load through one API, physical topology and effective-load masks are built, 2021 data is aligned, normalized, windowed, and validated, and a smoke inventory is generated. G1 is model development: implement ST-FedNet and baselines without changing the G0 source format.


