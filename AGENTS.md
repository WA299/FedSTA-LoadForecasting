# FedSTA project instructions

## Fixed project identity

The project name is **基于时空注意力机制与联邦学习的配电网负荷预测**. The local root is `E:\Dachuang_FedSTA`; raw data is kept under `E:\Dachuang_FedSTA\Datasets` and is never rewritten by preprocessing.

## Technical route

The original G1 ST-FedNet remains GAT (2 heads and self-loops) -> single-layer GRU -> single-head Additive Temporal Attention -> Linear. G1.1 adds a separate `stfednet_residual` variant: GAT -> Linear back to one channel -> `x + alpha * spatial`, with learnable scalar alpha initialized to 0.05 -> the same GRU/temporal attention/output head. Do not change old G1 model shapes or checkpoints. Both use hidden dimension 32, dropout 0.1, 48-to-1 windows; preserve configurable G0 48-to-24 preprocessing. Later phases add FedAvg, FedProx, DP-SGD and epsilon/utility analysis.

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

G0 data engineering remains available. G1 implements Persistence, Seasonal Naive, GRU, GRU+Temporal Attention, GCN+GRU, GAT+GRU and original ST-FedNet; G1.1 adds `stfednet_residual` without altering them. `src/training` provides train/validation, masked Huber Loss, AdamW, early stopping (patience 8), best/latest checkpoints, epoch-boundary recovery and independent testing. Use validation WAPE for selection; never access the test split during training or the G1.1 comparison. Restore physical kW before metrics, exclude non-load nodes, and use |true| >= 0.1 kW for MAPE. Summary must include client-equal WAPE alongside pooled metrics.

## Runtime and commands

Always use `E:\Dachuang_FedSTA\.venv\Scripts\python.exe` (Python 3.13.7). The incorrect `venv` directory created during setup was deleted; `.venv` is the actual PyCharm interpreter. The verified runtime is PyTorch 2.10.0+cu126 on the RTX 4060 Laptop GPU, driver 560.94, CUDA runtime 12.6. Native PyTorch implements graph layers; no PyTorch Geometric is required.

Train: `python -m scripts.train_model --model stfednet --client lv_39 --epochs 30`. Resume: `python -m scripts.train_model --resume <latest.pt> --epochs 50`. Evaluate independently: `python -m scripts.evaluate_model --checkpoint <best.pt>`. Activate `.venv` first or use the absolute interpreter. Run `python -m pytest -q` for regression tests. See `docs/G1_RUN_GUIDE.md` for complete commands.

Formal runs are user-owned in PyCharm. Codex runs only small smoke batches unless explicitly asked to train longer. A checkpoint trained/evaluated with limited batches is marked `smoke`; do not report it as a formal result. Checkpoints and prediction NPZ remain local; source, configs, logs and compact JSON results may be committed. Checkpoints bind to a hash of data/topology/IDs and save scaler, optimizer and RNG state. Resume restores the last completed epoch, not a partial epoch.

G1 validation completed: 47 automated tests passed, including the six G0 tests; 35 actual-data CUDA model/client smoke combinations passed, including 25 neural-model resume checks. See `docs/G1_ACCEPTANCE_REPORT.md`. G1/G1.1/G1.2 source and local-only checkpoints remain available as G2 comparison baselines.

G1.1 adds `scripts/train_clients.py` for five-client training, `scripts/compare_validation.py` for seed/protocol-matched validation WAPE comparisons against GRU+Attention and original ST-FedNet, and `scripts/smoke_g11.py` for bounded train/validation-only checks. Write G1.1 runs under `results/g1_1/`; do not overwrite the user's existing G1 runs. `space_scale` is logged to terminal/CSV/checkpoints each epoch. See `docs/G1_1_TECHNICAL_NOTE.md`; no formal test evaluation is part of G1.1 development.

G1.2 adds `stfednet_gated`: reuse the two-head GAT, project 32 features back to one channel S, compute `gate=sigmoid(Linear(2,1)(concat(X,S)))`, and fuse `X+gate*S` before the unchanged single GRU/additive attention/linear head. Initialize gate weights to zero and bias to -3 (gate about 0.047426). Gate parameters are shared across all samples, times, nodes and clients; there are no node-specific learnable parameters. Preserve all eight previous model architectures/checkpoints.

G1.2 uses the unchanged G1 scientific configuration (48-to-1, hidden32, dropout0.1, batch32, lr0.001, epochs30, patience8, seed42). Both training CLIs default gated runs to `results/g1_2/`; batch summary follows the output root. Validation collects gates from the same forward pass as predictions across all evaluated windows, all 48 history steps and all nodes, including non-load buses; population std (ddof=0), repeated window occurrences counted. Full formal runs cover every validation window; smoke stats explicitly record partial coverage. Log GateMean/GateStd per epoch and save the validation-best epoch's stats in Best and `best_gate_statistics.json`.

Use `python -m scripts.train_clients --model stfednet_gated --device cuda` for user-owned five-client training and `python -m scripts.compare_g12 --seed 42` afterwards for validation-only four-model comparisons. The comparator checks client, seed, 48-to-1, data signature and training config, skips smoke and reports four physical-unit metrics, equal-client WAPE and win counts. `scripts.smoke_g12` is bounded training/validation only with test access forbidden. See `docs/G1_2_TECHNICAL_NOTE.md` and `results/g1_2/validation.json` for validation evidence. No G2 work or formal test evaluation is authorized in G1.2. After G1.2, freeze forecasting structure in principle; await user-owned validation comparison and explicit G2 authorization.

## G2 and G3 extension boundary

Model parameters must stay independent of node count so five clients can share them. Reuse `build_model`, `masked_huber`, `MetricAccumulator` and client `PreparedClient` in G2. G3 privacy units must be stated explicitly: one sliding training window is a record containing all nodes and hours; nodes are not independent DP records. Overlapping windows do not automatically provide user-level privacy. Do not claim epsilon before defining adjacency, sampling and accountant assumptions.

## G2 implementation status and operating rules

G2 now provides a single-machine five-client simulation for `fedavg`, `fedprox`, `local_only` comparison and `centralized`. The only federated model choices are `stfednet_gated` and `gru_attention`. `src/fl/client.py` trains one private variable-size graph and returns only a model state, training-window count and aggregate local-loss scalars; `src/fl/aggregation.py` performs sample-count weighted FedAvg and the trainable-parameter FedProx penalty; `src/fl/server.py` evaluates one shared global model on each validation split after every round. `src/fl/centralized.py` uses one optimizer and visits client batches sequentially without concatenating node axes.

G2 uses `configs/g2.yaml`: 48-to-1 windows, 70/15/15 chronological target split, train-fitted scalers, batch 32, AdamW, Huber loss, hidden dimension 32, dropout 0.1, seed 42, 20 rounds, one local epoch and FedProx `mu=0.01`. Global selection is by five-client equal-weight validation WAPE; pooled overall WAPE is recorded separately. `edge_index`, node IDs, masks, scalers and raw data never enter the server state dictionary. G2 checkpoints contain model metadata, round/best-round state, client/data signatures, RNG state, configuration, validation summaries and the explicit per-client window counts, but no raw client arrays or scaler payloads.

G2 commands are validation-only during development. Use `python -m scripts.train_federated --algorithm fedavg --model stfednet_gated`, `python -m scripts.train_federated --algorithm fedprox --model stfednet_gated --mu 0.01`, and `python -m scripts.train_centralized --model stfednet_gated`. `python -m scripts.smoke_g2 --device cuda` runs two rounds for both models and both federated algorithms plus centralized controls with `smoke=true`; it forbids access to test windows. `python -m scripts.compare_g2 --root results/g2` reads only non-smoke best global validation checkpoints and can append existing G1 local-only summaries. No formal G2 run, test evaluation, FedPer, FedAtt or DP-SGD was started in this phase. The simulator is not a deployment or a privacy guarantee; federated updates can still reveal information and G3 must define records, adjacency, clipping and accounting before any epsilon claim.

G2 validation: 99 automated tests passed (83 existing, 16 added); six real-five-client CUDA model/algorithm combinations completed with strict reload, actual parameter updates and exact same-device recovery. Mid-round FedAvg/FedProx interruption, CSV repair and smoke-to-formal resume rejection are tested. Ten existing local formal Best checkpoints strictly loaded. Evidence: `results/g2/development_validation.json`, `results/g2_smoke/summary.json` and `smoke.log`; implementation details and complete commands: `docs/G2_TECHNICAL_NOTE.md`. `scripts.compare_g2 --local-only` summarizes the existing G1/G1.2 runs without training or data forwards. Next work is user-owned G2 formal validation comparison and overall acceptance; further algorithms and G3 require a new request.


