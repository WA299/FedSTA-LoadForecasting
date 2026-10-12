# FedSTA G0 数据说明

## 原始目录

`Datasets/norway_4_lv_grids` contains four MATPOWER-style low-voltage cases:

| client | case | bus | physical branches | load buses | p/q files |
|---|---|---:|---:|---:|---|
| `lv_39` | 39-bus semi-urban | 39 | 38 | 28 | `p_load.csv`, `q_load.csv` |
| `lv_50` | 50-bus rural | 50 | 49 | 21 | `p_load.csv`, `q_load.csv` |
| `lv_56` | 56-bus semi-urban | 56 | 55 | 44 | `p_load.csv`, `q_load.csv` |
| `lv_80` | 80-bus rural | 80 | 79 | 32 | `p_load.csv`, `q_load.csv` |

Each case also has `mpc_bus.csv`, `mpc_branch.csv`, `mpc_base_mva.csv`, a `load_bus_extra.csv` mapping, and a branch description file. The header of `p_load.csv` is not a one-to-one node list: it contains user/load columns. In particular, `lv_39` repeats bus 30, while `lv_56` repeats buses 54 (six columns), 35, 34, 3, and 52. The loader sums all columns carrying the same bus identifier.

`Datasets/norway_industrial_mvlv` has 76 buses and 75 raw branch rows in `branch.csv`; one row is disabled (`r1v11.0b10`–`r2v11.0b2`), leaving 74 active physical branches. There are 45 semicolon-separated `*.txt` load files, each with `Bus_ID;Timestamp;Load_kWh`, and `bus.csv`/`gen.csv`. One-hour samples cover different start dates, but all active files have data for the complete 2021 window after alignment. The industrial active topology has two connected components (58 and 18 nodes) because the disabled tie is excluded from `edge_index`; no isolated node was found.

## Time and units

The first experiment period is the inclusive hourly range 2021-01-01 00:00 through 2021-12-31 23:00 (8,760 rows). LV timestamps are strictly hourly and unique. Industrial files contain one identical duplicate at 2021-06-01 00:00 per file (45 rows total); the loader removes the repeat. No conflicting industrial duplicate was found, and no missing values remain in the selected 2021 matrix.

The LV files do not declare a unit. Their values and MATPOWER context are treated as MW and converted to kW with `p_load * 1000`; this assumption is stored in `GridData.metadata`. Industrial `Load_kWh` is hourly energy, so its numeric value equals the average kW over each one-hour interval. Reactive-load `q_load.csv` is inspected and retained as a documented source for a future feature extension; G0's unified target is active power.

## Topology and alignment

`node_ids` preserves the order in `mpc_bus.csv` or `bus.csv`. `edge_index` maps this order and contains both directions of every active physical branch, so its shapes are `[2, 76]`, `[2, 98]`, `[2, 110]`, `[2, 158]`, and `[2, 148]` for the five clients. Intermediate buses remain in the graph even when their load mask is false. The effective-load mask is true exactly for buses represented by one or more load columns/files.

Exact duplicate active curves occur between some LV cases (17 cross-case pairs between `lv_39` and `lv_56`), while no active industrial curve exactly duplicates an LV curve. These are source characteristics, not deduplicated away: each physical client keeps its own topology and client identity.

## Preprocessing contract

`prepare_client` computes chronological boundaries at 70/85% of the 8,760 rows: 6,132 training rows, 1,314 validation rows, and 1,314 test rows. The scaler is fitted on the first 6,132 rows only, independently per node. Windows are generated without crossing target split boundaries: each `X` is `[48, num_nodes, 1]`, each `Y` is `[24, num_nodes]`. For the current one-year period this yields 6,061, 1,291, and 1,291 windows per split. `StandardScaler.inverse_transform` restores kW for physical error metrics.

The implementation is in `src/data/loader.py`; the smoke command is:

```powershell
python -m scripts.validate_data --output results/g0_inventory.json
```

