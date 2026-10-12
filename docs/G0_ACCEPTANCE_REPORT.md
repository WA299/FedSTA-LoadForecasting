# G0 验收报告

## 完成项

G0 已完成数据工程实现。`src/data/loader.py` now discovers the five supplied grids, parses both source formats, aggregates repeated load columns, builds bidirectional physical topology, aligns the 2021 hourly period, fits a train-only scaler, and provides reproducible 48-to-24 sliding windows. `GridData.as_torch()` and `WindowDataset.as_torch()` are ready for the PyTorch model phase; PyTorch itself is intentionally not installed or trained in G0.

## Five-client inventory

| client | nodes | physical branches | active load nodes | 2021 samples | train/val/test windows | topology |
|---|---:|---:|---:|---:|---:|---|
| `lv_39` | 39 | 38 | 28 | 8,760 | 6,061 / 1,291 / 1,291 | connected |
| `lv_50` | 50 | 49 | 21 | 8,760 | 6,061 / 1,291 / 1,291 | connected |
| `lv_56` | 56 | 55 | 44 | 8,760 | 6,061 / 1,291 / 1,291 | connected |
| `lv_80` | 80 | 79 | 32 | 8,760 | 6,061 / 1,291 / 1,291 | connected |
| `industrial_mvlv` | 76 | 74 active (75 raw rows) | 45 | 8,760 | 6,061 / 1,291 / 1,291 | 2 components: 58 + 18 |

The machine-readable inventory is `results/g0_inventory.json`.

## Quality findings and decisions

* LV `p_load.csv` column headers identify load buses, not unique nodes. Repeated columns are summed. The 39-bus case repeats bus 30; the 56-bus case repeats buses 54, 35, 34, 3, and 52.
* LV timestamps are complete and unique at one-hour frequency over 2021.
* Each industrial file has one identical repeated row at 2021-06-01 00:00 (45 duplicate rows total); duplicates are removed. There are no conflicting duplicate values and no missing values in 2021 after alignment.
* One industrial branch is disabled. It is excluded from the active graph, which correctly leaves two physical connected components. No isolated node was found.
* LV units are not stated in the files. The implementation documents the MATPOWER-based MW-to-kW assumption. Industrial `Load_kWh` is treated as hourly kWh and therefore numerically kW at one-hour resolution.
* There are 17 exact active-curve pairs between `lv_39` and `lv_56`; curves are not deduplicated because they belong to different physical client cases.
* The dataset snapshot is approximately 74 MB. No license file or explicit license notice is present in the supplied directory; the raw snapshot is included because the project owner explicitly authorized versioning it in this repository.

## Validation

`tests/test_data_pipeline.py` covers: loading all five grids, legal edge endpoints, date range, finite values, duplicate-column aggregation, industrial duplicate-row handling, reproducible splits, target disjointness, train-only scaler fit, load masks, and window shapes. On the local Python 3.13 environment:

```text
6 passed
```

The bundled Codex runtime Python did not include pytest; the system Python 3.13.7 did. The fast smoke command also completed and wrote `results/g0_inventory.json`.

## G1 readiness

The data layer is ready for G1 model development. Confirm the LV unit assumption against the source documentation. Model work can consume the variable-node `GridData`/`PreparedClient` API without modifying original files.


