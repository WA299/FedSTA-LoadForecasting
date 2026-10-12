"""Run a fast G0 data validation and write a machine-readable inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.loader import load_all_grids, prepare_all, validate_prepared


def _components(grid):
    adjacency = [set() for _ in grid.node_ids]
    for source, target in grid.edge_index.T:
        adjacency[int(source)].add(int(target))
    seen = set()
    components = []
    for start in range(grid.num_nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _branch_quality(grid):
    # The first half of edge_index stores source branch rows in source order.
    physical = grid.edge_index[:, :grid.physical_branch_count].T
    pairs = [tuple(sorted((int(source), int(target)))) for source, target in physical]
    return {"self_loops": sum(source == target for source, target in physical),
            "parallel_branch_pairs": len(pairs) - len(set(pairs)),
            "invalid_endpoints": int(np.sum((physical < 0) | (physical >= grid.num_nodes)))}


def _curve_duplicates(grids):
    curves = {}
    for client_id, grid in grids.items():
        for node_index in np.flatnonzero(grid.load_mask):
            digest = hashlib.sha256(grid.load_matrix[:, node_index].tobytes()).hexdigest()
            curves.setdefault(digest, []).append((client_id, grid.node_ids[int(node_index)]))
    return [members for members in curves.values() if len(members) > 1]


def _q_load_quality(grid, data_root):
    source = Path(data_root) / "norway_4_lv_grids" / grid.name / "q_load.csv"
    if not source.exists():
        return None
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        samples = 0
        timestamps_match_p = True
        finite = True
        for row in reader:
            if samples >= grid.timestamps.size or np.datetime64(row[0].replace(" ", "T"), "h") != grid.timestamps[samples]:
                timestamps_match_p = False
            finite = finite and all(np.isfinite(float(value)) for value in row[1:])
            samples += 1
    return {"columns": len(header) - 1, "samples": samples,
            "timestamps_match_p": timestamps_match_p and samples == grid.timestamps.size,
            "all_finite": finite}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g0.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="results/g0_inventory.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))["data"]
    data_root = args.data_root or config["root"]
    grids = load_all_grids(data_root, start=np.datetime64(config["start"], "h"),
                           end=np.datetime64(config["end"], "h"))
    prepared = prepare_all(grids, input_hours=int(config["input_hours"]),
                           horizon_hours=int(config["horizon_hours"]),
                           train_ratio=float(config["train_ratio"]),
                           val_ratio=float(config["val_ratio"]))
    for item in prepared.values():
        validate_prepared(item)
    data_files = sorted(path for path in Path(data_root).rglob("*") if path.is_file())
    duplicate_groups = _curve_duplicates(grids)
    inventory = {
        "config": config,
        "source_files": {"count": len(data_files),
                         "total_bytes": sum(path.stat().st_size for path in data_files),
                         "largest_bytes": max(path.stat().st_size for path in data_files),
                         "license_notice_found": any("license" in path.name.lower() or "licence" in path.name.lower() for path in data_files)},
        "clients": [
            {
                "client_id": grid.client_id,
                "name": grid.name,
                "nodes": grid.num_nodes,
                "branches": grid.physical_branch_count,
                "raw_branch_rows": grid.raw_branch_count,
                "directed_edge_index_columns": grid.num_edges,
                "load_nodes": int(grid.load_mask.sum()),
                "component_sizes": sorted((len(component) for component in _components(grid)), reverse=True),
                "isolated_nodes": [grid.node_ids[component[0]] for component in _components(grid) if len(component) == 1],
                "branch_quality": _branch_quality(grid),
                "q_load_quality": _q_load_quality(grid, data_root),
                "timestamps": [str(grid.timestamps[0]), str(grid.timestamps[-1])],
                "samples": int(grid.timestamps.size),
                "train_windows": len(prepared[client_id].train),
                "val_windows": len(prepared[client_id].val),
                "test_windows": len(prepared[client_id].test),
                "split_boundaries": [str(grid.timestamps[index]) for index in prepared[client_id].split_indices],
                "metadata": grid.metadata,
            }
            for client_id, grid in grids.items()
        ],
        "exact_duplicate_active_curve_groups": duplicate_groups,
        "interface": {
            "X": "[num_windows, 48, num_nodes, 1]",
            "Y": "[num_windows, 24, num_nodes]",
            "edge_index": "[2, 2 * active_physical_branches] (bidirectional)",
            "load_mask": "[num_nodes]",
            "unit": "kW",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
