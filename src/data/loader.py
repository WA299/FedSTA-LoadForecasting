"""Load and validate the Norway distribution-grid data.

The source data contains two formats:

* the four low-voltage cases use MATPOWER-like CSV files and hourly ``p_load``
  values for 2021;
* the industrial case uses semicolon-separated hourly ``Load_kWh`` files,
  one file per load bus, plus ``bus.csv`` and ``branch.csv``.

The public API deliberately uses NumPy arrays so the data layer can be used
before PyTorch is installed.  ``PreparedClient.as_torch`` and
``WindowDataset.as_torch`` provide the later PyTorch bridge.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_START = np.datetime64("2021-01-01T00:00", "h")
DEFAULT_END = np.datetime64("2021-12-31T23:00", "h")
DEFAULT_INPUT_HOURS = 48
DEFAULT_HORIZON_HOURS = 24


def _read_csv(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _to_datetime64(value: str) -> np.datetime64:
    # ISO timestamps are used by the LV files.  Industrial files use
    # day/month/year and are parsed explicitly below.
    return np.datetime64(value.replace(" ", "T"), "h")


def _target_timestamps(start: np.datetime64 = DEFAULT_START,
                       end: np.datetime64 = DEFAULT_END) -> np.ndarray:
    result = np.arange(start, end + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    if result.size == 0:
        raise ValueError("The requested data period is empty")
    return result


def _normalise_id(value: object) -> str:
    text = str(value).strip()
    # MATPOWER bus IDs are integer-like.  Keeping the canonical decimal form
    # avoids mismatches between a CSV header ("30.0") and bus.csv ("30").
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def _parse_float(value: str) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def _deduplicate_series(times: Sequence[np.datetime64], values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Sort a series and collapse repeated timestamps.

    Exact repeated rows are a known artifact in the industrial files. They
    are kept once. Conflicting rows are rejected because a single bus/hour
    value cannot be resolved safely from the source data alone.
    """
    grouped: Dict[np.datetime64, List[float]] = {}
    for time, value in zip(times, values):
        grouped.setdefault(time, []).append(float(value))
    duplicate_count = sum(max(0, len(items) - 1) for items in grouped.values())
    conflict_count = sum(1 for items in grouped.values() if len(items) > 1 and not np.allclose(items, items[0]))
    if conflict_count:
        raise ValueError(f"Found {conflict_count} conflicting duplicate load timestamps")
    ordered = sorted(grouped)
    collapsed = [grouped[time][0] for time in ordered]
    return np.asarray(ordered, dtype="datetime64[h]"), np.asarray(collapsed, dtype=np.float64), duplicate_count, conflict_count


def _fill_to_index(times: np.ndarray, values: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Align one series to the requested hourly index; reject missing hours.

    G0 intentionally does not impute gaps, because interpolating a training
    hour from a later validation/test hour would leak future information.
    """
    result = np.full(target.shape, np.nan, dtype=np.float64)
    positions = {time: i for i, time in enumerate(target)}
    for time, value in zip(times, values):
        if time in positions:
            result[positions[time]] = value
    missing = int(np.isnan(result).sum())
    if missing:
        raise ValueError(f"Load series is missing {missing} hourly values in the requested period")
    return result, missing, 0


def _load_lv_case(path: Path, start: np.datetime64, end: np.datetime64,
                  client_id: str) -> "GridData":
    bus_rows = _read_csv(path / "mpc_bus.csv")
    if not bus_rows:
        raise ValueError(f"No bus rows found in {path / 'mpc_bus.csv'}")
    node_ids = [_normalise_id(row.get("bus_i", "")) for row in bus_rows]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError(f"Duplicate bus IDs in {path / 'mpc_bus.csv'}")
    node_to_index = {node: index for index, node in enumerate(node_ids)}

    branch_rows = _read_csv(path / "mpc_branch.csv")
    raw_edges: List[Tuple[int, int]] = []
    disabled = 0
    for row in branch_rows:
        if str(row.get("status", "1")).strip() in {"0", "0.0"}:
            disabled += 1
            continue
        source = _normalise_id(row["fbus"])
        target = _normalise_id(row["tbus"])
        if source not in node_to_index or target not in node_to_index:
            raise ValueError(f"Branch endpoint ({source}, {target}) is not present in bus.csv")
        raw_edges.append((node_to_index[source], node_to_index[target]))
    edge_index = _bidirectional_edges(raw_edges)

    p_rows = list(csv.reader((path / "p_load.csv").open(encoding="utf-8-sig", newline="")))
    if not p_rows or not p_rows[0] or p_rows[0][0].strip().lower() != "date":
        raise ValueError(f"Unexpected p_load header in {path}")
    headers = [_normalise_id(item) for item in p_rows[0][1:]]
    times = np.asarray([_to_datetime64(row[0]) for row in p_rows[1:]], dtype="datetime64[h]")
    target_times = _target_timestamps(start, end)
    if len(set(times.tolist())) != times.size:
        raise ValueError(f"Low-voltage timestamps are not unique in {path / 'p_load.csv'}")
    if times.size and np.any(times[1:] <= times[:-1]):
        raise ValueError(f"Low-voltage timestamps are not strictly increasing in {path / 'p_load.csv'}")

    # Duplicate CSV columns are multiple users attached to one bus.  Values
    # are MW in the supplied MATPOWER-style files; expose a unified kW API.
    by_bus: Dict[str, List[int]] = {}
    for column, bus in enumerate(headers):
        if bus not in node_to_index:
            raise ValueError(f"Load column bus {bus} is not in mpc_bus.csv for {path}")
        by_bus.setdefault(bus, []).append(column)
    raw = np.asarray([[float(value) for value in row[1:]] for row in p_rows[1:]], dtype=np.float64)
    load = np.zeros((target_times.size, len(node_ids)), dtype=np.float64)
    missing_before_fill = 0
    for bus, columns in by_bus.items():
        series = raw[:, columns].sum(axis=1)
        aligned, missing, _ = _fill_to_index(times, series, target_times)
        load[:, node_to_index[bus]] = aligned * 1000.0
        missing_before_fill += missing
    mask = np.asarray([node in by_bus for node in node_ids], dtype=bool)
    metadata = {
        "format": "matpower_csv",
        "source_unit": "MW (assumed from MATPOWER-style p_load)",
        "unit": "kW",
        "conversion": "p_load * 1000",
        "base_mva_file": str(path / "mpc_base_mva.csv"),
        "raw_branch_count": len(branch_rows),
        "disabled_branch_count": disabled,
        "duplicate_load_columns": {bus: len(columns) for bus, columns in by_bus.items() if len(columns) > 1},
        "missing_values_before_fill": missing_before_fill,
        "source_time_start": str(times.min()) if times.size else None,
        "source_time_end": str(times.max()) if times.size else None,
    }
    return GridData(client_id, path.name, node_ids, edge_index, load, target_times, mask, metadata)


def _bidirectional_edges(edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    directed = list(edges) + [(target, source) for source, target in edges]
    return np.asarray(directed, dtype=np.int64).T


def _load_industrial_case(path: Path, start: np.datetime64, end: np.datetime64,
                          client_id: str) -> "GridData":
    bus_rows = _read_csv(path / "bus.csv", delimiter=";")
    node_ids = [_normalise_id(row["BUS_I"]) for row in bus_rows]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError(f"Duplicate bus IDs in {path / 'bus.csv'}")
    node_to_index = {node: index for index, node in enumerate(node_ids)}
    branch_rows = _read_csv(path / "branch.csv", delimiter=";")
    raw_edges: List[Tuple[int, int]] = []
    disabled = 0
    for row in branch_rows:
        if str(row.get("BR_STATUS", "1")).strip() in {"0", "0.0"}:
            disabled += 1
            continue
        source = _normalise_id(row["F_BUS"])
        target = _normalise_id(row["T_BUS"])
        if source not in node_to_index or target not in node_to_index:
            raise ValueError(f"Industrial branch endpoint ({source}, {target}) is not in bus.csv")
        raw_edges.append((node_to_index[source], node_to_index[target]))
    edge_index = _bidirectional_edges(raw_edges)
    target_times = _target_timestamps(start, end)
    load = np.zeros((target_times.size, len(node_ids)), dtype=np.float64)
    source_files = sorted(path.glob("*.txt"))
    if not source_files:
        raise ValueError(f"No industrial load files found in {path}")
    bus_file_count: Dict[str, int] = {}
    duplicate_rows = 0
    conflicting_rows = 0
    missing_before_fill = 0
    source_starts: Dict[str, str] = {}
    source_ends: Dict[str, str] = {}
    for source_file in source_files:
        rows = _read_csv(source_file, delimiter=";")
        if not rows:
            continue
        file_bus = _normalise_id(rows[0].get("Bus_ID", ""))
        if file_bus not in node_to_index:
            raise ValueError(f"Load file {source_file.name} references unknown bus {file_bus}")
        times = []
        values = []
        for row in rows:
            current_bus = _normalise_id(row.get("Bus_ID", ""))
            if current_bus != file_bus:
                raise ValueError(f"Mixed Bus_ID values in {source_file}")
            times.append(np.datetime64(datetime.strptime(row["Timestamp"], "%d/%m/%Y %H:%M:%S"), "h"))
            values.append(_parse_float(row["Load_kWh"]))
        times, values, duplicate_count, conflict_count = _deduplicate_series(times, values)
        source_starts[source_file.name] = str(times[0])
        source_ends[source_file.name] = str(times[-1])
        duplicate_rows += duplicate_count
        conflicting_rows += conflict_count
        aligned, missing, _ = _fill_to_index(times, values, target_times)
        missing_before_fill += missing
        load[:, node_to_index[file_bus]] += aligned  # kWh over one hour == average kW
        bus_file_count[file_bus] = bus_file_count.get(file_bus, 0) + 1
    mask = np.asarray([node in bus_file_count for node in node_ids], dtype=bool)
    metadata = {
        "format": "industrial_semicolon_txt",
        "source_unit": "kWh per hourly interval",
        "unit": "kW",
        "conversion": "numerically unchanged because interval is one hour",
        "raw_branch_count": len(branch_rows),
        "disabled_branch_count": disabled,
        "load_file_count": len(source_files),
        "duplicate_load_files": {bus: count for bus, count in bus_file_count.items() if count > 1},
        "duplicate_timestamp_rows_removed": duplicate_rows,
        "conflicting_duplicate_rows": conflicting_rows,
        "missing_values_before_fill": missing_before_fill,
        "source_time_start": min(source_starts.values()),
        "source_time_end": max(source_ends.values()),
        "source_start_variants": {name: stamp for name, stamp in source_starts.items() if stamp != min(source_starts.values())},
    }
    return GridData(client_id, path.name, node_ids, edge_index, load, target_times, mask, metadata)


@dataclass
class GridData:
    """A single physical grid and its aligned node-level load matrix."""

    client_id: str
    name: str
    node_ids: List[str]
    edge_index: np.ndarray
    load_matrix: np.ndarray
    timestamps: np.ndarray
    load_mask: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.edge_index = np.asarray(self.edge_index, dtype=np.int64)
        self.load_matrix = np.asarray(self.load_matrix, dtype=np.float32)
        self.timestamps = np.asarray(self.timestamps, dtype="datetime64[h]")
        self.load_mask = np.asarray(self.load_mask, dtype=bool)
        validate_grid(self)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def node_index(self) -> Dict[str, int]:
        """Map the source bus identifier to its row in every node tensor."""
        return {node_id: index for index, node_id in enumerate(self.node_ids)}

    @property
    def node_index(self) -> Dict[str, int]:
        """Map the source bus identifier to its row in every node tensor."""
        return {node_id: index for index, node_id in enumerate(self.node_ids)}

    @property
    def raw_branch_count(self) -> int:
        return int(self.metadata.get("raw_branch_count", self.num_edges // 2))

    @property
    def physical_branch_count(self) -> int:
        """Number of active branches represented by the bidirectional graph."""
        return self.num_edges // 2

    def as_torch(self) -> Dict[str, object]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised when torch is absent
            raise ImportError("PyTorch is required for as_torch(); install it from pytorch.org") from error
        return {
            "load": torch.from_numpy(self.load_matrix.copy()),
            "edge_index": torch.from_numpy(self.edge_index.copy()),
            "load_mask": torch.from_numpy(self.load_mask.copy()),
            "timestamps": self.timestamps.copy(),
            "client_id": self.client_id,
        }


def validate_grid(grid: GridData) -> None:
    if grid.edge_index.ndim != 2 or grid.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if grid.edge_index.size and (grid.edge_index.min() < 0 or grid.edge_index.max() >= grid.num_nodes):
        raise ValueError(f"edge_index endpoint out of range for {grid.name}")
    if grid.load_matrix.shape != (grid.timestamps.size, grid.num_nodes):
        raise ValueError(f"load matrix shape {grid.load_matrix.shape} does not match timestamps/nodes")
    if grid.load_mask.shape != (grid.num_nodes,):
        raise ValueError("load_mask must have one entry per node")
    if np.any(~np.isfinite(grid.load_matrix)):
        raise ValueError(f"NaN or Inf in aligned load matrix for {grid.name}")
    if grid.timestamps.size < 2 or np.any(np.diff(grid.timestamps) != np.timedelta64(1, "h")):
        raise ValueError(f"timestamps are not continuous hourly samples for {grid.name}")


def discover_grids(data_root: os.PathLike[str] | str) -> List[Tuple[str, Path]]:
    root = Path(data_root)
    lv_root = root / "norway_4_lv_grids"
    industrial = root / "norway_industrial_mvlv"
    discovered: List[Tuple[str, Path]] = []
    for path in sorted(lv_root.iterdir()) if lv_root.exists() else []:
        if path.is_dir() and (path / "mpc_bus.csv").exists():
            discovered.append((f"lv_{path.name.split('_')[0]}", path))
    if industrial.exists() and (industrial / "bus.csv").exists():
        discovered.append(("industrial_mvlv", industrial))
    if len(discovered) != 5:
        raise ValueError(f"Expected five grid directories, found {len(discovered)} under {root}")
    return discovered


def load_grid(path: os.PathLike[str] | str, client_id: Optional[str] = None,
              start: np.datetime64 = DEFAULT_START, end: np.datetime64 = DEFAULT_END) -> GridData:
    path = Path(path)
    client_id = client_id or path.name
    if (path / "mpc_bus.csv").exists():
        return _load_lv_case(path, start, end, client_id)
    if (path / "bus.csv").exists() and (path / "branch.csv").exists():
        return _load_industrial_case(path, start, end, client_id)
    raise ValueError(f"Unrecognised grid directory: {path}")


def load_all_grids(data_root: os.PathLike[str] | str = "Datasets",
                   start: np.datetime64 = DEFAULT_START,
                   end: np.datetime64 = DEFAULT_END) -> Dict[str, GridData]:
    return {client_id: load_grid(path, client_id, start, end) for client_id, path in discover_grids(data_root)}


@dataclass
class StandardScaler:
    """Per-node standardization fitted only on the training time range."""

    mean: np.ndarray
    scale: np.ndarray
    fit_start: np.datetime64
    fit_end: np.datetime64

    @classmethod
    def fit(cls, values: np.ndarray, timestamps: np.ndarray, end_index: int) -> "StandardScaler":
        if end_index <= 0 or end_index > values.shape[0]:
            raise ValueError("Invalid training boundary for scaler")
        train = np.asarray(values[:end_index], dtype=np.float64)
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32), timestamps[0], timestamps[end_index - 1])

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.scale + self.mean).astype(np.float32)


@dataclass
class WindowDataset:
    """Lazy sliding-window view with ``X=[N, lookback, nodes, 1]`` and
    ``Y=[N, horizon, nodes]`` semantics."""

    values: np.ndarray
    timestamps: np.ndarray
    starts: np.ndarray
    input_hours: int
    horizon_hours: int
    load_mask: np.ndarray
    as_torch_tensors: bool = False

    def __len__(self) -> int:
        return int(self.starts.size)

    @property
    def num_nodes(self) -> int:
        return int(self.values.shape[1])

    @property
    def x_shape(self) -> Tuple[int, int, int, int]:
        return (self.input_hours, self.num_nodes, 1)

    @property
    def y_shape(self) -> Tuple[int, int]:
        return (self.horizon_hours, self.num_nodes)

    @property
    def target_timestamps(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, self.horizon_hours), dtype="datetime64[h]")
        return np.stack([self.timestamps[s + self.input_hours:s + self.input_hours + self.horizon_hours] for s in self.starts])

    def __getitem__(self, item: int) -> Tuple[object, object]:
        start = int(self.starts[item])
        x = self.values[start:start + self.input_hours, :, None]
        y = self.values[start + self.input_hours:start + self.input_hours + self.horizon_hours, :]
        if self.as_torch_tensors:
            try:
                import torch
            except ImportError as error:  # pragma: no cover
                raise ImportError("PyTorch is required when as_torch_tensors=True") from error
            return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
        return x.copy(), y.copy()

    def as_torch(self) -> "WindowDataset":
        self.as_torch_tensors = True
        return self


@dataclass
class PreparedClient:
    grid: GridData
    scaler: StandardScaler
    values: np.ndarray
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    split_indices: Tuple[int, int]

    @property
    def client_id(self) -> str:
        return self.grid.client_id

    def as_torch(self) -> "PreparedClient":
        self.train.as_torch()
        self.val.as_torch()
        self.test.as_torch()
        return self

    def sample(self, split: str, index: int, torch_tensors: bool = False) -> Dict[str, object]:
        """Return one graph-aware forecasting sample with a shared API.

        `timestamps` corresponds to the 24 target hours; `history_timestamps`
        corresponds to the 48 input hours. The source `node_ids` and
        `node_index` in `grid` define every node axis in the returned tensors.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        dataset = getattr(self, split)
        start = int(dataset.starts[index])
        x, y = dataset[index]
        edge_index = self.grid.edge_index.copy()
        load_mask = self.grid.load_mask.copy()
        if torch_tensors:
            try:
                import torch
            except ImportError as error:  # pragma: no cover
                raise ImportError("PyTorch is required when torch_tensors=True") from error
            if not isinstance(x, torch.Tensor):
                x, y = torch.from_numpy(x), torch.from_numpy(y)
            edge_index = torch.from_numpy(edge_index)
            load_mask = torch.from_numpy(load_mask)
        return {
            "X": x,
            "Y": y,
            "edge_index": edge_index,
            "load_mask": load_mask,
            "history_timestamps": self.grid.timestamps[start:start + dataset.input_hours].copy(),
            "timestamps": self.grid.timestamps[start + dataset.input_hours:start + dataset.input_hours + dataset.horizon_hours].copy(),
            "scaler": self.scaler,
            "client_id": self.client_id,
        }


def _window_starts(total_steps: int, input_hours: int, horizon_hours: int,
                   target_start: int, target_end: int) -> np.ndarray:
    last_start = total_steps - input_hours - horizon_hours
    if last_start < 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.arange(last_start + 1, dtype=np.int64)
    target_begin = candidates + input_hours
    target_finish = target_begin + horizon_hours
    return candidates[(target_begin >= target_start) & (target_finish <= target_end)]


def prepare_client(grid: GridData, input_hours: int = DEFAULT_INPUT_HOURS,
                   horizon_hours: int = DEFAULT_HORIZON_HOURS,
                   train_ratio: float = 0.70, val_ratio: float = 0.15) -> PreparedClient:
    if input_hours <= 0 or horizon_hours <= 0:
        raise ValueError("input_hours and horizon_hours must be positive")
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must leave a positive test split")
    total_steps = grid.timestamps.size
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))
    scaler = StandardScaler.fit(grid.load_matrix, grid.timestamps, train_end)
    values = scaler.transform(grid.load_matrix)
    common = dict(values=values, timestamps=grid.timestamps, input_hours=input_hours,
                  horizon_hours=horizon_hours, load_mask=grid.load_mask)
    train = WindowDataset(starts=_window_starts(total_steps, input_hours, horizon_hours, 0, train_end), **common)
    val = WindowDataset(starts=_window_starts(total_steps, input_hours, horizon_hours, train_end, val_end), **common)
    test = WindowDataset(starts=_window_starts(total_steps, input_hours, horizon_hours, val_end, total_steps), **common)
    return PreparedClient(grid, scaler, values, train, val, test, (train_end, val_end))


def prepare_all(grids: Mapping[str, GridData], **kwargs: object) -> Dict[str, PreparedClient]:
    return {client_id: prepare_client(grid, **kwargs) for client_id, grid in grids.items()}


def validate_prepared(prepared: PreparedClient) -> None:
    train_end, val_end = prepared.split_indices
    splits = [prepared.train, prepared.val, prepared.test]
    target_sets = [set(split.target_timestamps.reshape(-1).tolist()) for split in splits]
    for left in range(3):
        for right in range(left + 1, 3):
            if target_sets[left] & target_sets[right]:
                raise ValueError("Target timestamps overlap between chronological splits")
    if prepared.scaler.fit_end != prepared.grid.timestamps[train_end - 1]:
        raise ValueError("Scaler fit range is not limited to the training period")
    for split in splits:
        if len(split):
            x, y = split[0]
            if x.shape != split.x_shape or y.shape != split.y_shape:
                raise ValueError("Window shape mismatch")
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError("NaN or Inf in prepared windows")


__all__ = [
    "DEFAULT_END", "DEFAULT_HORIZON_HOURS", "DEFAULT_INPUT_HOURS", "DEFAULT_START",
    "GridData", "PreparedClient", "StandardScaler", "WindowDataset",
    "discover_grids", "load_all_grids", "load_grid", "prepare_all", "prepare_client",
    "validate_grid", "validate_prepared",
]
