from functools import lru_cache
from pathlib import Path

import numpy as np

from src.data.loader import GridData, load_all_grids, prepare_all, prepare_client, validate_prepared


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _grids():
    return load_all_grids(ROOT / "Datasets")


def test_all_five_grids_load_and_have_valid_topology():
    grids = _grids()
    assert len(grids) == 5
    assert sorted(grid.num_nodes for grid in grids.values()) == [39, 50, 56, 76, 80]
    assert sorted(grid.physical_branch_count for grid in grids.values()) == [38, 49, 55, 74, 79]
    assert grids["industrial_mvlv"].raw_branch_count == 75
    for grid in grids.values():
        assert grid.edge_index.shape[0] == 2
        assert grid.edge_index.min() >= 0
        assert grid.edge_index.max() < grid.num_nodes
        assert grid.load_matrix.shape == (8760, grid.num_nodes)
        assert grid.timestamps[0] == np.datetime64("2021-01-01T00:00", "h")
        assert grid.timestamps[-1] == np.datetime64("2021-12-31T23:00", "h")
        assert np.isfinite(grid.load_matrix).all()
        assert grid.load_mask.any()


def test_prepare_splits_are_disjoint_and_reproducible():
    grids = _grids()
    prepared_a = prepare_all(grids)
    prepared_b = prepare_all(grids)
    for client_id in grids:
        a, b = prepared_a[client_id], prepared_b[client_id]
        validate_prepared(a)
        assert a.split_indices == b.split_indices
        assert np.array_equal(a.train.starts, b.train.starts)
        assert np.array_equal(a.val.starts, b.val.starts)
        assert np.array_equal(a.test.starts, b.test.starts)
        assert a.train.x_shape == (48, a.grid.num_nodes, 1)
        assert a.train.y_shape == (24, a.grid.num_nodes)
        assert a.scaler.fit_end == a.grid.timestamps[a.split_indices[0] - 1]
        # Float32 storage introduces a small reduction error for the scaler;
        # the fit itself is performed over exactly the training rows.
        assert np.allclose(a.values[: a.split_indices[0]].mean(axis=0), 0.0, atol=2e-5)


def test_duplicate_load_columns_are_aggregated():
    grid = _grids()["lv_39"]
    assert grid.metadata["duplicate_load_columns"] == {"30": 2}
    bus_index = grid.node_ids.index("30")
    # Source first row contains 30 twice: 0.004714 + 0.000758 MW = 5.472 kW.
    assert np.isclose(grid.load_matrix[0, bus_index], 5.472, atol=1e-5)


def test_industrial_duplicate_rows_are_removed():
    grid = _grids()["industrial_mvlv"]
    assert grid.metadata["duplicate_timestamp_rows_removed"] == 45
    assert grid.metadata["conflicting_duplicate_rows"] == 0


def test_masks_and_graph_sample_are_aligned():
    grids = _grids()
    for grid in grids.values():
        assert np.all(grid.load_matrix[:, ~grid.load_mask] == 0)
        assert grid.load_mask.sum() == len({grid.node_ids[i] for i in np.flatnonzero(grid.load_mask)})
        prepared = prepare_client(grid)
        sample = prepared.sample("train", 0)
        assert sample["client_id"] == grid.client_id
        assert sample["X"].shape == (48, grid.num_nodes, 1)
        assert sample["Y"].shape == (24, grid.num_nodes)
        assert np.array_equal(sample["edge_index"], grid.edge_index)
        assert np.array_equal(sample["load_mask"], grid.load_mask)
        assert np.allclose(sample["scaler"].inverse_transform(sample["Y"]),
                           grid.load_matrix[48:72], atol=1e-5)


def test_scaler_is_unchanged_by_future_values():
    timestamps = np.arange(np.datetime64("2021-01-01T00", "h"),
                           np.datetime64("2021-01-21T00", "h"), np.timedelta64(1, "h"))
    base = np.arange(timestamps.size, dtype=np.float32)[:, None]
    altered = base.copy()
    altered[int(timestamps.size * 0.70):] += 1_000_000
    common = dict(client_id="synthetic", name="synthetic", node_ids=["1"],
                  edge_index=np.empty((2, 0), dtype=np.int64), timestamps=timestamps,
                  load_mask=np.asarray([True]))
    first = prepare_client(GridData(load_matrix=base, **common))
    second = prepare_client(GridData(load_matrix=altered, **common))
    assert np.array_equal(first.scaler.mean, second.scaler.mean)
    assert np.array_equal(first.scaler.scale, second.scaler.scale)
