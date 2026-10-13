import pytest
import torch

from scripts.compare_validation import check_protocol, find_best
from src.training.engine import read_config


def checkpoint(**changes):
    return dict({"kind": "best", "smoke": False, "seed": 42,
                 "model_name": "stfednet_residual", "client_id": "lv_39",
                 "config": read_config(), "data_signature": "data2021"}, **changes)


def test_comparison_rejects_data_or_training_protocol_changes():
    reference = checkpoint()
    check_protocol(reference, checkpoint())
    with pytest.raises(ValueError, match="data/signature"):
        check_protocol(reference, checkpoint(data_signature="different"))
    with pytest.raises(ValueError, match="batch_size"):
        check_protocol(reference, checkpoint(config=dict(read_config(), batch_size=16)))


def test_find_validation_checkpoint_skips_smoke_and_other_seed(tmp_path):
    root = tmp_path / "stfednet_residual" / "lv_39"
    for name, metadata in [("formal", checkpoint()), ("smoke", checkpoint(smoke=True)),
                           ("otherseed", checkpoint(seed=0))]:
        directory = root / name
        directory.mkdir(parents=True)
        torch.save(metadata, directory / "best.pt")
    path, loaded = find_best(tmp_path, "stfednet_residual", "lv_39", 42)
    assert path == root / "formal" / "best.pt" and not loaded["smoke"]
    with pytest.raises(FileNotFoundError):
        find_best(tmp_path, "stfednet_residual", "lv_39", 123)
