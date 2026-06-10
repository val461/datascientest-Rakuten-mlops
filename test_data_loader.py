import json

import pandas as pd
import pytest

from src import data_loader
from src.simulation import data_growth, split_manager


@pytest.fixture
def isolated_dataset(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    split_dir = tmp_path / "splits"
    raw_dir.mkdir()

    X = pd.DataFrame(
        {
            "designation": [f"product {index}" for index in range(100)],
            "description": [f"description {index}" for index in range(100)],
            "productid": range(1000, 1100),
            "imageid": range(2000, 2100),
        },
        index=range(100),
    )
    y = pd.DataFrame(
        {"prdtypecode": [10] * 50 + [20] * 50},
        index=range(100),
    )

    x_path = raw_dir / "X_train.csv"
    y_path = raw_dir / "Y_train.csv"
    X.to_csv(x_path, index_label=data_loader.INDEX_COLUMN)
    y.to_csv(y_path, index_label=data_loader.INDEX_COLUMN)

    monkeypatch.setattr(data_loader, "X_TRAIN_PATH", x_path)
    monkeypatch.setattr(data_loader, "Y_TRAIN_PATH", y_path)
    monkeypatch.setattr(
        split_manager,
        "VALIDATION_INDICES_PATH",
        split_dir / "validation_indices.json",
    )
    monkeypatch.setattr(
        split_manager,
        "STREAM_INDICES_PATH",
        split_dir / "stream_indices.json",
    )
    monkeypatch.setattr(
        split_manager,
        "SPLIT_METADATA_PATH",
        split_dir / "split_metadata.json",
    )
    return x_path, y_path


def test_split_is_persisted_and_reused(isolated_dataset):
    X, y = data_loader.load_training_csv()

    validation_first, stream_first, metadata_first = (
        split_manager.load_or_create_simulation_split(X, y)
    )
    validation_second, stream_second, metadata_second = (
        split_manager.load_or_create_simulation_split(X, y)
    )

    assert validation_first == validation_second
    assert stream_first == stream_second
    assert metadata_first == metadata_second
    assert len(validation_first) == 20
    assert len(stream_first) == 80
    assert set(validation_first).isdisjoint(stream_first)
    assert set(validation_first) | set(stream_first) == set(X.index)


def test_steps_are_cumulative_and_keep_fixed_validation(isolated_dataset):
    X_step_0, X_valid_0, y_step_0, y_valid_0, metadata_0 = (
        data_growth.load_simulation_split(step=0)
    )
    X_step_1, X_valid_1, _, y_valid_1, metadata_1 = (
        data_growth.load_simulation_split(step=1)
    )
    X_step_10, X_valid_10, _, y_valid_10, metadata_10 = (
        data_growth.load_simulation_split(step=10)
    )

    assert len(X_step_0) == len(y_step_0) == 40
    assert len(X_step_1) == 44
    assert len(X_step_10) == 80
    assert set(X_step_0.index) < set(X_step_1.index) < set(X_step_10.index)
    assert X_valid_0.index.tolist() == X_valid_1.index.tolist()
    assert X_valid_0.index.tolist() == X_valid_10.index.tolist()
    assert y_valid_0.index.tolist() == y_valid_1.index.tolist()
    assert y_valid_0.index.tolist() == y_valid_10.index.tolist()
    assert metadata_0["available_ratio"] == 0.50
    assert metadata_1["available_ratio"] == 0.55
    assert metadata_10["available_ratio"] == 1.0
    assert metadata_10["future_rows"] == 0


@pytest.mark.parametrize("step", [-1, 11])
def test_invalid_steps_are_rejected(step):
    with pytest.raises(ValueError):
        data_growth.simulation_ratio(step)


def test_non_integer_step_is_rejected():
    with pytest.raises(TypeError):
        data_growth.simulation_ratio(0.5)


def test_dataset_change_invalidates_existing_split(isolated_dataset):
    x_path, _ = isolated_dataset
    X, y = data_loader.load_training_csv()
    split_manager.load_or_create_simulation_split(X, y)

    changed_X = pd.read_csv(x_path, index_col=data_loader.INDEX_COLUMN)
    changed_X.loc[0, "designation"] = "changed product"
    changed_X.to_csv(x_path, index_label=data_loader.INDEX_COLUMN)

    changed_X, unchanged_y = data_loader.load_training_csv()
    with pytest.raises(ValueError, match="dataset brut a change"):
        split_manager.load_or_create_simulation_split(changed_X, unchanged_y)


def test_split_metadata_is_written(isolated_dataset):
    X, y = data_loader.load_training_csv()
    split_manager.load_or_create_simulation_split(X, y)

    metadata = json.loads(
        split_manager.SPLIT_METADATA_PATH.read_text(encoding="utf-8")
    )
    assert metadata["split_version"] == 1
    assert metadata["total_rows"] == 100
    assert metadata["validation_rows"] == 20
    assert metadata["stream_rows"] == 80
    assert metadata["dataset_fingerprint"].startswith("sha256:")
