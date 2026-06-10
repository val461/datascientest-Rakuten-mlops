from contextlib import contextmanager
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from src import trainer


def simulation_data(step: int):
    X_train = pd.DataFrame(
        {"designation": ["a", "b"], "description": ["c", "d"]}
    )
    X_valid = pd.DataFrame(
        {"designation": ["e"], "description": ["f"]}
    )
    y_train = pd.Series([10, 20])
    y_valid = pd.Series([10])
    metadata = {
        "simulation_step": step,
        "available_ratio": 0.5 + step * 0.05,
        "split_version": 1,
    }
    return X_train, X_valid, y_train, y_valid, metadata


def test_simulation_step_uses_history_path_without_deployment(
    tmp_path,
    monkeypatch,
):
    history_dir = tmp_path / "history"
    deployed_path = tmp_path / "model.joblib"
    calls = []

    monkeypatch.setattr(trainer, "MODEL_HISTORY_DIR", history_dir)
    monkeypatch.setattr(trainer, "MODEL_PATH", deployed_path)
    monkeypatch.setattr(
        trainer,
        "load_simulation_split",
        lambda step: simulation_data(step),
    )

    def fake_train_and_save(*args, **kwargs):
        calls.append(kwargs)
        return {"deployed": kwargs["deployment_path"] is not None}

    monkeypatch.setattr(trainer, "_train_and_save", fake_train_and_save)

    result = trainer.train_and_save_simulation_model(step=4, deploy=False)

    assert result["deployed"] is False
    assert calls[0]["model_path"] == history_dir / "model_step_04.joblib"
    assert calls[0]["deployment_path"] is None


def test_simulation_step_deploys_only_when_requested(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    deployed_path = tmp_path / "model.joblib"
    calls = []

    monkeypatch.setattr(trainer, "MODEL_HISTORY_DIR", history_dir)
    monkeypatch.setattr(trainer, "MODEL_PATH", deployed_path)
    monkeypatch.setattr(
        trainer,
        "load_simulation_split",
        lambda step: simulation_data(step),
    )

    def fake_train_and_save(*args, **kwargs):
        calls.append(kwargs)
        return {"deployed": kwargs["deployment_path"] is not None}

    monkeypatch.setattr(trainer, "_train_and_save", fake_train_and_save)

    result = trainer.train_and_save_simulation_model(step=10, deploy=True)

    assert result["deployed"] is True
    assert calls[0]["model_path"] == history_dir / "model_step_10.joblib"
    assert calls[0]["deployment_path"] == deployed_path


def test_atomic_save_preserves_previous_model_when_validation_fails(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "model.joblib"
    previous_bundle = {"version": "previous"}
    joblib.dump(previous_bundle, model_path)
    previous_bytes = model_path.read_bytes()

    def fail_validation(path):
        raise ValueError("invalid serialized model")

    monkeypatch.setattr(trainer.joblib, "load", fail_validation)

    with pytest.raises(ValueError, match="invalid serialized model"):
        trainer.save_model({"version": "new"}, model_path)

    assert model_path.exists()
    assert model_path.read_bytes() == previous_bytes

    # Read the untouched deployed file with the original joblib implementation.
    monkeypatch.undo()
    assert joblib.load(model_path) == previous_bundle


def test_log_training_run_records_simulation_metadata(monkeypatch, tmp_path):
    recorded = {
        "run_name": None,
        "tags": None,
        "params": None,
        "metrics": None,
        "text": None,
        "model_artifact": None,
        "preprocessing_artifact": None,
    }

    @contextmanager
    def fake_start_training_run(run_name):
        recorded["run_name"] = run_name
        run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
        yield run, {
            "tracking_uri": "file:///tmp/mlruns",
            "experiment_name": "test-experiment",
        }

    monkeypatch.setattr(trainer, "start_training_run", fake_start_training_run)
    monkeypatch.setattr(
        trainer,
        "log_tags",
        lambda value: recorded.update(tags=value),
    )
    monkeypatch.setattr(
        trainer,
        "log_params",
        lambda value: recorded.update(params=value),
    )
    monkeypatch.setattr(
        trainer,
        "log_metrics",
        lambda value: recorded.update(metrics=value),
    )
    monkeypatch.setattr(
        trainer,
        "log_text",
        lambda content, path: recorded.update(text=(content, path)),
    )
    monkeypatch.setattr(
        trainer,
        "log_artifact_if_exists",
        lambda path, artifact_path=None: recorded.update(
            model_artifact=(path, artifact_path)
        ),
    )
    monkeypatch.setattr(
        trainer,
        "log_artifacts_if_exists",
        lambda path, artifact_path=None: recorded.update(
            preprocessing_artifact=(path, artifact_path)
        ),
    )

    model_path = tmp_path / "model_step_04.joblib"
    model = trainer.create_model()
    metrics = {
        "accuracy": 0.82,
        "f1_macro": 0.80,
        "f1_weighted": 0.81,
        "classification_report": "report",
    }
    metadata = {
        "simulation_step": 4,
        "available_ratio": 0.70,
        "split_version": 1,
        "available_rows": 47000,
    }

    result = trainer.log_training_run(
        model=model,
        train_rows=47000,
        validation_rows=16000,
        num_classes=27,
        metrics=metrics,
        model_path=model_path,
        run_metadata=metadata,
    )

    assert recorded["run_name"] == "LinearSVC-step-04-ratio-0.70"
    assert recorded["tags"]["simulation"] == "data-growth"
    assert recorded["tags"]["split_version"] == "1"
    assert recorded["params"]["simulation_step"] == 4
    assert recorded["params"]["available_rows"] == 47000
    assert recorded["metrics"] == {
        "accuracy": 0.82,
        "f1_macro": 0.80,
        "f1_weighted": 0.81,
    }
    assert recorded["text"] == (
        "report",
        "reports/classification_report.txt",
    )
    assert recorded["model_artifact"] == (model_path, "model")
    assert recorded["preprocessing_artifact"][1] == "preprocessing"
    assert result["mlflow_run_id"] == "run-123"
