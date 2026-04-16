import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

import mlflow

DEFAULT_EXPERIMENT_NAME = "rakuten-text-classification"
MLFLOW_DIR = Path("mlruns")


def configure_mlflow() -> dict[str, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        tracking_uri = MLFLOW_DIR.resolve().as_uri()

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
    }


def log_params(params: Mapping[str, object]) -> None:
    sanitized = {key: str(value) for key, value in params.items()}
    mlflow.log_params(sanitized)


def log_metrics(metrics: Mapping[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_tags(tags: Mapping[str, str]) -> None:
    mlflow.set_tags(tags)


def log_text(content: str, artifact_file: str) -> None:
    mlflow.log_text(content, artifact_file)


@contextmanager
def start_training_run(run_name: str) -> Iterator[tuple[mlflow.ActiveRun, dict[str, str]]]:
    mlflow_context = configure_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        yield run, mlflow_context


def log_artifact_if_exists(path: Path, artifact_path: str | None = None) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_artifacts_if_exists(path: Path, artifact_path: str | None = None) -> None:
    if path.exists():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
