from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

from src.data_loader import load_split
from src.mlflow_tracking import (
    log_artifact_if_exists,
    log_artifacts_if_exists,
    log_metrics,
    log_params,
    log_tags,
    log_text,
    start_training_run,
)
from src.preprocessor import (
    CLASS_WEIGHT,
    PREPROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    fit_transform_features,
    get_preprocessing_metadata,
    save_preprocessing_artifacts,
    transform_features,
)
from src.simulation.data_growth import load_simulation_split
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")
MODEL_HISTORY_DIR = Path("models/history")


def create_model() -> LinearSVC:
    return LinearSVC(
        C=1.0,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )


def train(model: LinearSVC, X_train_vectors, y_train) -> None:
    logger.info("🚀 Entraînement du classifieur texte...")
    model.fit(X_train_vectors, y_train)


def save_model(model_bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    joblib.dump(model_bundle, temporary_path)
    joblib.load(temporary_path)
    temporary_path.replace(path)
    logger.info(f"✅ Modèle sauvegardé : {path}")


def evaluate(model: LinearSVC, X_valid_vectors, y_valid) -> dict:
    predictions = model.predict(X_valid_vectors)
    accuracy = accuracy_score(y_valid, predictions)
    f1_macro = f1_score(y_valid, predictions, average="macro")
    f1_weighted = f1_score(y_valid, predictions, average="weighted")
    report = classification_report(y_valid, predictions, output_dict=False)

    logger.info(f"Accuracy validation : {accuracy:.4f}")
    logger.info(f"F1 macro validation : {f1_macro:.4f}")
    logger.info(f"F1 pondéré validation : {f1_weighted:.4f}")

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
    }


def get_training_params(
    model: LinearSVC,
    train_rows: int,
    validation_rows: int,
    num_classes: int,
    run_metadata: dict | None = None,
) -> dict[str, object]:
    params = {
        "model_name": model.__class__.__name__,
        "model_c": model.C,
        "model_class_weight": model.class_weight,
        "model_random_state": model.random_state,
        "test_size": TEST_SIZE,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "num_classes": num_classes,
        **get_preprocessing_metadata(),
    }
    if run_metadata:
        params.update(run_metadata)
    return params


def log_training_run(
    model: LinearSVC,
    train_rows: int,
    validation_rows: int,
    num_classes: int,
    metrics: dict[str, object],
    model_path: Path,
    run_metadata: dict | None = None,
) -> dict[str, str]:
    if run_metadata and "simulation_step" in run_metadata:
        step = int(run_metadata["simulation_step"])
        ratio = float(run_metadata["available_ratio"])
        run_name = f"{model.__class__.__name__}-step-{step:02d}-ratio-{ratio:.2f}"
    else:
        run_name = f"{model.__class__.__name__}-full-training"

    with start_training_run(run_name=run_name) as (run, mlflow_context):
        tags = {
            "project": "datascientest-rakuten-mlops",
            "task": "product-category-classification",
        }
        if run_metadata and "simulation_step" in run_metadata:
            tags["simulation"] = "data-growth"
            tags["split_version"] = str(run_metadata["split_version"])
        log_tags(tags)
        log_params(
            get_training_params(
                model,
                train_rows,
                validation_rows,
                num_classes,
                run_metadata=run_metadata,
            )
        )
        log_metrics(
            {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_weighted": float(metrics["f1_weighted"]),
            }
        )
        log_text(str(metrics["classification_report"]), "reports/classification_report.txt")
        log_artifact_if_exists(model_path, artifact_path="model")
        log_artifacts_if_exists(PREPROCESSED_DIR, artifact_path="preprocessing")

        return {
            "mlflow_run_id": run.info.run_id,
            **mlflow_context,
        }


def _train_and_save(
    X_train,
    X_valid,
    y_train,
    y_valid,
    model_path: Path,
    deployment_path: Path | None = None,
    run_metadata: dict | None = None,
) -> dict:
    logger.info('Preprocessor fit-transforming X_train. May take 6mn.')
    preprocessor, X_train_vectors = fit_transform_features(X_train)
    logger.info('Preprocessor transforming X_valid. May take 2mn.')
    X_valid_vectors = transform_features(preprocessor, X_valid)

    save_preprocessing_artifacts(preprocessor, X_train_vectors, X_valid_vectors, y_train, y_valid)

    model = create_model()
    logger.info('Training. May take 1mn.')
    train(model, X_train_vectors, y_train)

    logger.info('Evaluating.')
    metrics = evaluate(model, X_valid_vectors, y_valid)

    model_bundle = {
        "classifier": model,
        "preprocessor": preprocessor,
        "metadata": {
            **get_preprocessing_metadata(),
            **(run_metadata or {}),
            "metrics": {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
            },
        },
    }
    save_model(model_bundle, model_path)
    if deployment_path is not None and deployment_path != model_path:
        save_model(model_bundle, deployment_path)

    mlflow_info = log_training_run(
        model=model,
        train_rows=int(X_train.shape[0]),
        validation_rows=int(X_valid.shape[0]),
        num_classes=int(y_valid.nunique()),
        metrics=metrics,
        model_path=model_path,
        run_metadata=run_metadata,
    )

    result = {
        **metrics,
        "model_path": str(model_path),
        "deployed": deployment_path is not None or model_path == MODEL_PATH,
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_valid.shape[0]),
        "num_classes": int(y_valid.nunique()),
        **mlflow_info,
    }
    if run_metadata:
        result.update(run_metadata)
    return result


def train_and_save_model() -> dict:
    """Train with the original full-dataset workflow."""
    X_train, X_valid, y_train, y_valid = load_split()
    return _train_and_save(
        X_train,
        X_valid,
        y_train,
        y_valid,
        model_path=MODEL_PATH,
    )


def train_and_save_simulation_model(step: int = 0, deploy: bool = True) -> dict:
    """Train one cumulative data-growth simulation step."""
    X_train, X_valid, y_train, y_valid, simulation_metadata = (
        load_simulation_split(step)
    )
    history_path = MODEL_HISTORY_DIR / f"model_step_{step:02d}.joblib"
    return _train_and_save(
        X_train,
        X_valid,
        y_train,
        y_valid,
        model_path=history_path,
        deployment_path=MODEL_PATH if deploy else None,
        run_metadata=simulation_metadata,
    )
