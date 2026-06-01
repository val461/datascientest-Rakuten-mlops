from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.data_loader import load_training_data
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

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.joblib"


def create_model() -> LinearSVC:
    return LinearSVC(
        C=1.0,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )


def train(model: LinearSVC, X_train_vectors, y_train) -> None:
    print("Entraînement du classifieur texte...")
    model.fit(X_train_vectors, y_train)


def evaluate(model: LinearSVC, X_valid_vectors, y_valid) -> dict[str, object]:
    predictions = model.predict(X_valid_vectors)
    accuracy = accuracy_score(y_valid, predictions)
    f1_macro = f1_score(y_valid, predictions, average="macro")
    f1_weighted = f1_score(y_valid, predictions, average="weighted")
    report = classification_report(y_valid, predictions, output_dict=False)

    print(f"Accuracy validation : {accuracy:.4f}")
    print(f"F1 macro validation : {f1_macro:.4f}")
    print(f"F1 pondéré validation : {f1_weighted:.4f}")

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
    }


def prepare_training_data() -> dict[str, Any]:
    """
    Prépare un split fixe train/validation une seule fois
    pour comparer équitablement plusieurs ratios d'entraînement.
    """
    X, y = load_training_data()

    X_train_full, X_valid, y_train_full, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return {
        "X_train_full": X_train_full,
        "X_valid": X_valid,
        "y_train_full": y_train_full,
        "y_valid": y_valid,
        "num_classes": int(y.nunique()),
        "total_rows": int(len(X)),
    }


def build_train_subset(X_train_full, y_train_full, train_ratio: float):
    """
    Crée un sous-ensemble stratifié aléatoire à partir du train fixe.
    """
    if not 0 < train_ratio <= 1:
        raise ValueError("train_ratio doit être strictement supérieur à 0 et inférieur ou égal à 1")

    if train_ratio == 1.0:
        return X_train_full.copy(), y_train_full.copy()

    X_train_subset, _, y_train_subset, _ = train_test_split(
        X_train_full,
        y_train_full,
        train_size=train_ratio,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    return X_train_subset, y_train_subset


def get_training_params(
    model: LinearSVC,
    train_pool_rows: int,
    train_rows: int,
    validation_rows: int,
    num_classes: int,
    split_id: int,
    train_ratio: float,
) -> dict[str, object]:
    return {
        "model_name": model.__class__.__name__,
        "model_c": model.C,
        "model_class_weight": model.class_weight,
        "model_random_state": model.random_state,
        "test_size": TEST_SIZE,
        "train_pool_rows": train_pool_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "num_classes": num_classes,
        "split_id": split_id,
        "train_ratio": train_ratio,
        "validation_strategy": "fixed_holdout",
        **get_preprocessing_metadata(),
    }


def log_training_run(
    model: LinearSVC,
    model_bundle: dict[str, Any],
    train_pool_rows: int,
    train_rows: int,
    validation_rows: int,
    num_classes: int,
    metrics: dict[str, object],
    split_id: int,
    train_ratio: float,
) -> dict[str, str]:
    run_name = f"{model.__class__.__name__}-training-split-{split_id}"

    with start_training_run(run_name=run_name) as (run, mlflow_context):
        log_tags(
            {
                "project": "datascientest-rakuten-mlops",
                "task": "product-category-classification",
                "split_id": str(split_id),
            }
        )

        log_params(
            get_training_params(
                model=model,
                train_pool_rows=train_pool_rows,
                train_rows=train_rows,
                validation_rows=validation_rows,
                num_classes=num_classes,
                split_id=split_id,
                train_ratio=train_ratio,
            )
        )

        log_metrics(
            {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_weighted": float(metrics["f1_weighted"]),
            }
        )

        log_text(
            str(metrics["classification_report"]),
            f"reports/classification_report_split_{split_id}.txt",
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_model_path = Path(tmp_dir) / f"model_split_{split_id}.joblib"
            joblib.dump(model_bundle, tmp_model_path)
            log_artifact_if_exists(tmp_model_path, artifact_path="model")

        log_artifacts_if_exists(
            PREPROCESSED_DIR,
            artifact_path=f"preprocessing_split_{split_id}",
        )

        return {
            "mlflow_run_id": run.info.run_id,
            **mlflow_context,
        }


def train_and_save_model(
    train_ratio: float,
    split_id: int,
    dataset_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Entraîne un modèle sur un sous-ensemble stratifié du train fixe,
    puis l'évalue sur la même validation pour tous les ratios.
    """
    data = dataset_bundle or prepare_training_data()

    X_train_full = data["X_train_full"]
    X_valid = data["X_valid"]
    y_train_full = data["y_train_full"]
    y_valid = data["y_valid"]
    num_classes = int(data["num_classes"])

    X_train, y_train = build_train_subset(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        train_ratio=train_ratio,
    )

    preprocessor, X_train_vectors = fit_transform_features(X_train)
    X_valid_vectors = transform_features(preprocessor, X_valid)

    save_preprocessing_artifacts(
        preprocessor,
        X_train_vectors,
        X_valid_vectors,
        y_train,
        y_valid,
    )

    model = create_model()
    train(model, X_train_vectors, y_train)

    model_bundle = {
        "classifier": model,
        "preprocessor": preprocessor,
        "metadata": {
            **get_preprocessing_metadata(),
            "split_id": split_id,
            "train_ratio": train_ratio,
        },
    }

    metrics = evaluate(model, X_valid_vectors, y_valid)

    mlflow_info = log_training_run(
        model=model,
        model_bundle=model_bundle,
        train_pool_rows=int(X_train_full.shape[0]),
        train_rows=int(X_train.shape[0]),
        validation_rows=int(X_valid.shape[0]),
        num_classes=num_classes,
        metrics=metrics,
        split_id=split_id,
        train_ratio=train_ratio,
    )

    return {
        **metrics,
        "split_id": split_id,
        "train_ratio": train_ratio,
        "model_path": str(MODEL_PATH),
        "train_pool_rows": int(X_train_full.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_valid.shape[0]),
        "num_classes": num_classes,
        **mlflow_info,
        "model_bundle": model_bundle,
    }
