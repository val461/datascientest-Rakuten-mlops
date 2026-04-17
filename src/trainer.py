from pathlib import Path

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
import logging
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")


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
    joblib.dump(model_bundle, path)
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
) -> dict[str, object]:
    return {
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


def log_training_run(
    model: LinearSVC,
    train_rows: int,
    validation_rows: int,
    num_classes: int,
    metrics: dict[str, object],
) -> dict[str, str]:
    run_name = f"{model.__class__.__name__}-training"

    with start_training_run(run_name=run_name) as (run, mlflow_context):
        log_tags(
            {
                "project": "datascientest-rakuten-mlops",
                "task": "product-category-classification",
            }
        )
        log_params(get_training_params(model, train_rows, validation_rows, num_classes))
        log_metrics(
            {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_weighted": float(metrics["f1_weighted"]),
            }
        )
        log_text(str(metrics["classification_report"]), "reports/classification_report.txt")
        log_artifact_if_exists(MODEL_PATH, artifact_path="model")
        log_artifacts_if_exists(PREPROCESSED_DIR, artifact_path="preprocessing")

        return {
            "mlflow_run_id": run.info.run_id,
            **mlflow_context,
        }


def train_and_save_model() -> dict:
    X, y = load_training_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info('Preprocessor fit-transforming X_train. May take 6mn.')
    preprocessor, X_train_vectors = fit_transform_features(X_train)
    logger.info('Preprocessor transforming X_valid. May take 2mn.')
    X_valid_vectors = transform_features(preprocessor, X_valid)

    save_preprocessing_artifacts(preprocessor, X_train_vectors, X_valid_vectors, y_train, y_valid)

    model = create_model()
    logger.info('Training. May take 1mn.')
    train(model, X_train_vectors, y_train)

    model_bundle = {
        "classifier": model,
        "preprocessor": preprocessor,
        "metadata": get_preprocessing_metadata(),
    }
    save_model(model_bundle, MODEL_PATH)

    logger.info('Evaluating.')
    metrics = evaluate(model, X_valid_vectors, y_valid)
    mlflow_info = log_training_run(
        model=model,
        train_rows=int(X_train.shape[0]),
        validation_rows=int(X_valid.shape[0]),
        num_classes=int(y.nunique()),
        metrics=metrics,
    )

    return {
        **metrics,
        "model_path": str(MODEL_PATH),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_valid.shape[0]),
        "num_classes": int(y.nunique()),
        **mlflow_info,
    }
