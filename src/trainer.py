from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.data_loader import load_training_data
from src.preprocessor import (
    CLASS_WEIGHT,
    RANDOM_STATE,
    TEST_SIZE,
    fit_transform_features,
    get_preprocessing_metadata,
    save_preprocessing_artifacts,
    transform_features,
)

MODEL_PATH = Path("models/model.joblib")


def create_model() -> LinearSVC:
    return LinearSVC(
        C=1.0,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )


def train(model: LinearSVC, X_train_vectors, y_train) -> None:
    print("🚀 Entraînement du classifieur texte...")
    model.fit(X_train_vectors, y_train)


def save_model(model_bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path)
    print(f"✅ Modèle sauvegardé : {path}")


def evaluate(model: LinearSVC, X_valid_vectors, y_valid) -> dict:
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


def train_and_save_model() -> dict:
    X, y = load_training_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor, X_train_vectors = fit_transform_features(X_train)
    X_valid_vectors = transform_features(preprocessor, X_valid)

    save_preprocessing_artifacts(preprocessor, X_train_vectors, X_valid_vectors, y_train, y_valid)

    model = create_model()
    train(model, X_train_vectors, y_train)

    model_bundle = {
        "classifier": model,
        "preprocessor": preprocessor,
        "metadata": get_preprocessing_metadata(),
    }
    save_model(model_bundle, MODEL_PATH)

    metrics = evaluate(model, X_valid_vectors, y_valid)
    return {
        **metrics,
        "model_path": str(MODEL_PATH),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_valid.shape[0]),
        "num_classes": int(y.nunique()),
    }
