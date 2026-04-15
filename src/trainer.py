from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from src.data_loader import load_data
from src.preprocessor import preprocess_data

MODEL_PATH = Path("models/iris_model.joblib")


def train_and_save_model() -> dict:
    """Entraîne et sauvegarde le modèle"""
    X, y = load_data()
    X = preprocess_data(X)

    print("🚀 Entraînement RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    accuracy = model.score(X, y)
    print(f"✅ Modèle sauvegardé → Accuracy: {accuracy:.4f}")

    return {"accuracy": float(accuracy), "model_path": str(MODEL_PATH)}
