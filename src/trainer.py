from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from src.data_loader import load_data
from src.preprocessor import preprocess_data

MODEL_PATH = Path("models/model.joblib")


def create_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model


def train(model, X, y):
    print("🚀 Entraînement...")
    model.fit(X, y)


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Modèle sauvegardé")


def evaluate(model, X, y):
    # TODO: return weighted average F1-score (and maybe other metrics)
    accuracy = model.score(X, y)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def train_and_save_model() -> dict:
    """
    Entraîne et sauvegarde le modèle.
    Retourne des valeurs pour l'API : score et chemin du modèle.
    """
    X, y = load_data()
    X = preprocess_data(X)

    model = create_model()
    train(model, X, y)
    save_model(model, MODEL_PATH)

    score = evaluate(model, X, y)

    # Return
    return {"accuracy": float(score), "model_path": str(MODEL_PATH)}
