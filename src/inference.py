import joblib
import pandas as pd
from pathlib import Path
from src.trainer import train_and_save_model

MODEL_PATH = Path("models/model.joblib")
model = None


def load_model():
    global model
    if model is None:
        if MODEL_PATH.exists():
            print(f"✅ Chargement modèle : {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
        else:
            print("⚠️ Modèle absent : entraînement automatique")
            train_and_save_model()
            model = joblib.load(MODEL_PATH)
    return model


def predict(features: dict) -> int:
    model = load_model()
    # Rename features as needed by model
    data = pd.DataFrame([{
        "sepal length (cm)": features["sepal_length"],
        "sepal width (cm)": features["sepal_width"],
        "petal length (cm)": features["petal_length"],
        "petal width (cm)": features["petal_width"]
    }])
    return int(model.predict(data)[0])
