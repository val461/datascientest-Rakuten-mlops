from pathlib import Path

import joblib
import pandas as pd

from src.preprocessor import transform_features
from src.trainer import train_and_save_model

MODEL_PATH = Path("models/model.joblib")
model_bundle = None


def load_model(force_reload: bool = False):
    global model_bundle

    if force_reload:
        model_bundle = None

    if model_bundle is None:
        if MODEL_PATH.exists():
            print(f"✅ Chargement modèle : {MODEL_PATH}")
            model_bundle = joblib.load(MODEL_PATH)
        else:
            print("⚠️ Modèle absent : entraînement automatique")
            train_and_save_model()
            model_bundle = joblib.load(MODEL_PATH)
    return model_bundle


def predict(features: dict) -> int:
    bundle = load_model()
    data = pd.DataFrame(
        [
            {
                "designation": features["designation"],
                "description": features.get("description"),
                "productid": features.get("productid"),
                "imageid": features.get("imageid"),
            }
        ]
    )
    vectors = transform_features(bundle["preprocessor"], data)
    return int(bundle["classifier"].predict(vectors)[0])
