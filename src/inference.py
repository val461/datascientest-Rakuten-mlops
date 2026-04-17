from pathlib import Path

import joblib
import pandas as pd

from src.preprocessor import transform_features
import logging
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")
model_bundle = None


class ModelNotAvailableError(FileNotFoundError):
    pass


def is_model_available() -> bool:
    return MODEL_PATH.exists()


def load_model(force_reload: bool = False, require_exists: bool = True):
    global model_bundle

    if force_reload:
        model_bundle = None

    if model_bundle is None:
        if MODEL_PATH.exists():
            logger.info(f"✅ Chargement modèle : {MODEL_PATH}")
            model_bundle = joblib.load(MODEL_PATH)
        elif require_exists:
            raise ModelNotAvailableError(
                "Aucun modèle entraîné n'est disponible. Utilisez l'endpoint /train d'abord."
            )
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
