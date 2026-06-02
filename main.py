import logging
import os
from contextlib import asynccontextmanager
from threading import Lock

import joblib
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from src.inference import (
    ModelNotAvailableError,
    is_model_available,
    load_model,
    predict,
)
from src.split_config import SPLITS
from src.trainer import MODEL_PATH, prepare_training_data, train_and_save_model

logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "dev").lower()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    if APP_ENV in {"dev", "local", "test"}:
        API_KEY = "secret"
    else:
        raise RuntimeError("API_KEY doit être définie hors environnement de développement")

api_key_header = APIKeyHeader(name="X-API-Key")
TRAINING_LOCK = Lock()


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide")
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(require_exists=False)
    yield


app = FastAPI(
    title="Rakuten Prediction API",
    lifespan=lifespan,
)


class ProductFeatures(BaseModel):
    designation: str
    description: str | None = None
    productid: int | None = None
    imageid: int | None = None


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_endpoint(features: ProductFeatures):
    try:
        prediction = predict(features.model_dump())
        return {"prediction": prediction}

    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Erreur interne pendant la prédiction")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant la prédiction",
        ) from exc


@app.post("/train", dependencies=[Depends(verify_api_key)])
def train_endpoint():
    if not TRAINING_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Un entraînement est déjà en cours",
        )

    try:
        training_data = prepare_training_data()
        results = []

        for split_id, train_ratio in enumerate(SPLITS):
            result = train_and_save_model(
                train_ratio=train_ratio,
                split_id=split_id,
                dataset_bundle=training_data,
            )
            results.append(result)

        best_run = max(results, key=lambda r: r["f1_macro"])

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_run["model_bundle"], MODEL_PATH)

        load_model(force_reload=True)

        public_runs = []
        for run in results:
            public_runs.append(
                {key: value for key, value in run.items() if key != "model_bundle"}
            )

        return {
            "status": "success",
            "message": "Tous les splits ont été entraînés",
            "best_split": best_run["train_ratio"],
            "best_split_id": best_run["split_id"],
            "best_f1_macro": best_run["f1_macro"],
            "model_path": str(MODEL_PATH),
            "runs": public_runs,
        }

    except ValueError as exc:
        logger.warning("Erreur de validation pendant l'entraînement: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Erreur interne pendant l'entraînement")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant l'entraînement",
        ) from exc

    finally:
        TRAINING_LOCK.release()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": is_model_available(),
        "training_in_progress": TRAINING_LOCK.locked(),
    }
