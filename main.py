from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import ModelNotAvailableError, is_model_available, load_model, predict
from src.trainer import train_and_save_model

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s %(funcName)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

app = FastAPI(title="Rakuten Prediction API")


class ProductFeatures(BaseModel):
    designation: str
    description: str | None = None
    productid: int | None = None
    imageid: int | None = None


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up.")
    load_model(require_exists=False)


@app.post("/predict")
def predict_endpoint(features: ProductFeatures):
    try:
        prediction = predict(features.model_dump())
        return {"prediction": prediction}
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train")
def train_endpoint():
    try:
        result = train_and_save_model()
        load_model(force_reload=True)
        return {
            "status": "success",
            "message": "Modele reentraine avec succes",
            **result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "healthy", "service": "inference-api", "model_available": is_model_available()}
