import os
from fastapi import FastAPI, HTTPException, Depends, Security
from pydantic import BaseModel

from src.inference import ModelNotAvailableError, is_model_available, load_model, predict
from src.trainer import train_and_save_model

from fastapi.security.api_key import APIKeyHeader


API_KEY = os.getenv("API_KEY", "secret")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide")

app = FastAPI(title="Rakuten Prediction API")


class ProductFeatures(BaseModel):
    designation: str
    description: str | None = None
    productid: int | None = None
    imageid: int | None = None


@app.on_event("startup")
async def startup_event():
    load_model(require_exists=False)


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_endpoint(features: ProductFeatures):
    try:
        prediction = predict(features.model_dump())
        return {"prediction": prediction}
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train", dependencies=[Depends(verify_api_key)])
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
    return {"status": "healthy", "model_loaded": is_model_available()}
