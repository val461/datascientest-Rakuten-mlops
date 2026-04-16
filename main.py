from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import load_model, predict
from src.trainer import train_and_save_model

app = FastAPI(title="Rakuten Prediction API")


class ProductFeatures(BaseModel):
    designation: str
    description: str | None = None
    productid: int | None = None
    imageid: int | None = None


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post("/predict")
def predict_endpoint(features: ProductFeatures):
    try:
        prediction = predict(features.model_dump())
        return {"prediction": prediction}
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
    return {"status": "healthy", "model_loaded": load_model() is not None}
