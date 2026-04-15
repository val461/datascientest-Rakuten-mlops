from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from src.inference import predict, load_model
from src.trainer import train_and_save_model

app = FastAPI(title="Prediction API")


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post("/predict")
def predict_endpoint(features: IrisFeatures):
    try:
        pred = predict(features.model_dump())
        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "prediction": pred,
            "class_name": class_names[pred]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_endpoint(background_tasks: BackgroundTasks):
    """Endpoint pour réentraîner le modèle"""
    try:
        result = train_and_save_model()
        # Recharger le modèle après entraînement
        load_model()
        return {
            "status": "success",
            "message": "Modèle réentraîné avec succès",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}
