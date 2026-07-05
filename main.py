import copy
import logging
import os
from datetime import datetime, timezone
from threading import Lock

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Query,
    Security,
    status,
)
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from src.inference import (
    ModelNotAvailableError,
    get_model_metadata,
    is_model_available,
    load_model,
    predict,
)
from src.simulation.data_growth import MAX_SIMULATION_STEP
from src.simulation.split_manager import load_split_metadata
from src.trainer import train_and_save_model, train_and_save_simulation_model

logging.basicConfig(
    format=(
        "%(asctime)s %(levelname)s %(filename)s %(funcName)s - %(message)s"
    ),
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rakuten Prediction API",
    description=(
        "Product classification API with full training and reproducible "
        "data-growth simulation."
    ),
)
Instrumentator().instrument(app).expose(app)

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")
training_lock = Lock()
training_ongoing = False
campaign_state = {
    "status": "idle",
    "current_step": None,
    "completed_steps": 0,
    "total_steps": MAX_SIMULATION_STEP + 1,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "results": [],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Cle API invalide")


def claim_training() -> None:
    global training_ongoing
    with training_lock:
        if training_ongoing:
            raise HTTPException(
                status_code=409,
                detail="Un entrainement est deja en cours",
            )
        training_ongoing = True


def release_training() -> None:
    global training_ongoing
    with training_lock:
        training_ongoing = False


def campaign_snapshot() -> dict:
    with training_lock:
        return copy.deepcopy(campaign_state)


def reset_campaign_state() -> None:
    campaign_state.update(
        {
            "status": "idle",
            "current_step": None,
            "completed_steps": 0,
            "total_steps": MAX_SIMULATION_STEP + 1,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "results": [],
        }
    )


def summarize_training_result(result: dict) -> dict:
    return {
        "step": result["simulation_step"],
        "available_ratio": result["available_ratio"],
        "training_rows": result["train_rows"],
        "accuracy": result["accuracy"],
        "f1_macro": result["f1_macro"],
        "f1_weighted": result["f1_weighted"],
        "mlflow_run_id": result["mlflow_run_id"],
        "deployed": result["deployed"],
    }


def run_simulation_campaign() -> None:
    try:
        for step in range(MAX_SIMULATION_STEP + 1):
            with training_lock:
                campaign_state["current_step"] = step

            result = train_and_save_simulation_model(
                step=step,
                deploy=step == MAX_SIMULATION_STEP,
            )

            with training_lock:
                campaign_state["completed_steps"] = step + 1
                campaign_state["results"].append(
                    summarize_training_result(result)
                )

        load_model(force_reload=True)
        with training_lock:
            campaign_state["status"] = "completed"
            campaign_state["finished_at"] = utc_now()
    except Exception as exc:
        logger.exception("Simulation campaign failed")
        with training_lock:
            campaign_state["status"] = "failed"
            campaign_state["error"] = str(exc)
            campaign_state["finished_at"] = utc_now()
    finally:
        release_training()


class ProductFeatures(BaseModel):
    designation: str
    description: str | None = None
    productid: int | None = None
    imageid: int | None = None


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up")
    load_model(require_exists=False)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Rakuten Prediction API",
        "documentation": "/docs",
        "health": "/health",
        "mlflow": "http://localhost:5001",
    }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_endpoint(features: ProductFeatures):
    try:
        prediction = predict(features.model_dump())
        return {"prediction": prediction}
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/train",
    dependencies=[Depends(verify_api_key)],
    summary="Train the classic full-data model",
)
def train_endpoint():
    claim_training()
    try:
        result = train_and_save_model()
        load_model(force_reload=True)
        return {
            "status": "success",
            "message": "Modele complet entraine et deploye",
            **result,
        }
    except Exception as exc:
        logger.exception("Full training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        release_training()


@app.post(
    "/train/simulation",
    dependencies=[Depends(verify_api_key)],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Launch the complete 0-to-10 simulation campaign",
)
def start_simulation_campaign(background_tasks: BackgroundTasks):
    claim_training()
    with training_lock:
        reset_campaign_state()
        campaign_state.update(
            {
                "status": "running",
                "started_at": utc_now(),
            }
        )

    background_tasks.add_task(run_simulation_campaign)
    return {
        "status": "accepted",
        "message": "Campagne de simulation lancee",
        "status_url": "/train/simulation/status",
        "total_steps": MAX_SIMULATION_STEP + 1,
    }


@app.get(
    "/train/simulation/status",
    dependencies=[Depends(verify_api_key)],
    summary="Get the simulation campaign progress",
)
def simulation_campaign_status():
    return campaign_snapshot()


@app.post(
    "/train/simulation/{step}",
    dependencies=[Depends(verify_api_key)],
    summary="Train one simulation step",
)
def train_simulation_step(
    step: int = Path(
        ge=0,
        le=MAX_SIMULATION_STEP,
        description="0=50% of the stream, 10=100% of the stream",
    ),
    deploy: bool = Query(
        default=False,
        description="Deploy this step as the model served by /predict",
    ),
):
    claim_training()
    try:
        result = train_and_save_simulation_model(step=step, deploy=deploy)
        if deploy:
            load_model(force_reload=True)
        return {
            "status": "success",
            "message": (
                "Etape entrainee et deployee"
                if deploy
                else "Etape entrainee sans deploiement"
            ),
            **result,
        }
    except Exception as exc:
        logger.exception("Simulation step %s failed", step)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        release_training()


@app.get("/data-status", dependencies=[Depends(verify_api_key)])
def data_status():
    split_metadata = load_split_metadata()
    model_metadata = get_model_metadata()

    if split_metadata is None:
        return {
            "split_initialized": False,
            "model_available": is_model_available(),
        }

    response = {
        "split_initialized": True,
        "model_available": is_model_available(),
        **split_metadata,
    }
    if model_metadata and "simulation_step" in model_metadata:
        response.update(
            {
                "current_model_step": model_metadata["simulation_step"],
                "current_available_ratio": model_metadata["available_ratio"],
                "current_training_rows": model_metadata["available_rows"],
                "current_future_rows": model_metadata["future_rows"],
            }
        )
    return response


@app.get("/health")
def health():
    model_metadata = get_model_metadata()
    return {
        "status": "healthy",
        "service": "inference-api",
        "model_available": is_model_available(),
        "training_ongoing": training_ongoing,
        "campaign_status": campaign_snapshot()["status"],
        "model_step": (
            model_metadata.get("simulation_step") if model_metadata else None
        ),
    }
