import json

from src.simulation.data_growth import MAX_SIMULATION_STEP
from src.trainer import train_and_save_simulation_model


def summarize(result: dict) -> dict:
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


def main() -> None:
    results = []
    for step in range(MAX_SIMULATION_STEP + 1):
        print(f"Starting simulation step {step}/{MAX_SIMULATION_STEP}")
        result = train_and_save_simulation_model(
            step=step,
            deploy=step == MAX_SIMULATION_STEP,
        )
        results.append(summarize(result))
        print(json.dumps(results[-1], ensure_ascii=False, indent=2))

    print(json.dumps({"status": "completed", "results": results}, indent=2))


if __name__ == "__main__":
    main()
