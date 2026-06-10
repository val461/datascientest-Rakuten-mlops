import pandas as pd

from src.data_loader import load_training_csv
from src.simulation.split_manager import load_or_create_simulation_split

INITIAL_AVAILABLE_RATIO = 0.50
RATIO_INCREMENT = 0.05
MAX_SIMULATION_STEP = 10


def simulation_ratio(step: int) -> float:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("Le step de simulation doit etre un entier")
    if not 0 <= step <= MAX_SIMULATION_STEP:
        raise ValueError(
            f"Le step de simulation doit etre compris entre 0 et "
            f"{MAX_SIMULATION_STEP}"
        )
    if step == MAX_SIMULATION_STEP:
        return 1.0
    return round(INITIAL_AVAILABLE_RATIO + step * RATIO_INCREMENT, 2)


def load_simulation_split(
    step: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    available_ratio = simulation_ratio(step)
    X, y = load_training_csv()
    validation_indices, stream_indices, split_metadata = (
        load_or_create_simulation_split(X, y)
    )

    if step == MAX_SIMULATION_STEP:
        available_count = len(stream_indices)
    else:
        available_count = int(len(stream_indices) * available_ratio)

    available_indices = stream_indices[:available_count]
    future_indices = stream_indices[available_count:]
    X_available = X.loc[available_indices]
    y_available = y.loc[available_indices]
    X_validation = X.loc[validation_indices]
    y_validation = y.loc[validation_indices]

    simulation_metadata = {
        **split_metadata,
        "simulation_step": step,
        "available_ratio": available_ratio,
        "available_rows": len(available_indices),
        "future_rows": len(future_indices),
    }
    return (
        X_available,
        X_validation,
        y_available,
        y_validation,
        simulation_metadata,
    )
