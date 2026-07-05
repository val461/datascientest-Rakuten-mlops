import argparse
import json

from src.simulation.data_growth import MAX_SIMULATION_STEP
from src.trainer import train_and_save_simulation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one cumulative data-growth simulation step."
    )
    parser.add_argument(
        "step",
        type=int,
        choices=range(MAX_SIMULATION_STEP + 1),
        help="Simulation step: 0=50%% of the stream, 10=100%%.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy this step as models/model.joblib.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_save_simulation_model(
        step=args.step,
        deploy=args.deploy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
