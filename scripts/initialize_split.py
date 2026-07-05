import json

from src.data_loader import load_training_csv
from src.simulation.split_manager import load_or_create_simulation_split


def main() -> None:
    X, y = load_training_csv()
    _, _, metadata = load_or_create_simulation_split(X, y)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
