import json

from src.trainer import train_and_save_model


def main() -> None:
    result = train_and_save_model()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
