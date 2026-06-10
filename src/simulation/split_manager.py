import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import data_loader
from src.preprocessor import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)

SPLIT_DIR = Path("data/splits")
VALIDATION_INDICES_PATH = SPLIT_DIR / "validation_indices.json"
STREAM_INDICES_PATH = SPLIT_DIR / "stream_indices.json"
SPLIT_METADATA_PATH = SPLIT_DIR / "split_metadata.json"

SPLIT_VERSION = 1
VALIDATION_RATIO = TEST_SIZE
HASH_CHUNK_SIZE = 1024 * 1024


def calculate_dataset_fingerprint(paths: tuple[Path, ...] | None = None) -> str:
    source_paths = paths or (
        data_loader.X_TRAIN_PATH,
        data_loader.Y_TRAIN_PATH,
    )
    digest = hashlib.sha256()

    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as source:
            while chunk := source.read(HASH_CHUNK_SIZE):
                digest.update(chunk)

    return f"sha256:{digest.hexdigest()}"


def _write_json_atomic(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fichier JSON invalide : {path}") from exc


def _validate_partition(
    all_indices: pd.Index,
    validation_indices: list,
    stream_indices: list,
) -> None:
    all_index_set = set(all_indices.tolist())
    validation_set = set(validation_indices)
    stream_set = set(stream_indices)

    if len(validation_set) != len(validation_indices):
        raise ValueError("Le split de validation contient des index dupliques")
    if len(stream_set) != len(stream_indices):
        raise ValueError("Le flux contient des index dupliques")
    if validation_set & stream_set:
        raise ValueError("Le split de validation et le flux se chevauchent")
    if validation_set | stream_set != all_index_set:
        raise ValueError("Les index sauvegardes ne couvrent pas le dataset courant")


def create_simulation_split(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_fingerprint: str,
) -> tuple[list, list, dict]:
    stream_indices, validation_indices = train_test_split(
        X.index.tolist(),
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_STATE,
        stratify=y.loc[X.index],
    )

    # The persisted order represents the simulated order of data arrival.
    stream_indices = (
        pd.Series(stream_indices)
        .sample(frac=1, random_state=RANDOM_STATE)
        .tolist()
    )
    validation_indices = list(validation_indices)
    _validate_partition(X.index, validation_indices, stream_indices)

    metadata = {
        "split_version": SPLIT_VERSION,
        "random_state": RANDOM_STATE,
        "validation_ratio": VALIDATION_RATIO,
        "total_rows": int(len(X)),
        "validation_rows": int(len(validation_indices)),
        "stream_rows": int(len(stream_indices)),
        "dataset_fingerprint": dataset_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _write_json_atomic(VALIDATION_INDICES_PATH, validation_indices)
    _write_json_atomic(STREAM_INDICES_PATH, stream_indices)
    _write_json_atomic(SPLIT_METADATA_PATH, metadata)
    logger.info(
        "Split cree : %s lignes de validation, %s lignes dans le flux",
        len(validation_indices),
        len(stream_indices),
    )
    return validation_indices, stream_indices, metadata


def load_or_create_simulation_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[list, list, dict]:
    split_paths = (
        VALIDATION_INDICES_PATH,
        STREAM_INDICES_PATH,
        SPLIT_METADATA_PATH,
    )
    existing_paths = [path.exists() for path in split_paths]
    dataset_fingerprint = calculate_dataset_fingerprint()

    if not any(existing_paths):
        return create_simulation_split(X, y, dataset_fingerprint)
    if not all(existing_paths):
        raise ValueError(
            "Le split est incomplet. Supprimez volontairement les fichiers "
            "restants de data/splits avant de le recreer."
        )

    validation_indices = _read_json(VALIDATION_INDICES_PATH)
    stream_indices = _read_json(STREAM_INDICES_PATH)
    metadata = _read_json(SPLIT_METADATA_PATH)
    if not isinstance(validation_indices, list) or not isinstance(stream_indices, list):
        raise ValueError("Les fichiers d'index du split doivent contenir des listes")
    if not isinstance(metadata, dict):
        raise ValueError("Les metadonnees du split doivent contenir un objet JSON")
    if metadata.get("split_version") != SPLIT_VERSION:
        raise ValueError("La version du split sauvegarde n'est pas supportee")
    if metadata.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError(
            "Le dataset brut a change depuis la creation du split. "
            "Un nouveau split doit etre cree volontairement."
        )

    expected_counts = {
        "total_rows": len(X),
        "validation_rows": len(validation_indices),
        "stream_rows": len(stream_indices),
    }
    for key, expected_value in expected_counts.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"Metadonnee de split incoherente : {key}")

    _validate_partition(X.index, validation_indices, stream_indices)
    return validation_indices, stream_indices, metadata


def load_split_metadata() -> dict | None:
    if not SPLIT_METADATA_PATH.exists():
        return None
    metadata = _read_json(SPLIT_METADATA_PATH)
    if not isinstance(metadata, dict):
        raise ValueError("Les metadonnees du split doivent contenir un objet JSON")
    return metadata
