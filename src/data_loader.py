import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessor import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
INDEX_COLUMN = "Unnamed: 0"
X_TRAIN_PATH = DATA_DIR / "X_train.csv"
Y_TRAIN_PATH = DATA_DIR / "Y_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path, index_col=INDEX_COLUMN)


def load_training_csv() -> tuple[pd.DataFrame, pd.Series]:
    X_train = _read_csv(X_TRAIN_PATH)
    y_train_df = _read_csv(Y_TRAIN_PATH)

    if not X_train.index.is_unique or not y_train_df.index.is_unique:
        raise ValueError("Les index des fichiers d'entrainement doivent etre uniques")
    if "prdtypecode" not in y_train_df.columns:
        raise ValueError("La colonne `prdtypecode` est absente de Y_train.csv")

    y_train = y_train_df["prdtypecode"].astype("int64")
    if not X_train.index.equals(y_train.index):
        logger.warning("Reindexation de y_train sur les index de X_train")
        y_train = y_train.reindex(X_train.index)
        if y_train.isna().any():
            raise ValueError(
                "Les index de X_train.csv et Y_train.csv ne sont pas alignes"
            )

    if os.getenv("SMALL_DATASET_FOR_DEBUGGING","false")=="true":
        X_train = X_train.iloc[:1000]
        y_train = y_train.iloc[:1000]

    logger.info("Chargement du jeu d'entrainement : %s lignes", X_train.shape[0])
    return X_train, y_train


def load_split():
    """Return the original random split used by full-dataset training."""
    X, y = load_training_csv()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_valid, y_train, y_valid


def load_test_csv() -> pd.DataFrame:
    X_test = _read_csv(X_TEST_PATH)
    logger.info("Chargement du jeu de test : %s lignes", X_test.shape[0])
    return X_test
