from pathlib import Path

import pandas as pd
import logging
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


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X_train = _read_csv(X_TRAIN_PATH)
    y_train_df = _read_csv(Y_TRAIN_PATH)

    if "prdtypecode" not in y_train_df.columns:
        raise ValueError("La colonne `prdtypecode` est absente de Y_train.csv")

    y_train = y_train_df["prdtypecode"].astype("int64")
    if not X_train.index.equals(y_train.index):
        logger.warning("⚠️ reindexing y_train.")
        y_train = y_train.reindex(X_train.index)
        if y_train.isna().any():
            raise ValueError("Les index de X_train.csv et Y_train.csv ne sont pas alignés")

    logger.info(f"✅ Chargement du jeu d'entraînement : {X_train.shape[0]} lignes")
    return X_train, y_train


def load_test_data() -> pd.DataFrame:
    X_test = _read_csv(X_TEST_PATH)
    logger.info(f"✅ Chargement du jeu de test : {X_test.shape[0]} lignes")
    return X_test
