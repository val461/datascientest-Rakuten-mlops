import json
import os
import re
from functools import lru_cache
from html import unescape
from pathlib import Path
from typing import Iterable, Set

import joblib
import nltk
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from scipy.sparse import save_npz
from spacy.cli import download as spacy_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from unidecode import unidecode

RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASS_WEIGHT = "balanced"
APPLY_DF_FILTER = True
WORD_MIN_DF = 5
WORD_MAX_DF = 0.8
CHAR_MIN_DF = 5
CHAR_MAX_DF = 0.9
MIN_ITEMS_FOR_MULTIPROCESSING = 1_000
MAX_SPACY_PROCESSES = 4

PREPROCESSED_DIR = Path("data/preprocessed")
VECTORIZER_PATH = PREPROCESSED_DIR / "vectorizer.joblib"
TRAIN_VECTORS_PATH = PREPROCESSED_DIR / "X_train_vectors.npz"
VALID_VECTORS_PATH = PREPROCESSED_DIR / "X_valid_vectors.npz"
Y_TRAIN_PATH = PREPROCESSED_DIR / "y_train.csv"
Y_VALID_PATH = PREPROCESSED_DIR / "y_valid.csv"
LABEL_NAMES_PATH = PREPROCESSED_DIR / "label_names.json"
METADATA_PATH = PREPROCESSED_DIR / "metadata.json"

TEXT_COLUMN = "text"
CLEAN_TEXT_COLUMN = "clean_text"
LEMMA_TEXT_COLUMN = "lemma_text"
HTML_SPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def normalize_whitespace(text: str) -> str:
    return HTML_SPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    if text is None:
        text = ""
    text = unescape(str(text))
    text = strip_html(text)
    text = text.lower()
    text = unidecode(text)
    return normalize_whitespace(text)


def combine_text_columns(X: pd.DataFrame) -> pd.DataFrame:
    missing_columns = {"designation", "description"} - set(X.columns)
    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {sorted(missing_columns)}")

    combined = (
        X["designation"].fillna("").astype(str)
        + " "
        + X["description"].fillna("").astype(str)
    ).map(normalize_whitespace)

    return pd.DataFrame({TEXT_COLUMN: combined}, index=X.index)


def select_text_column(df: pd.DataFrame) -> pd.Series:
    return df[TEXT_COLUMN]


def select_clean_text_column(df: pd.DataFrame) -> pd.Series:
    return df[CLEAN_TEXT_COLUMN]


def select_lemma_text_column(df: pd.DataFrame) -> pd.Series:
    return df[LEMMA_TEXT_COLUMN]


def ensure_nltk_stopwords() -> None:
    nltk.download("stopwords", quiet=True)


def normalize_stopword_list(words: Iterable[str]) -> Set[str]:
    return {unidecode(word).lower() for word in words}


@lru_cache(maxsize=1)
def get_stopword_set() -> Set[str]:
    ensure_nltk_stopwords()
    french_stopwords = normalize_stopword_list(stopwords.words("french"))
    english_stopwords = normalize_stopword_list(stopwords.words("english"))
    return french_stopwords | english_stopwords


@lru_cache(maxsize=1)
def get_spacy_model() -> spacy.language.Language:
    model_name = "fr_core_news_sm"
    disabled_components = ["ner", "parser", "textcat"]
    try:
        return spacy.load(model_name, disable=disabled_components)
    except OSError:
        spacy_download(model_name)
        return spacy.load(model_name, disable=disabled_components)


def lemmatize_texts(texts: Iterable[str]) -> list[str]:
    texts = list(texts)
    if not texts:
        return []

    stopword_set = get_stopword_set()
    nlp = get_spacy_model()
    if len(texts) < MIN_ITEMS_FOR_MULTIPROCESSING:
        n_process = 1
    else:
        n_process = max(1, min(MAX_SPACY_PROCESSES, os.cpu_count() or 1))
    lemmatized_texts: list[str] = []
    for doc in nlp.pipe(texts, batch_size=256, n_process=n_process):
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and len(token) > 2 and token.lemma_.lower() not in stopword_set
        ]
        lemmatized_texts.append(" ".join(lemmas))
    return lemmatized_texts


def prepare_text_features(df: pd.DataFrame) -> pd.DataFrame:
    text_series = select_text_column(df)
    clean_series = text_series.map(clean_text)
    lemma_series = pd.Series(lemmatize_texts(clean_series.tolist()), index=df.index)

    return pd.DataFrame(
        {
            CLEAN_TEXT_COLUMN: clean_series,
            LEMMA_TEXT_COLUMN: lemma_series,
        },
        index=df.index,
    )


def build_text_preprocessor() -> Pipeline:
    word_vectorizer = TfidfVectorizer(
        preprocessor=None,
        tokenizer=str.split,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=WORD_MIN_DF if APPLY_DF_FILTER else 1,
        max_df=WORD_MAX_DF if APPLY_DF_FILTER else 1.0,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        preprocessor=None,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=CHAR_MIN_DF if APPLY_DF_FILTER else 1,
        max_df=CHAR_MAX_DF if APPLY_DF_FILTER else 1.0,
        sublinear_tf=True,
    )

    return Pipeline(
        steps=[
            ("prepare_text", FunctionTransformer(prepare_text_features, validate=False)),
            (
                "vectorize",
                FeatureUnion(
                    [
                        (
                            "word",
                            Pipeline(
                                steps=[
                                    (
                                        "select_lemma_text",
                                        FunctionTransformer(select_lemma_text_column, validate=False),
                                    ),
                                    ("tfidf", word_vectorizer),
                                ]
                            ),
                        ),
                        (
                            "char",
                            Pipeline(
                                steps=[
                                    (
                                        "select_clean_text",
                                        FunctionTransformer(select_clean_text_column, validate=False),
                                    ),
                                    ("tfidf", char_vectorizer),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
def fit_transform_features(X_train: pd.DataFrame):
    features = combine_text_columns(X_train)
    preprocessor = build_text_preprocessor()
    vectors = preprocessor.fit_transform(features)
    return preprocessor, vectors


def transform_features(preprocessor: Pipeline, X: pd.DataFrame):
    features = combine_text_columns(X)
    return preprocessor.transform(features)


def get_preprocessing_metadata() -> dict:
    return {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "class_weight": CLASS_WEIGHT,
        "word_min_df": WORD_MIN_DF if APPLY_DF_FILTER else 1,
        "word_max_df": WORD_MAX_DF if APPLY_DF_FILTER else 1.0,
        "char_min_df": CHAR_MIN_DF if APPLY_DF_FILTER else 1,
        "char_max_df": CHAR_MAX_DF if APPLY_DF_FILTER else 1.0,
    }


def save_preprocessing_artifacts(
    preprocessor: Pipeline,
    X_train_vectors,
    X_valid_vectors,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> None:
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, VECTORIZER_PATH)
    save_npz(TRAIN_VECTORS_PATH, X_train_vectors)
    save_npz(VALID_VECTORS_PATH, X_valid_vectors)
    y_train.rename("prdtypecode").to_csv(Y_TRAIN_PATH, index=True)
    y_valid.rename("prdtypecode").to_csv(Y_VALID_PATH, index=True)
    label_names = sorted(pd.concat([y_train, y_valid]).unique().tolist())
    LABEL_NAMES_PATH.write_text(
        json.dumps(label_names, ensure_ascii=False, indent=2)
    )
    METADATA_PATH.write_text(
        json.dumps(get_preprocessing_metadata(), ensure_ascii=False, indent=2)
    )
