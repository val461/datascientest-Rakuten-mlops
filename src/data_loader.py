import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris

DATA_PATH = Path("data/raw/iris.csv")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if DATA_PATH.exists():
        print(f"Chargement du CSV : {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y
    else:
        print("CSV non trouvé : utilisation dataset Iris (test)")
        iris = load_iris()
        return pd.DataFrame(iris.data, columns=iris.feature_names), pd.Series(iris.target)
