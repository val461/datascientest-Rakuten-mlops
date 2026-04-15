from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    """Remplacer par vrai preprocessing"""
    print("🔄 Preprocessing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)
