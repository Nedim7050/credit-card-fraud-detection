import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into stratified train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing features and target column `Class`.
    test_size : float
        Fraction for test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess(X: pd.DataFrame, scaler: RobustScaler | None = None) -> Tuple[pd.DataFrame, RobustScaler]:
    """Scale numerical features. Uses RobustScaler on `Amount` and leaves PCA V1..V28 as-is.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe including `Amount` (and optionally `Time`).
    scaler : RobustScaler | None
        Optional scaler to apply. If None, a new scaler is fitted.

    Returns
    -------
    X_scaled : pd.DataFrame
        Dataframe with scaled `Amount` and original other columns.
    scaler : RobustScaler
        The fitted scaler.
    """
    X = X.copy()

    # Ensure columns exist
    if "Amount" not in X.columns:
        raise ValueError("Expected column 'Amount' in features")

    if scaler is None:
        scaler = RobustScaler()
        X.loc[:, "Amount"] = scaler.fit_transform(X[["Amount"]])
    else:
        X.loc[:, "Amount"] = scaler.transform(X[["Amount"]])

    return X, scaler


def main():
    parser = argparse.ArgumentParser(description="Preprocess credit card fraud dataset and persist splits.")
    parser.add_argument("--raw", default=str(Path("data/raw/creditcard.csv")), help="Path to raw CSV")
    parser.add_argument("--out", default=str(Path("data/processed")), help="Output directory for splits")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--save-scaler", action="store_true", help="Persist the fitted scaler to models/scaler.joblib")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(str(raw_path))
    X_train, X_test, y_train, y_test = split_data(df, test_size=args.test_size)

    X_train_scaled, scaler = preprocess(X_train)
    X_test_scaled, _ = preprocess(X_test, scaler=scaler)

    # Persist
    X_train_scaled.to_csv(out_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    if args.save_scaler:
        joblib.dump(scaler, models_dir / "scaler.joblib")

    print(f"Saved processed splits to {out_dir}")
    if args.save_scaler:
        print("Saved scaler to models/scaler.joblib")


if __name__ == "__main__":
    main()
