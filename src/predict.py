from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    probability: float
    label: int


def predict_single(model, scaler, data_dict: Dict[str, float], threshold: float = 0.5) -> PredictionResult:
    """Predict for a single transaction.

    Parameters
    ----------
    model : object
        Trained classifier with predict_proba.
    scaler : sklearn-like scaler
        Fitted scaler for the `Amount` feature.
    data_dict : Dict[str, float]
        Dictionary of features including keys V1..V28 (optional subset), `Amount`, and optionally `Time`.
        Missing features will be filled with 0.
    threshold : float
        Decision threshold for positive class.
    """
    # Define expected columns order
    base_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

    row = {col: data_dict.get(col, 0.0) for col in base_cols}
    df = pd.DataFrame([row])

    if scaler is not None:
        df.loc[:, "Amount"] = scaler.transform(df[["Amount"]])

    proba = float(model.predict_proba(df)[:, 1][0])
    label = int(proba >= threshold)
    return PredictionResult(probability=proba, label=label)


def predict_batch(model, scaler, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Predict probabilities for a batch of transactions.

    Parameters
    ----------
    model : object
        Trained classifier with predict_proba.
    scaler : sklearn-like scaler
        Fitted scaler for the `Amount` feature.
    df : pd.DataFrame
        Dataframe with columns V1..V28, Time, Amount (no Class column).
    threshold : float
        Decision threshold.

    Returns
    -------
    pd.DataFrame
        Original df with added columns `fraud_proba` and `fraud_label`.
    """
    data = df.copy()
    if "Class" in data.columns:
        data = data.drop(columns=["Class"])  # ensure no target

    if scaler is not None and "Amount" in data.columns:
        data.loc[:, "Amount"] = scaler.transform(data[["Amount"]])

    proba = model.predict_proba(data)[:, 1]
    label = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_proba"] = proba
    out["fraud_label"] = label
    return out
