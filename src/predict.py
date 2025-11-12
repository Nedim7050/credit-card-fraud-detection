from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


@dataclass
class PredictionResult:
    probability: float
    label: int


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has all expected columns with numeric dtype."""
    data = df.copy()

    # Drop target if present
    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    # Keep only relevant columns
    extra_cols = [col for col in data.columns if col not in EXPECTED_COLUMNS]
    if extra_cols:
        data = data.drop(columns=extra_cols)

    # Add missing columns with zeros
    for col in EXPECTED_COLUMNS:
        if col not in data.columns:
            data[col] = 0.0

    # Reorder
    data = data[EXPECTED_COLUMNS]

    # Ensure numeric dtype; replace NaNs with 0
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return data


def predict_single(model, scaler, data_dict: Dict[str, float], threshold: float = 0.5) -> PredictionResult:
    """Predict for a single transaction."""
    row = {col: data_dict.get(col, 0.0) for col in EXPECTED_COLUMNS}
    df = pd.DataFrame([row])
    df = _prepare_dataframe(df)

    if scaler is not None:
        df.loc[:, "Amount"] = scaler.transform(df[["Amount"]])

    proba = float(model.predict_proba(df)[:, 1][0])
    label = int(proba >= threshold)
    return PredictionResult(probability=proba, label=label)


def predict_batch(model, scaler, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Predict probabilities for a batch of transactions."""
    data = _prepare_dataframe(df)

    if scaler is not None:
        data.loc[:, "Amount"] = scaler.transform(data[["Amount"]])

    proba = model.predict_proba(data)[:, 1]
    label = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_proba"] = proba.astype(float)
    out["fraud_label"] = label
    return out
