import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
except Exception as exc:  # pragma: no cover
    raise SystemExit("imbalanced-learn is required: pip install imbalanced-learn") from exc


SEED = 42
N_SPLITS = 5


def load_processed(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series, use_smote: bool) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    roc_list: list[float] = []
    pr_list: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if use_smote:
            sm = SMOTE(random_state=SEED)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, proba)
        pr = average_precision_score(y_val, proba)
        roc_list.append(roc)
        pr_list.append(pr)

    return float(np.mean(roc_list)), float(np.mean(pr_list))


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models and select the best.")
    parser.add_argument("--data-dir", default=str(Path("data/processed")), help="Processed data directory")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE on training folds")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed(data_dir)

    candidates: list[tuple[str, object]] = [
        ("log_reg", LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED, n_jobs=None)),
        ("rf", RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced", random_state=SEED)),
    ]
    if HAS_XGB:
        candidates.append(
            (
                "xgb",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            )
        )

    best_name = None
    best_model = None
    best_score = -np.inf
    lines: list[str] = []

    for name, model in candidates:
        roc, pr = evaluate_cv(model, X_train, y_train, use_smote=args.use_smote)
        lines.append(f"Model={name} | CV ROC-AUC={roc:.4f} | CV PR-AUC={pr:.4f}")
        # prefer PR-AUC; tiebreaker ROC
        score = pr * 1.0 + 0.1 * roc
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    assert best_model is not None
    # Fit on full training (optionally SMOTE). SMOTE should be applied only to training data.
    X_tr, y_tr = (X_train, y_train)
    if args.use_smote:
        sm = SMOTE(random_state=SEED)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    best_model.fit(X_tr, y_tr)

    # Persist model
    joblib.dump(best_model, models_dir / "fraud_model.joblib")

    # Evaluate on held-out test for info
    proba_test = best_model.predict_proba(X_test)[:, 1]
    roc_test = roc_auc_score(y_test, proba_test)
    pr_test = average_precision_score(y_test, proba_test)

    lines.append("")
    lines.append(f"Selected={best_name} | Test ROC-AUC={roc_test:.4f} | Test PR-AUC={pr_test:.4f}")

    (models_dir / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Saved model to models/fraud_model.joblib and metrics to models/metrics.txt")


if __name__ == "__main__":
    main()
