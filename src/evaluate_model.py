from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt


def main():
    data_dir = Path("data/processed")
    models_dir = Path("models")
    plots_dir = models_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(models_dir / "fraud_model.joblib")

    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    report = classification_report(y_test, preds, digits=4)
    cm = confusion_matrix(y_test, preds)

    (models_dir / "evaluation.txt").write_text(
        f"ROC-AUC: {roc:.4f}\nPR-AUC: {pr:.4f}\n\nClassification Report:\n{report}\n\nConfusion Matrix:\n{cm}\n",
        encoding="utf-8",
    )

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("ROC Curve - Test")
    plt.savefig(plots_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Precision-Recall curve
    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title("Precision-Recall Curve - Test")
    plt.savefig(plots_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Test")
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved evaluation to {models_dir} and plots to {plots_dir}")


if __name__ == "__main__":
    main()
