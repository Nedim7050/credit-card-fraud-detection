# Credit Card Fraud Detection

Detect fraudulent credit card transactions using classical ML models.

## Overview
- Input features: `V1..V28` (anonymized), `Time`, `Amount`
- Target: `Class` (1 = fraud, 0 = legit)
- Handles extreme class imbalance (~0.17% frauds)

## Quickstart

### 1) Clone and setup
```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

### 2) Data
- If you have Kaggle's `creditcard.csv`, place it at `data/raw/creditcard.csv`.
- Otherwise generate a synthetic dataset:
```bash
python scripts/generate_synthetic.py --rows 100000 --fraud-rate 0.0017
```

### 3) Exploration (optional)
Launch Jupyter and open the notebooks in `notebooks/`:
```bash
jupyter lab
```
- `01_EDA.ipynb`: data exploration and `sample_for_app.csv` export
- `02_modeling.ipynb`: baselines, tuning, SHAP demo

### 4) Preprocess and Train
```bash
python src/preprocess.py --save-scaler --test-size 0.2
python src/train.py --use-smote
```
Outputs:
- Best model: `models/fraud_model.joblib`
- Scaler: `models/scaler.joblib`
- Metrics: `models/metrics.txt`

### 5) Evaluate
```bash
python src/evaluate_model.py
```
Saves ROC and PR curves in `models/plots/`.

### 6) Run the Streamlit app
```bash
streamlit run app/app.py
```
- Upload a CSV for batch predictions
- Or use the single-transaction form
- Adjust the decision threshold and download annotated results

## Metrics: ROC-AUC vs PR-AUC
For highly imbalanced problems, **PR-AUC** is often more informative than ROC-AUC
because it focuses on precision/recall trade-offs on the positive (rare) class.
We report both for completeness.

## Deploy
- Streamlit Cloud or Hugging Face Spaces
- Set the app entrypoint to `app/app.py`

## Author
Nedim Mejri
