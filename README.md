# 🔍 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)

> A machine learning system for detecting fraudulent credit card transactions using ensemble methods and deployed as an interactive web application.

🌐 **Live Application**: [https://credit-card-fraud-detection-hrczoamwj8xufv5umabmg6.streamlit.app/](https://credit-card-fraud-detection-hrczoamwj8xufv5umabmg6.streamlit.app/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Performance Metrics](#-performance-metrics)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

This project implements a comprehensive fraud detection system capable of identifying fraudulent credit card transactions in real-time. The system handles highly imbalanced datasets (fraud rate ~0.17%) using advanced machine learning techniques including SMOTE oversampling, ensemble methods, and optimized hyperparameter tuning.

### Key Highlights

- **Robust ML Pipeline**: Preprocessing, feature engineering, model training, and evaluation
- **Interactive Web App**: User-friendly Streamlit interface for single and batch predictions
- **Production-Ready**: Deployed on Streamlit Cloud with automatic dependency management
- **Comprehensive Analysis**: Jupyter notebooks for exploratory data analysis and model interpretation

---

## ✨ Features

- 🔄 **Automated Preprocessing**: Robust scaling and data splitting with stratified sampling
- 🤖 **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost with cross-validation
- ⚖️ **Imbalance Handling**: SMOTE oversampling and class-weighted models
- 📊 **Interactive Dashboard**: Real-time predictions with adjustable decision thresholds
- 📈 **Performance Visualization**: ROC curves, Precision-Recall curves, and confusion matrices
- 🔍 **Model Interpretability**: SHAP integration for feature importance analysis
- 💾 **Batch Processing**: Upload CSV files for bulk transaction analysis
- 📥 **Export Results**: Download predictions with fraud probabilities and labels

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/                    # Raw dataset (creditcard.csv)
│   └── processed/              # Processed splits (X_train, X_test, y_train, y_test)
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   └── 02_modeling.ipynb      # Model training and evaluation
├── src/
│   ├── preprocess.py          # Data preprocessing and scaling
│   ├── train.py               # Model training pipeline
│   ├── evaluate_model.py      # Model evaluation and visualization
│   └── predict.py             # Prediction functions (single & batch)
├── app/
│   └── app.py                 # Streamlit web application
├── models/
│   ├── fraud_model.joblib     # Trained XGBoost model
│   ├── scaler.joblib          # Fitted RobustScaler
│   ├── metrics.txt            # Performance metrics
│   └── plots/                 # Evaluation plots (ROC, PR curves)
├── scripts/
│   └── generate_synthetic.py  # Synthetic data generator
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## 🛠 Technologies Used

- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, SHAP
- **Web Framework**: Streamlit
- **Model Persistence**: joblib
- **Development**: Jupyter Lab

---

## 📊 Performance Metrics

The best-performing model (XGBoost) achieved:

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.9784 |
| **PR-AUC** | 0.6970 |

> **Note**: For highly imbalanced problems, PR-AUC (Precision-Recall Area Under Curve) is often more informative than ROC-AUC as it focuses on the precision/recall trade-offs for the rare positive class.

### Model Comparison

| Model | CV ROC-AUC | CV PR-AUC |
|-------|------------|-----------|
| Logistic Regression | 0.9907 | 0.4677 |
| Random Forest | 0.9498 | 0.1879 |
| **XGBoost** | **0.9610** | **0.5912** |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nedim7050/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset**
   
   Option A: Use Kaggle dataset
   - Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place it in `data/raw/creditcard.csv`
   
   Option B: Generate synthetic data
   ```bash
   python scripts/generate_synthetic.py --rows 100000 --fraud-rate 0.0017
   ```

5. **Preprocess and train the model**
   ```bash
   python src/preprocess.py --save-scaler --test-size 0.2
   python src/train.py --use-smote
   ```

6. **Evaluate the model** (optional)
   ```bash
   python src/evaluate_model.py
   ```

7. **Run the Streamlit app**
   ```bash
   streamlit run app/app.py
   ```

---

## 💻 Usage

### Web Application

1. **Single Transaction Prediction**
   - Enter transaction details (Amount, Time, optional V1-V5 features)
   - Adjust the decision threshold slider
   - Click "Predict Single" or "Try Demo Single" for a quick test

2. **Batch Prediction**
   - Upload a CSV file with transaction data
   - Or use "Generate mini sample" for a quick demo
   - View top 10 highest-risk transactions
   - Download results as CSV

### Python API

```python
from src.predict import predict_single, predict_batch
import joblib

# Load model and scaler
model = joblib.load('models/fraud_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Single prediction
result = predict_single(
    model, scaler,
    data_dict={'Amount': 250.0, 'Time': 12345.0, 'V1': 2.0},
    threshold=0.5
)
print(f"Fraud probability: {result.probability:.4f}")
print(f"Predicted label: {result.label}")

# Batch prediction
import pandas as pd
df = pd.read_csv('your_transactions.csv')
results = predict_batch(model, scaler, df, threshold=0.5)
```

### Jupyter Notebooks

- **`01_EDA.ipynb`**: Explore data distributions, correlations, and class imbalance
- **`02_modeling.ipynb`**: Train models, perform hyperparameter tuning, and generate SHAP plots

---

## 🌐 Deployment

### Streamlit Cloud

✅ **Already Deployed!** Access the live app: [https://credit-card-fraud-detection-hrczoamwj8xufv5umabmg6.streamlit.app/](https://credit-card-fraud-detection-hrczoamwj8xufv5umabmg6.streamlit.app/)

To deploy your own instance:

1. Fork this repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select your repository
5. Set main file path: `app/app.py`
6. Click "Deploy"

Streamlit Cloud will automatically install dependencies from `requirements.txt`.

### Local Deployment

```bash
streamlit run app/app.py
```

The app will be available at `http://localhost:8501`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Nedim Mejri**

- GitHub: [@Nedim7050](https://github.com/Nedim7050)
- Project Link: [https://github.com/Nedim7050/credit-card-fraud-detection](https://github.com/Nedim7050/credit-card-fraud-detection)

---

## 🙏 Acknowledgments

- Dataset inspiration from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Streamlit for the web framework
- The open-source ML community

---

⭐ If you find this project helpful, please consider giving it a star!
