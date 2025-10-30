import io
from pathlib import Path
import sys

# Ensure project root is on sys.path for `import src.*` when running from app/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import predict_single, predict_batch

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🔍", layout="wide")

# Light styling
st.markdown(
	"""
	<style>
		.main > div { padding-top: 1rem; }
		.block-container { padding-top: 1rem; }
		.sidebar .sidebar-content { background-color: #0e1117; }
		.metric { border: 1px solid #eaeaea; border-radius: 8px; padding: 8px; }
	</style>
	""",
	unsafe_allow_html=True,
)

st.title("🔍 Credit Card Fraud Detector")

models_dir = Path("models")
model_path = models_dir / "fraud_model.joblib"
scaler_path = models_dir / "scaler.joblib"
metrics_path = models_dir / "metrics.txt"

model = joblib.load(model_path) if model_path.exists() else None
scaler = joblib.load(scaler_path) if scaler_path.exists() else None

# Sidebar controls
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
st.sidebar.caption("Lower threshold ⇒ more recalls, higher threshold ⇒ more precision")

# Show metrics if available
def parse_metrics(path: Path) -> tuple[str | None, float | None, float | None]:
	try:
		text = path.read_text(encoding="utf-8")
		last = [ln for ln in text.splitlines() if ln.strip()][-1]
		# Expected: Selected=xgb | Test ROC-AUC=0.9784 | Test PR-AUC=0.6970
		parts = [p.strip() for p in last.split("|")]
		selected = parts[0].split("=")[1].strip() if len(parts) > 0 and "=" in parts[0] else None
		roc = float(parts[1].split("=")[1]) if len(parts) > 1 and "=" in parts[1] else None
		pr = float(parts[2].split("=")[1]) if len(parts) > 2 and "=" in parts[2] else None
		return selected, roc, pr
	except Exception:
		return None, None, None

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
	st.subheader("Model Metrics")
if metrics_path.exists():
	selected, roc_val, pr_val = parse_metrics(metrics_path)
	with col_m2:
		st.metric("Test ROC-AUC", f"{roc_val:.4f}" if roc_val is not None else "-")
	with col_m3:
		st.metric("Test PR-AUC", f"{pr_val:.4f}" if pr_val is not None else "-")
	st.caption(f"Selected model: {selected}")
else:
	st.info("No metrics available. Train the model to generate metrics.")

st.divider()

col_left, col_right = st.columns([1, 1])

# Single prediction
with col_left:
	st.subheader("Single Transaction Prediction")
	amount = st.number_input("Amount", min_value=0.0, value=100.0, step=1.0)
	time_val = st.number_input("Time", min_value=0.0, value=0.0, step=1.0)

	v_inputs = {}
	with st.expander("V1..V5 (optional)"):
		for i in range(1, 6):
			v_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, format="%.4f")

	col_btn1, col_btn2 = st.columns(2)
	with col_btn1:
		predict_single_btn = st.button("Predict Single")
	with col_btn2:
		demo_single_btn = st.button("Try Demo Single")

	if predict_single_btn:
		if model is None:
			st.error("Model not found. Train the model first.")
		else:
			features = {**{"Amount": amount, "Time": time_val}, **v_inputs}
			res = predict_single(model, scaler, features, threshold=threshold)
			st.metric("Fraud Probability", f"{res.probability:.4f}")
			st.write("Predicted Label:", int(res.label))

	if demo_single_btn:
		if model is None:
			st.error("Model not found. Train the model first.")
		else:
			# simple demo values
			demo_features = {"Amount": 250.0, "Time": 12345.0, "V1": 2.0, "V2": -1.5, "V3": 0.7}
			res = predict_single(model, scaler, demo_features, threshold=threshold)
			st.success("Demo single prediction executed")
			st.metric("Fraud Probability", f"{res.probability:.4f}")
			st.write("Predicted Label:", int(res.label))

# Batch prediction
with col_right:
	st.subheader("Batch Prediction")
	uploaded = st.file_uploader("Upload CSV", type=["csv"])
	col_b1, col_b2, col_b3 = st.columns(3)
	with col_b1:
		predict_batch_btn = st.button("Predict Batch")
	with col_b2:
		use_processed_btn = st.button("Use processed sample")
	with col_b3:
		mini_sample_btn = st.button("Generate mini sample")

	def run_batch(df: pd.DataFrame):
		result = predict_batch(model, scaler, df, threshold=threshold)
		st.write("Top 10 Highest-Risk Transactions")
		st.dataframe(result.sort_values("fraud_proba", ascending=False).head(10))
		csv_buf = io.StringIO()
		result.to_csv(csv_buf, index=False)
		st.download_button(
			label="Download results",
			data=csv_buf.getvalue(),
			file_name="predictions.csv",
			mime="text/csv",
		)

	if predict_batch_btn:
		if model is None:
			st.error("Model not found. Train the model first.")
		elif uploaded is None:
			st.warning("Please upload a CSV or use one of the demo buttons.")
		else:
			df = pd.read_csv(uploaded)
			run_batch(df)

	if use_processed_btn:
		if model is None:
			st.error("Model not found. Train the model first.")
		else:
			path = Path("data/processed/sample_for_app.csv")
			if not path.exists():
				st.warning("sample_for_app.csv not found. Open 01_EDA notebook to generate it or use mini sample.")
			else:
				df = pd.read_csv(path)
				st.info(f"Loaded {path}")
				run_batch(df)

	if mini_sample_btn:
		if model is None:
			st.error("Model not found. Train the model first.")
		else:
			# Create a tiny 10-row sample with columns expected by the model
			cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
			df = pd.DataFrame(0.0, index=range(10), columns=cols)
			df.loc[:, "Time"] = np.linspace(0, 20000, 10)
			df.loc[:, "Amount"] = np.linspace(20, 500, 10)
			st.info("Generated mini sample (10 rows)")
			run_batch(df)
