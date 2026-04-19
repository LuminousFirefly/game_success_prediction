"""
Run this once locally before deploying:
    python streamlit_app/train_models.py

Saves trained pipelines + metadata to streamlit_app/models/.
Commit the models/ folder so Streamlit Cloud can load them.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from utils import make_pipeline

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "cleaned_dataset"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_DIR / "steam_finalized_dataset.csv")
importance_df = pd.read_csv(DATA_DIR / "feature_importances.csv")

# --- Feature selection (mirrors classification.ipynb exactly) ---
threshold = 0.003
selected_features = importance_df[
    importance_df["importance"] > threshold
]["feature"].tolist()
for col in ["average_playtime", "median_playtime"]:
    if col in df.columns and col not in selected_features:
        selected_features.append(col)
print(f"Selected {len(selected_features)} features")

X = df[selected_features].copy()

# --- Targets ---
le3 = LabelEncoder()
merge_map = {
    "Negative":        "Bad",
    "Mostly Negative": "Bad",
    "Mixed":           "Mixed",
    "Mostly Positive": "Good",
    "Positive":        "Good",
}
y_clf3 = le3.fit_transform(df["rating_category"].map(merge_map))
y_reg  = df["wilson_score"].values

y_clf5_raw = df["rating_category"]

# Train on the full dataset for deployment (CV metrics reported separately in app)
print("Training 3-class LightGBM classifier...")
clf_pipe = make_pipeline(
    LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1,
    )
)
clf_pipe.fit(X, y_clf3)

print("Training XGBoost regressor...")
reg_pipe = make_pipeline(
    XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
)
reg_pipe.fit(X, y_reg)

# --- Save pipelines and metadata ---
joblib.dump(clf_pipe, MODELS_DIR / "clf3_pipeline.joblib")
joblib.dump(reg_pipe, MODELS_DIR / "reg_pipeline.joblib")
joblib.dump(le3,      MODELS_DIR / "le3.joblib")

with open(MODELS_DIR / "feature_cols.json", "w") as f:
    json.dump(selected_features, f)

# Default values for features not in the UI (median of training data)
defaults = {col: float(X[col].median()) for col in selected_features}
with open(MODELS_DIR / "feature_defaults.json", "w") as f:
    json.dump(defaults, f)

# --- Precompute EDA summaries (avoids loading the 100MB CSV on Streamlit Cloud) ---
print("Precomputing EDA summaries...")

eda = {}

# Class distributions
eda["dist_5class"] = df["rating_category"].value_counts().to_dict()
eda["dist_3class"] = df["rating_category"].map(merge_map).value_counts().to_dict()

# Wilson score histogram
counts, edges = np.histogram(df["wilson_score"].dropna(), bins=40)
eda["wilson_hist"] = {
    "counts": counts.tolist(),
    "edges":  edges.tolist(),
}

# Top 30 feature importances
eda["feature_importances"] = (
    importance_df.sort_values("importance", ascending=False)
    .head(30)
    [["feature", "importance"]]
    .to_dict(orient="records")
)

# Top 25 most common binary tags (sum of 1s across rows)
tag_cols = [c for c in X.columns if c not in list({"developer","publisher","support_email",
    "website","support_url","price","achievements","owners_log","release_year","release_month",
    "processor_Ghz","RAM_mb","GPU_mb","storage_mb","average_playtime","median_playtime",
    "Steam Cloud_cat","Steam Trading Cards_cat","Full controller support_cat",
    "Partial Controller Support_cat","mac","mac_sup"})]
tag_sums = X[tag_cols].sum().sort_values(ascending=False).head(25)
eda["tag_frequency"] = tag_sums.to_dict()

# Price distribution histogram (capped at $60)
price_data = df["price"].clip(upper=60)
counts_p, edges_p = np.histogram(price_data, bins=30)
eda["price_hist"] = {"counts": counts_p.tolist(), "edges": edges_p.tolist()}

# Scatter sample: owners_log vs wilson_score (2 000 points, stratified by rating)
sample = (
    df[["owners_log", "wilson_score", "rating_category"]]
    .dropna()
    .groupby("rating_category", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), 400), random_state=42))
    .reset_index(drop=True)
)
eda["scatter_sample"] = sample.to_dict(orient="records")

with open(MODELS_DIR / "eda_data.json", "w") as f:
    json.dump(eda, f)

print(f"\nAll artifacts saved to {MODELS_DIR}")
print("Done! Commit the models/ folder and deploy.")
