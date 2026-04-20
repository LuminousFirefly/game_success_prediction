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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import auc, confusion_matrix, r2_score, roc_curve, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier, XGBRegressor

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
le5 = LabelEncoder()
le3 = LabelEncoder()
merge_map = {
    "Negative":        "Bad",
    "Mostly Negative": "Bad",
    "Mixed":           "Mixed",
    "Mostly Positive": "Good",
    "Positive":        "Good",
}
y_clf5 = le5.fit_transform(df["rating_category"])
y_clf3 = le3.fit_transform(df["rating_category"].map(merge_map))
y_reg  = df["wilson_score"].values

# --- Deployment models: trained on full dataset ---
print("Training deployment models (full dataset)...")
clf_pipe = make_pipeline(
    LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1,
    )
)
clf_pipe.fit(X, y_clf3)

reg_pipe = make_pipeline(
    XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
)
reg_pipe.fit(X, y_reg)

joblib.dump(clf_pipe, MODELS_DIR / "clf3_pipeline.joblib")
joblib.dump(reg_pipe, MODELS_DIR / "reg_pipeline.joblib")
joblib.dump(le3,      MODELS_DIR / "le3.joblib")
joblib.dump(le5,      MODELS_DIR / "le5.joblib")

with open(MODELS_DIR / "feature_cols.json", "w") as f:
    json.dump(selected_features, f)

defaults = {col: float(X[col].median()) for col in selected_features}
with open(MODELS_DIR / "feature_defaults.json", "w") as f:
    json.dump(defaults, f)

# --- Evaluation: 80/20 split ---
print("\nPrecomputing evaluation metrics (80/20 split)...")
X_tr, X_te, y5_tr, y5_te, y3_tr, y3_te, yr_tr, yr_te = train_test_split(
    X, y_clf5, y_clf3, y_reg,
    test_size=0.2, random_state=42, stratify=y_clf5,
)

clf5_defs = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), scale=True),
    "Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
    "XGBoost": make_pipeline(
        XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                      eval_metric="mlogloss", verbosity=0)),
}
clf3_defs = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), scale=True),
    "Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
    "XGBoost": make_pipeline(
        XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                      eval_metric="mlogloss", verbosity=0)),
}
reg_defs = {
    "Linear Regression": make_pipeline(LinearRegression(), scale=True),
    "Random Forest": make_pipeline(
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    "XGBoost": make_pipeline(
        XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)),
}

def compute_roc(pipeline, X_test, y_test, classes):
    y_bin   = label_binarize(y_test, classes=range(len(classes)))
    y_score = pipeline.predict_proba(X_test)
    out = {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        out[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc(fpr, tpr), 4)}
    return out

eval_data = {"clf5": {}, "clf3": {}, "reg": {}}

for name, pipe in clf5_defs.items():
    print(f"  clf5 — {name}")
    pipe.fit(X_tr, y5_tr)
    preds = pipe.predict(X_te)
    eval_data["clf5"][name] = {
        "test_accuracy": round(float((preds == y5_te).mean()), 4),
        "classes":       le5.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y5_te, preds).tolist(),
        "roc": compute_roc(pipe, X_te, y5_te, le5.classes_.tolist()),
    }

for name, pipe in clf3_defs.items():
    print(f"  clf3 — {name}")
    pipe.fit(X_tr, y3_tr)
    preds = pipe.predict(X_te)
    eval_data["clf3"][name] = {
        "test_accuracy": round(float((preds == y3_te).mean()), 4),
        "classes":       le3.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y3_te, preds).tolist(),
        "roc": compute_roc(pipe, X_te, y3_te, le3.classes_.tolist()),
    }

# Sample 1500 points for scatter/residual plots
sample_idx = np.random.RandomState(42).choice(len(X_te), size=min(1500, len(X_te)), replace=False)
X_te_s  = X_te.iloc[sample_idx]
yr_te_s = yr_te[sample_idx]

for name, pipe in reg_defs.items():
    print(f"  reg  — {name}")
    pipe.fit(X_tr, yr_tr)
    preds_full   = pipe.predict(X_te)
    preds_sample = pipe.predict(X_te_s)
    eval_data["reg"][name] = {
        "r2":   round(float(r2_score(yr_te, preds_full)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(yr_te, preds_full))), 4),
        "scatter": {
            "actual":    yr_te_s.tolist(),
            "predicted": preds_sample.tolist(),
        },
        "residuals": {
            "predicted": preds_sample.tolist(),
            "residuals": (yr_te_s - preds_sample).tolist(),
        },
    }

with open(MODELS_DIR / "eval_data.json", "w") as f:
    json.dump(eval_data, f)
print("  eval_data.json saved")

# --- EDA summaries ---
print("\nPrecomputing EDA summaries...")
eda = {}
eda["dist_5class"] = df["rating_category"].value_counts().to_dict()
eda["dist_3class"] = df["rating_category"].map(merge_map).value_counts().to_dict()

counts, edges = np.histogram(df["wilson_score"].dropna(), bins=40)
eda["wilson_hist"] = {"counts": counts.tolist(), "edges": edges.tolist()}

eda["feature_importances"] = (
    importance_df.sort_values("importance", ascending=False)
    .head(30)[["feature", "importance"]]
    .to_dict(orient="records")
)

tag_cols = [c for c in X.columns if c not in {
    "developer", "publisher", "support_email", "website", "support_url",
    "price", "achievements", "owners_log", "release_year", "release_month",
    "processor_Ghz", "RAM_mb", "GPU_mb", "storage_mb", "average_playtime", "median_playtime",
    "Steam Cloud_cat", "Steam Trading Cards_cat", "Full controller support_cat",
    "Partial Controller Support_cat", "mac", "mac_sup",
}]
eda["tag_frequency"] = X[tag_cols].sum().sort_values(ascending=False).head(25).to_dict()

price_data = df["price"].clip(upper=60)
counts_p, edges_p = np.histogram(price_data, bins=30)
eda["price_hist"] = {"counts": counts_p.tolist(), "edges": edges_p.tolist()}

sample = (
    df[["owners_log", "wilson_score", "rating_category"]]
    .dropna()
    .sample(n=min(2000, len(df)), random_state=42)
    .reset_index(drop=True)
)
eda["scatter_sample"] = sample.to_dict(orient="records")

with open(MODELS_DIR / "eda_data.json", "w") as f:
    json.dump(eda, f)

print(f"\nAll artifacts saved to {MODELS_DIR}")
print("Done! Commit the models/ folder and deploy.")
