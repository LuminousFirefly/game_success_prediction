# Project Structure

> **CSC 240 — Game Success Prediction**
> Predicting Steam game quality ratings using machine learning on ~27,000 games.

---

## Directory Layout

```
Game Success Prediction/
│
├── dataset/                        # Raw data (read-only, sourced from Kaggle)
│   ├── steam.csv                   # Core game metadata: price, platforms, ratings, playtime etc.
│   ├── steam_requirements_data.csv # System requirements (CPU, RAM, GPU, storage)
│   ├── steam_support_info.csv      # Support URLs and contact emails
│   ├── steamspy_tag_data.csv       # Crowdsourced genre/tag vote counts (~400 tags)
│   ├── steam_description_data.csv  # Game descriptions (unused in modelling)
│   └── steam_media_data.csv        # Screenshots/trailer URLs (unused in modelling)
│
├── data_preprocessing/             # All cleaning and feature-engineering work
│   ├── Cleaning_real.ipynb         # MAIN cleaning notebook 
│   ├── cleaning_labels.ipynb       # Discretises wilson_score into rating categories
│   ├── cleaning_dim_reduction.ipynb# Experiments with dimensionality reduction and dataset cleaning
│   ├── Dataset_Analysis.ipynb      # EDA: distributions, correlations, outlier checks
│   ├── cleaning.ipynb              # Early draft
│   └── datasets/                   # Intermediate CSVs produced during cleaning
│       ├── steam_merged.csv        # Step 1: raw join of all source tables
│       ├── steam_proper.csv        # Step 2: columns dropped, types fixed
│       ├── steam_edit.csv          # Step 3: outliers removed, booleans encoded
│       ├── tech_spec.csv           # Tech specs extracted from requirements strings
│       ├── tech_spec2.csv          # Tech specs after outlier removal
│       ├── tech_spec_cleaned (1).csv
│       ├── steam_final.csv         # Step 4: tech specs merged in
│       └── steam_final_labelled.csv# Step 5: wilson_score + rating_category added
│
├── cleaned_dataset/                # Final outputs consumed by classification
│   ├── steam_finalized_dataset.csv # THE dataset used for modelling (~27k rows, 499 cols)
│   └── feature_importances.csv     # Random Forest importances used for feature selection
│
├── classification.ipynb            # MAIN modelling/classification notebook
│
└── plan.txt                        # Original project plan and role assignments
```

---

## Key Files at a Glance

| File | Purpose |
|------|---------|
| `dataset/` | Raw source data — never modified | 
| `data_preprocessing/Cleaning_real.ipynb` | Data cleaning pipeline | 
| `cleaned_dataset/steam_finalized_dataset.csv` | Model-ready dataset | 
| `cleaned_dataset/feature_importances.csv` | Feature selection input | 
| `classification.ipynb` | All classification & regression models | 

---

## Data Pipeline

```
Raw CSVs (dataset/)
       │
       ▼
  Merge & join  ──────────────────────────────────────────┐
  (steam_merged.csv)                                       │
       │                                                   │
       ▼                                                   │
  Drop cols, fix types, remove outliers                    │
  (steam_proper.csv → steam_edit.csv)                      │
       │                                                   │
       ▼                                                   ▼
  Parse tech specs ──────────────────► Merge tech specs
  (tech_spec.csv → tech_spec2.csv)     (steam_final.csv)
                                              │
                                              ▼
                                   Add wilson_score + rating_category
                                   (steam_final_labelled.csv)
                                              │
                                              ▼
                                   One-hot encode tags/genres/categories
                                   (steam_finalized_dataset.csv)  ← model input
```

---

## Target Variables

| Variable | Type | Description |
|----------|------|-------------|
| `wilson_score` | Continuous \[0–1\] | Wilson lower bound on positive review ratio — penalises low review counts |
| `rating_category` | 5-class | Negative / Mostly Negative / Mixed / Mostly Positive / Positive |
| `rating_3class` | 3-class | Bad / Mixed / Good (merged for classification — less boundary ambiguity) |

---

## Feature Groups

| Group | Count | Examples |
|-------|-------|---------|
| Game metadata | ~10 | `price`, `achievements`, `release_year`, `release_month` |
| Platform / tech specs | ~8 | `RAM_mb`, `GPU_mb`, `storage_mb`, `processor_Ghz` |
| Steam tags (binary) | ~400 | `indie`, `action`, `casual`, `anime`, `vr`, … |
| Steam genres (binary) | ~30 | `Action_genre`, `RPG_genre`, `Simulation_genre`, … |
| Steam categories (binary) | ~28 | `Single-player_cat`, `Steam Cloud_cat`, … |
| Engagement | ~4 | `owners_log`, `average_playtime`, `median_playtime` |
| Support presence | ~3 | `website`, `support_url`, `support_email` |

> Feature selection in `classification.ipynb` keeps features with Random Forest importance > 0.003, plus `average_playtime` and `median_playtime` added back manually.

---

## Models

### Classification (`classification.ipynb`)

| Model | Notes |
|-------|-------|
| Naive Bayes | Baseline |
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | 200 trees, `class_weight='balanced'` |
| XGBoost | 500 trees, `lr=0.05`, `max_depth=6`, subsampling |
| LightGBM | 500 trees, `lr=0.05`, `num_leaves=63`, `class_weight='balanced'` |
| KNN | k=10, scaled |

### Regression (`classification.ipynb`)

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Random Forest | 200 trees |
| XGBoost | 500 trees, `lr=0.05`, `max_depth=6` |

### Evaluation

- **Classification**: 5-fold stratified CV → accuracy + weighted F1
- **Regression**: 5-fold KFold CV → R² + RMSE
- Train/test split: 80/20, stratified by `rating_category`

---

## Roles

| Person | Responsibility |
|--------|---------------|
| Alvis | Data cleaning & preprocessing (Step 1) |
| Rainah | Clustering (Step 2) |
| Aryan | Classification models (Step 3a, 3b), Data Cleaning & preprcoessing (Step 1) |
| Amanda | ROC curves, evaluation metrics (Step 3c, 3d) |
