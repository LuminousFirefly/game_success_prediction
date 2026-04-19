# Game Success Prediction

Predicting the quality rating of Steam games using machine learning — applied to a dataset of ~27,000 titles.

**Course:** CSC 240 · Semester 8

---

## Overview

Can a game's metadata predict how well it will be received? This project uses Steam store data to predict game quality, measured by a [Wilson lower bound](https://medium.com/tech-that-works/wilson-lower-bound-score-and-bayesian-approximation-for-k-star-scale-rating-to-rate-products-c67ec6e30060) score derived from user reviews.

We approach the problem two ways:
- **Classification** — predict a rating category (Bad / Mixed / Good)
- **Regression** — predict the continuous Wilson score directly

---

## Dataset

Raw data sourced from Kaggle's Steam Store dataset, joined across six tables:

| Source | Contents |
|--------|---------|
| `steam.csv` | Price, platforms, playtime, review counts |
| `steam_requirements_data.csv` | CPU, RAM, GPU, storage requirements |
| `steam_support_info.csv` | Support URLs and contact info |
| `steamspy_tag_data.csv` | ~400 crowdsourced genre/tag votes |

After cleaning: **26,821 games × 499 features**

---

## Methodology

### 1. Data Preprocessing
- Merged all source tables on `appid`
- Parsed unstructured system requirements into numeric columns (RAM, GPU, storage, CPU GHz)
- Removed games with fewer than a minimum review threshold
- One-hot encoded tags, genres, and Steam categories
- Computed `wilson_score` from raw positive/negative review counts
- Discretised into 5-class and 3-class rating labels

### 2. Feature Selection
Random Forest importances computed on the full feature set; features with importance > 0.003 retained (~130 features). Top predictors: `owners_log`, `achievements`, `indie`, `price`, `singleplayer`.

### 3. Models
- Logistic Regression, Random Forest, XGBoost, LightGBM, KNN, Naive Bayes
- All tree models tuned (learning rate, depth, subsampling)
- Class imbalance handled via `class_weight='balanced'`
- Developer target-encoded within cross-validation folds to prevent leakage

---

## Results

| Task | Model | Accuracy / R² |
|------|-------|--------------|
| Classification (3-class) | LightGBM / XGBoost | ~58–60% |
| Classification (5-class) | Random Forest | ~40% |
| Regression | XGBoost | R² ≈ 0.27 |

> 3-class accuracy of 58–60% against a 33% random baseline. The 5-class problem is inherently harder due to fuzzy boundaries between adjacent categories.

---

## Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a full breakdown of every file and folder.

```
Game Success Prediction/
├── dataset/                  # Raw source CSVs (read-only)
├── data_preprocessing/       # Cleaning notebooks + intermediate datasets
├── cleaned_dataset/          # Final model-ready dataset + feature importances
├── classification.ipynb      # All models, CV, evaluation
└── plan.txt                  # Project plan and role assignments
```

---

## Team

| Name | Role |
|------|------|
| Alvis | Data cleaning & preprocessing |
| Rainah | Clustering |
| Aryan | Classification models, Data cleaning & preprocessing |
| Amanda | ROC curves & evaluation metrics |
