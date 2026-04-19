import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """Encodes a categorical column as the mean of the target, fitted on training data only."""
    def __init__(self, col):
        self.col = col

    def fit(self, X, y):
        X_ = X.copy() if hasattr(X, "copy") else pd.DataFrame(X)
        self.enc_map_ = pd.Series(y).groupby(
            X_[self.col] if isinstance(X_, pd.DataFrame) else X_[:, self.col]
        ).mean()
        self.global_mean_ = float(pd.Series(y).mean())
        return self

    def transform(self, X):
        X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_["developer_enc"] = X_[self.col].map(self.enc_map_).fillna(self.global_mean_)
        X_.drop(columns=[self.col], inplace=True)
        return X_


def make_pipeline(model, scale=False):
    steps = [("target_enc", TargetEncoderCV(col="developer"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


# Tags/binary features surfaced in the single-game predictor UI
UI_TAGS = [
    "indie", "singleplayer", "action", "casual", "adventure",
    "strategy", "simulation", "rpg", "puzzle", "multiplayer",
    "horror", "anime", "vr", "visual_novel", "great_soundtrack",
    "pixel_graphics", "2d", "early_access", "classic",
]

# Numeric inputs: (min, max, default, step)
NUMERIC_UI = {
    "price":           (0.0,   100.0,  9.99,  0.5),
    "achievements":    (0,     1000,   20,    1),
    "owners_log":      (9.2,   18.8,   10.5,  0.1),
    "release_year":    (2000,  2024,   2018,  1),
    "release_month":   (1,     12,     6,     1),
    "processor_Ghz":   (0.5,   4.2,    2.0,   0.1),
    "RAM_mb":          (256,   32768,  4096,  256),
    "GPU_mb":          (0,     16384,  1024,  128),
    "storage_mb":      (50,    102400, 2000,  50),
    "average_playtime":(0,     2000,   0,     10),
    "median_playtime": (0,     2000,   0,     10),
}

RATING_COLOR = {"Good": "#4ade80", "Mixed": "#facc15", "Bad": "#f87171"}
RATING_5_COLOR = {
    "Positive":       "#16a34a",
    "Mostly Positive":"#4ade80",
    "Mixed":          "#facc15",
    "Mostly Negative":"#fb923c",
    "Negative":       "#ef4444",
}
