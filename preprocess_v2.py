"""
preprocess_v2.py

Reads the FronkonGames Steam Games Dataset and produces
cleaned_dataset/steam_finalized_dataset_2.csv in the same format
as steam_finalized_dataset.csv, so the existing classification.ipynb
pipeline can be run on it with minimal changes.

Download the dataset from:
  https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
Place games.json (or games.csv) in the project root before running.

Usage:
    python preprocess_v2.py

Differences from v1:
  - Hardware requirements (processor_Ghz, RAM_mb, GPU_mb, storage_mb)
    are NOT available in this dataset — those columns are omitted.
  - New features added: peak_ccu, dlc_count, recommendations,
    average_playtime_2w, median_playtime_2w.
  - website / support_url / support_email are binary (0/1) instead of
    label-encoded integers, since raw URLs are available.
  - ~15x more games (124k raw vs 27k in v1).
"""

import itertools
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent
OUT_FILE = ROOT / "cleaned_dataset" / "steam_finalized_dataset_2.csv"

# ── Locate input file ──────────────────────────────────────────────────────
for candidate in [
    ROOT / "games.json",
    ROOT / "data" / "games.json",
    ROOT / "games.csv",
    ROOT / "data" / "games.csv",
]:
    if candidate.exists():
        IN_FILE = candidate
        break
else:
    raise FileNotFoundError(
        "games.json not found. Download from:\n"
        "  https://www.kaggle.com/datasets/fronkongames/steam-games-dataset\n"
        "and place it in the project root."
    )

print(f"Input file : {IN_FILE}")
print(f"Output file: {OUT_FILE}")


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def wilson_score(pos, neg, confidence=0.95):
    """Wilson lower-bound score (same formula as Cleaning_real.ipynb)."""
    n = pos + neg
    if n == 0:
        return 0.0
    p   = pos / n
    z   = norm.ppf((1 + confidence) / 2)
    num = p + z**2 / (2*n) - z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)
    den = 1 + z**2 / n
    return float(num / den)


def categorize_wilson(score):
    if score < 0.25:   return "Negative"
    elif score < 0.50: return "Mostly Negative"
    elif score < 0.70: return "Mixed"
    elif score < 0.83: return "Mostly Positive"
    else:              return "Positive"


# SteamSpy estimated_owners ranges → midpoint
_OWNER_MIDPOINTS = {
    "0 - 20000":              10_000,
    "20000 - 50000":          35_000,
    "50000 - 100000":         75_000,
    "100000 - 200000":       150_000,
    "200000 - 500000":       350_000,
    "500000 - 1000000":      750_000,
    "1000000 - 2000000":   1_500_000,
    "2000000 - 5000000":   3_500_000,
    "5000000 - 10000000":  7_500_000,
    "10000000 - 20000000": 15_000_000,
    "20000000 - 50000000": 35_000_000,
    "50000000 - 100000000":75_000_000,
}

def owners_to_log(s):
    s = str(s).strip()
    mid = _OWNER_MIDPOINTS.get(s)
    if mid is None:
        m = re.match(r"(\d+)\s*-\s*(\d+)", s)
        if m:
            mid = (int(m.group(1)) + int(m.group(2))) / 2
        else:
            return np.nan
    return np.log(max(mid, 1))


def norm_tag(t):
    """Normalise a tag name to match v1 column format (lowercase, underscores)."""
    return re.sub(r"[^a-z0-9]+", "_", t.lower().strip()).strip("_")


def parse_list_field(val):
    """Parse a field that may be a Python list, JSON string, or comma string."""
    if isinstance(val, list):
        return [str(v).strip() for v in val if v]
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v]
            if isinstance(parsed, dict):
                return [str(k).strip() for k in parsed.keys()]
        except (json.JSONDecodeError, ValueError):
            return [v.strip() for v in val.split(",") if v.strip()]
    return []


def parse_tags_field(val):
    """Tags may be a dict {name: votes} or a list of names."""
    if isinstance(val, dict):
        return list(val.keys())
    return parse_list_field(val)


def first_of_list(val):
    """Return the first element of a list field, or the value itself."""
    items = parse_list_field(val)
    return items[0] if items else ""


# ══════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════

print("\n── Loading data ──────────────────────────────────────────────")
if IN_FILE.suffix == ".json":
    print("  Parsing JSON (may take ~30s for 800 MB)...")
    with open(IN_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    # games.json is {AppID: {fields...}}
    if isinstance(raw, dict):
        records = [{"appid": k, **v} for k, v in raw.items()]
    else:
        records = raw
    df = pd.DataFrame(records)
else:
    print("  Reading CSV...")
    df = pd.read_csv(IN_FILE, low_memory=False)

print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

# ── Normalise column names (handle both "AppID" and "appid" variants) ──
rename_map = {
    # Title-case variants (CSV export)
    "AppID":                     "appid",
    "Name":                      "name",
    "Release date":              "release_date",
    "Estimated owners":          "estimated_owners",
    "Peak CCU":                  "peak_ccu",
    "Required age":              "required_age",
    "Price":                     "price",
    "DLC count":                 "dlc_count",
    "Supported languages":       "supported_languages",
    "Reviews":                   "reviews_text",
    "Website":                   "website",
    "Support url":               "support_url",
    "Support email":             "support_email",
    "Windows":                   "windows",
    "Mac":                       "mac",
    "Linux":                     "linux",
    "Metacritic score":          "metacritic_score",
    "User score":                "user_score",
    "Positive":                  "positive",
    "Negative":                  "negative",
    "Achievements":              "achievements",
    "Recommendations":           "recommendations",
    "Average playtime forever":  "average_playtime",
    "Average playtime 2 weeks":  "average_playtime_2w",
    "Median playtime forever":   "median_playtime",
    "Median playtime 2 weeks":   "median_playtime_2w",
    "Developers":                "developers",
    "Publishers":                "publishers",
    "Categories":                "categories",
    "Genres":                    "genres",
    "Tags":                      "tags",
    # Lowercase/underscore variants (actual JSON field names)
    "average_playtime_forever":  "average_playtime",
    "average_playtime_2weeks":   "average_playtime_2w",
    "median_playtime_forever":   "median_playtime",
    "median_playtime_2weeks":    "median_playtime_2w",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


# ══════════════════════════════════════════════════════════════════════════
# CLEAN CORE NUMERIC FIELDS
# ══════════════════════════════════════════════════════════════════════════

print("\n── Cleaning numeric fields ───────────────────────────────────")

int_cols = ["positive", "negative", "required_age", "achievements",
            "average_playtime", "median_playtime",
            "average_playtime_2w", "median_playtime_2w",
            "peak_ccu", "dlc_count", "recommendations",
            "metacritic_score", "user_score"]
for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    else:
        df[col] = 0

df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════
# FILTER: REQUIRE AT LEAST ONE REVIEW
# ══════════════════════════════════════════════════════════════════════════

n0 = len(df)
df = df[(df["positive"] + df["negative"]) > 0].copy()
print(f"Dropped {n0 - len(df):,} games with zero reviews  →  {len(df):,} remaining")


# ══════════════════════════════════════════════════════════════════════════
# WILSON SCORE + RATING CATEGORY
# ══════════════════════════════════════════════════════════════════════════

print("\n── Computing Wilson scores ───────────────────────────────────")
df["wilson_score"] = df.apply(
    lambda r: wilson_score(r["positive"], r["negative"]), axis=1
)
df["rating_category"] = df["wilson_score"].apply(categorize_wilson)
print(df["rating_category"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════
# OWNERS → LOG MIDPOINT  +  FILTER LOW-OWNER GAMES
# ══════════════════════════════════════════════════════════════════════════

print("\n── Converting estimated_owners → owners_log ──────────────────")
df["owners_log"] = df["estimated_owners"].apply(owners_to_log)

n1 = len(df)
df = df[df["owners_log"].notna()].copy()
# Same threshold as v1: log(10,000) ≈ 9.21
MIN_OWNERS_LOG = np.log(10_000)
df = df[df["owners_log"] >= MIN_OWNERS_LOG].copy()
print(f"Dropped {n1 - len(df):,} low-owner / unparseable rows  →  {len(df):,} remaining")


# ══════════════════════════════════════════════════════════════════════════
# RELEASE DATE
# ══════════════════════════════════════════════════════════════════════════

df["release_date"]  = pd.to_datetime(df.get("release_date"), errors="coerce")
df["release_year"]  = df["release_date"].dt.year.fillna(0).astype(int)
df["release_month"] = df["release_date"].dt.month.fillna(0).astype(int)


# ══════════════════════════════════════════════════════════════════════════
# ENGLISH FLAG
# ══════════════════════════════════════════════════════════════════════════

def _has_english(val):
    if isinstance(val, list):
        return int(any("english" in str(v).lower() for v in val))
    return int("english" in str(val).lower()) if pd.notna(val) else 0

df["english"] = df.get("supported_languages", pd.Series("", index=df.index)).apply(_has_english)


# ══════════════════════════════════════════════════════════════════════════
# DEVELOPER / PUBLISHER  (first entry from list)
# ══════════════════════════════════════════════════════════════════════════

df["developer"] = df["developers"].apply(first_of_list) if "developers" in df.columns else ""
df["publisher"]  = df["publishers"].apply(first_of_list) if "publishers" in df.columns else ""


# ══════════════════════════════════════════════════════════════════════════
# WEBSITE / SUPPORT → BINARY FLAGS
# (v1 stored label-encoded ints; v2 uses 0/1 since raw URLs are available)
# ══════════════════════════════════════════════════════════════════════════

def _has_value(x):
    return 0 if (pd.isna(x) or str(x).strip() in {"", "nan", "None", "0"}) else 1

for col in ["website", "support_url", "support_email"]:
    if col in df.columns:
        df[col] = df[col].apply(_has_value)
    else:
        df[col] = 0


# ══════════════════════════════════════════════════════════════════════════
# PLATFORM FLAGS
# ══════════════════════════════════════════════════════════════════════════

for col in ["windows", "mac", "linux"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: int(bool(x)))
    else:
        df[col] = 0

df["internet_required"] = 0   # not available in this dataset
df["windows_sup"] = df["windows"]
df["mac_sup"]     = df["mac"]
df["linux_sup"]   = df["linux"]


# ══════════════════════════════════════════════════════════════════════════
# GENRES → ONE-HOT  (_genre suffix, same as v1)
# ══════════════════════════════════════════════════════════════════════════

print("\n── One-hot encoding genres ───────────────────────────────────")
genres_lists = df["genres"].apply(parse_list_field) if "genres" in df.columns \
               else pd.Series([[] for _ in range(len(df))], index=df.index)

mlb_genre  = MultiLabelBinarizer()
genre_mat  = mlb_genre.fit_transform(genres_lists)
genre_df   = pd.DataFrame(
    genre_mat,
    columns=[f"{g}_genre" for g in mlb_genre.classes_],
    index=df.index,
)
print(f"  {len(mlb_genre.classes_)} unique genres → {len(genre_df.columns)} columns")


# ══════════════════════════════════════════════════════════════════════════
# CATEGORIES → ONE-HOT  (_cat suffix, same as v1)
# ══════════════════════════════════════════════════════════════════════════

print("\n── One-hot encoding categories ───────────────────────────────")
cat_lists = df["categories"].apply(parse_list_field) if "categories" in df.columns \
            else pd.Series([[] for _ in range(len(df))], index=df.index)

mlb_cat = MultiLabelBinarizer()
cat_mat = mlb_cat.fit_transform(cat_lists)
cat_df  = pd.DataFrame(
    cat_mat,
    columns=[f"{c}_cat" for c in mlb_cat.classes_],
    index=df.index,
)
print(f"  {len(mlb_cat.classes_)} unique categories → {len(cat_df.columns)} columns")


# ══════════════════════════════════════════════════════════════════════════
# TAGS → ONE-HOT  (normalised names, no suffix — same as v1)
# ══════════════════════════════════════════════════════════════════════════

print("\n── One-hot encoding tags ─────────────────────────────────────")
tags_lists = df["tags"].apply(parse_tags_field) if "tags" in df.columns \
             else pd.Series([[] for _ in range(len(df))], index=df.index)

# Normalise names to match v1 (e.g. "Action RPG" → "action_rpg")
tags_lists = tags_lists.apply(lambda tags: [norm_tag(t) for t in tags if t])

# Keep only tags that appear in ≥ 50 games to reduce sparsity
all_tags   = list(itertools.chain.from_iterable(tags_lists))
tag_counts = Counter(all_tags)
common_tags = {t for t, c in tag_counts.items() if c >= 50 and t}
tags_lists  = tags_lists.apply(lambda tags: [t for t in tags if t in common_tags])

mlb_tag = MultiLabelBinarizer()
tag_mat = mlb_tag.fit_transform(tags_lists)
tag_df  = pd.DataFrame(
    tag_mat,
    columns=list(mlb_tag.classes_),
    index=df.index,
)
print(f"  {len(mlb_tag.classes_)} tags appearing in ≥50 games → {len(tag_df.columns)} columns")


# ══════════════════════════════════════════════════════════════════════════
# ASSEMBLE FINAL DATAFRAME
# ══════════════════════════════════════════════════════════════════════════

print("\n── Assembling final dataframe ────────────────────────────────")

core_cols = [
    # targets (kept for ML pipeline, not used as features)
    "wilson_score", "rating_category",
    # metadata
    "english", "developer", "publisher",
    "required_age", "achievements",
    "average_playtime", "median_playtime",
    "price",
    "website", "support_url", "support_email",
    "release_year", "release_month",
    # ownership
    "owners_log",
    # platform
    "windows", "mac", "linux", "internet_required",
    "windows_sup", "mac_sup", "linux_sup",
    # NEW vs v1 (hardware requirements not available in this source)
    "peak_ccu", "dlc_count", "recommendations",
    "average_playtime_2w", "median_playtime_2w",
    "metacritic_score",
]
core_cols = [c for c in core_cols if c in df.columns]

df_final = pd.concat([df[core_cols], genre_df, cat_df, tag_df], axis=1)

# Remove any column name collisions between genre/cat/tag
df_final = df_final.loc[:, ~df_final.columns.duplicated()]

# Drop duplicate appids (using index which maps back to original df)
n_before = len(df_final)
df_final = df_final[~df.loc[df_final.index, "appid"].duplicated(keep="first")]
print(f"  Dropped {n_before - len(df_final):,} duplicate appids")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY + SAVE
# ══════════════════════════════════════════════════════════════════════════

print("\n── Final dataset summary ─────────────────────────────────────")
print(f"  Shape  : {df_final.shape}")
print(f"  Columns: {df_final.shape[1]}")
print(f"\n  Rating distribution:")
print(df_final["rating_category"].value_counts().to_string())
print(f"\n  Wilson score:")
print(df_final["wilson_score"].describe().round(4).to_string())
print(f"\n  owners_log:")
print(df_final["owners_log"].describe().round(4).to_string())

# Column groups
g_cols   = [c for c in df_final.columns if c.endswith("_genre")]
c_cols   = [c for c in df_final.columns if c.endswith("_cat")]
t_cols   = [c for c in df_final.columns if c in mlb_tag.classes_]
print(f"\n  Genre columns  : {len(g_cols)}")
print(f"  Category cols  : {len(c_cols)}")
print(f"  Tag columns    : {len(t_cols)}")

missing = df_final.isnull().sum()
if missing.any():
    print(f"\n  Columns with NaN:")
    print(missing[missing > 0].to_string())

OUT_FILE.parent.mkdir(exist_ok=True)
df_final.to_csv(OUT_FILE, index=False)
print(f"\nSaved → {OUT_FILE}")


# ══════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCES  (RandomForestRegressor on wilson_score, same method
# as cleaning_dim_reduction.ipynb)
# ══════════════════════════════════════════════════════════════════════════

print("\n── Computing feature importances ─────────────────────────────")
from sklearn.ensemble import RandomForestRegressor

DROP_FOR_FEATURES = {"wilson_score", "rating_category"}
feature_cols = [c for c in df_final.columns if c not in DROP_FOR_FEATURES]

X_imp = df_final[feature_cols].copy()
y_imp = df_final["wilson_score"].values

# Factorize any remaining string columns (developer, publisher)
for col in X_imp.select_dtypes(include="object").columns:
    X_imp[col] = pd.factorize(X_imp[col])[0]

X_imp = X_imp.fillna(-1)

print(f"  Fitting RandomForest on {X_imp.shape[0]:,} rows × {X_imp.shape[1]} features...")
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_imp, y_imp)

imp_df = (
    pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

IMP_FILE = ROOT / "cleaned_dataset" / "feature_importances_2.csv"
imp_df.to_csv(IMP_FILE, index=False)
print(f"  Top 15 features:")
print(imp_df.head(15).to_string(index=False))
print(f"\nSaved → {IMP_FILE}")
print("Done.")
