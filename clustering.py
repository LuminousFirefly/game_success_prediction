from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score 
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import gower


df = pd.read_csv("steam_finalized_dataset.csv")

# Stratified sample: keep e.g. 3000 rows, preserving class proportions
df_sample, _ = train_test_split(
    df,
    train_size=5000,
    stratify=df["rating_category"],
    random_state=42
)

# Verify the proportions match
print("Full dataset:\n", df["rating_category"].value_counts(normalize=True))
print("\nSample:\n", df_sample["rating_category"].value_counts(normalize=True))


numeric_cols = [
    "required_age", "achievements", "average_playtime", "median_playtime", 
    "price", "website", "support_url", "support_email", "processor_Ghz", "RAM_mb", "GPU_mb", 
    "storage_mb", "owners_log"
]

genre_cols = [c for c in df.columns if c.endswith("_genre")]
cat_cols = [c for c in df.columns if c.endswith("_cat")]

# Boolean/binary flags -- treat as numeric (0/1), Gower handles them fine
binary_cols = [
    "english", "internet_required", "linux", "mac", "windows"
] + genre_cols + cat_cols

# Categorical -- must be object or category dtype
categorical_cols = [
    "release_month", "developer", "publisher"
]


tag_cols = [c for c in df.columns if not c.endswith(("_genre", "_cat", "_sup"))
            and df[c].dtype in [float, int]
            and c not in numeric_cols + binary_cols]

top_tags = df[tag_cols].sum().nlargest(15).index.tolist()


feature_cols = numeric_cols + binary_cols + categorical_cols #+ top_tags

X = df_sample[feature_cols].copy()


cols_with_sentinel = ["storage_mb", "website", "support_email", "support_url"]
X[cols_with_sentinel] = X[cols_with_sentinel].replace(-1, np.nan)


# Ensure dtypes
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Fill NaNs -- Gower can handle NaN but it's safer to be explicit
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# Returns an (n x n) float32 matrix, values in [0, 1]
D = gower.gower_matrix(X.values)

# Sanity checks
print("Shape:", D.shape)
print("Min:", D.min(), "Max:", D.max())  # should be 0.0 to 1.0
print("Symmetric:", np.allclose(D, D.T))


sil_scores = {}
all_labels = {}



for k in range(2, 20):
    kmed = KMedoids(
            n_clusters=k,
            metric="precomputed",
            init="k-medoids++",
            random_state=42
        )
    labels = kmed.fit_predict(D)
    all_labels[k] = labels.copy()
    score = silhouette_score(D, labels, metric="precomputed")
    sil_scores[k] = score
    print(f"k={k}  silhouette={score:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\nBest k: {best_k}")

plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
plt.xlabel("k")
plt.ylabel("Mean silhouette score")
plt.title("Silhouette scores by k")
plt.xticks(range(2, 11))

# plt.show()

kmed_final = KMedoids(
    n_clusters=best_k,
    metric="precomputed",
    init="k-medoids++",
    random_state=42
)

kmed_final.fit(D)

best_labels = all_labels[best_k]
df_sample = df_sample.copy()
df_sample["cluster"] = kmed_final.labels_

# How many games in each cluster
print(df_sample["cluster"].value_counts())

# Cluster assignments alongside game names and rating
print(df_sample[["name", "rating_category", "cluster"]].head(20))

def wcpd(D, labels):
    total = 0
    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        sub = D[np.ix_(idx, idx)]
        total += sub[np.triu_indices(len(idx), k=1)].sum()
    return total

wcpd_score = wcpd(D, best_labels)
print(f"WCPD: {wcpd_score:.4f}")

sil_final = silhouette_score(D, best_labels, metric="precomputed")
print(f"Silhouette (k={best_k}): {sil_final:.4f}")

le = LabelEncoder()
true_labels = le.fit_transform(df_sample["rating_category"])

# Print the mapping
for i, cls in enumerate(le.classes_):
    print(f"{i} -> {cls}")


ari = adjusted_rand_score(true_labels, best_labels)
print(f"ARI: {ari:.4f}")

print(df_sample.groupby("cluster")["rating_category"].value_counts(normalize=True).round(3))
print(df_sample.groupby("cluster")["achievements"].mean())