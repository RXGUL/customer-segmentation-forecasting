"""
clustering.py  –  K-Means clustering on RFM features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
from pathlib import Path


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)):
    """Elbow + Silhouette analysis to pick optimal K."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    return list(k_range), inertias, silhouettes


def fit_kmeans(rfm: pd.DataFrame, n_clusters: int = 4,
               model_dir: str | Path = None):
    """
    Fit K-Means on log-transformed RFM features.
    Returns (rfm_with_cluster, scaler, model).
    """
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].copy()

    # Log-transform to reduce skew
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"]  = np.log1p(X["Monetary"])
    X["Recency"]   = np.log1p(X["Recency"])

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_scaled)

    rfm = rfm.copy()
    rfm["Cluster"] = km.labels_

    sil  = silhouette_score(X_scaled, km.labels_)
    db   = davies_bouldin_score(X_scaled, km.labels_)
    print(f"  K-Means (k={n_clusters}): Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")

    # Save model artefacts
    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(km,     model_dir / "kmeans_model.pkl")
        joblib.dump(scaler, model_dir / "scaler.pkl")
        print(f"  Models saved to {model_dir}")

    return rfm, scaler, km, X_scaled


def cluster_profiles(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean RFM metrics per cluster + segment label."""
    profile = (
        rfm_clustered.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
        .reset_index()
    )
    profile = profile.sort_values("Monetary", ascending=False).reset_index(drop=True)

    def label(row):
        if row["Monetary"] > profile["Monetary"].median() and row["Recency"] < profile["Recency"].median():
            return "High-Value Active"
        elif row["Monetary"] > profile["Monetary"].median():
            return "High-Value Inactive"
        elif row["Recency"] < profile["Recency"].median():
            return "Low-Value Active"
        else:
            return "Low-Value Inactive"

    profile["Cluster_Label"] = profile.apply(label, axis=1)
    return profile
