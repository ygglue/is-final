"""Unsupervised Learning module: K-Means Customer Segmentation with Elbow & PCA."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data_pipeline import CLUSTER_FEATURES


def _assign_persona_label(row: pd.Series, spend_median: float, freq_median: float, days_median: float) -> str:
    """Assign a human-readable retail persona to a cluster centroid."""
    high_spend = row["annual_spend"] >= spend_median
    high_freq = row["purchase_frequency"] >= freq_median
    recent = row["days_since_last_visit"] <= days_median

    if high_spend and high_freq and recent:
        return "VIP Loyalists"
    if high_spend and not high_freq:
        return "High-Ticket Occasional"
    if not recent and row["days_since_last_visit"] > days_median * 1.3:
        return "Dormant / Churn Risk"
    if high_freq and not high_spend:
        return "Frequent Budget Shoppers"
    return "Steady Core Buyers"


def run_customer_segmentation(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> dict:
    """Run StandardScaler + K-Means clustering, Elbow curve (k=2..8), Silhouette score, and 2D PCA."""
    n_clusters = int(np.clip(n_clusters, 2, 8))
    X = df[CLUSTER_FEATURES].copy()

    # Step 1: Standardize features so Euclidean distance treats all scales fairly
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Fit K-Means with k-means++ initialization
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = float(silhouette_score(X_scaled, labels))

    # Step 3: Compute Elbow Curve & Silhouette across k=2..8
    elbow_data = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        k_labels = km.fit_predict(X_scaled)
        elbow_data.append(
            {
                "k": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette": round(float(silhouette_score(X_scaled, k_labels)), 4),
            }
        )

    # Step 4: Project 4D scaled features to 2D via PCA for interactive scatter visualization
    pca = PCA(n_components=2, random_state=random_state)
    coords_2d = pca.fit_transform(X_scaled)
    centroids_2d = pca.transform(kmeans.cluster_centers_)

    # Step 5: Summarize cluster personas in original feature units
    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    spend_med = float(df["annual_spend"].median())
    freq_med = float(df["purchase_frequency"].median())
    days_med = float(df["days_since_last_visit"].median())

    cluster_profiles = []
    used_names = {}
    for cid in range(n_clusters):
        subset = clustered_df[clustered_df["cluster"] == cid]
        means = subset[CLUSTER_FEATURES].mean()
        base_name = _assign_persona_label(means, spend_med, freq_med, days_med)
        count_seen = used_names.get(base_name, 0)
        used_names[base_name] = count_seen + 1
        persona = base_name if count_seen == 0 else f"{base_name} #{count_seen + 1}"

        cluster_profiles.append(
            {
                "cluster_id": cid,
                "persona": persona,
                "count": int(len(subset)),
                "pct": round(len(subset) / len(df) * 100, 1),
                "avg_annual_spend": round(float(means["annual_spend"]), 2),
                "avg_purchase_frequency": round(float(means["purchase_frequency"]), 1),
                "avg_cart_value": round(float(means["avg_cart_value"]), 2),
                "avg_days_since_visit": round(float(means["days_since_last_visit"]), 1),
                "churn_rate": round(float(subset["churned"].mean()) * 100, 1) if "churned" in subset else 0.0,
            }
        )

    # Sample up to 350 points for smooth browser rendering while keeping all clusters represented
    sample_idx = np.linspace(0, len(df) - 1, min(350, len(df)), dtype=int)
    scatter_points = [
        {
            "customer_id": str(df.iloc[i]["customer_id"]),
            "pca_x": round(float(coords_2d[i, 0]), 3),
            "pca_y": round(float(coords_2d[i, 1]), 3),
            "cluster": int(labels[i]),
            "annual_spend": round(float(df.iloc[i]["annual_spend"]), 2),
            "purchase_frequency": int(df.iloc[i]["purchase_frequency"]),
            "days_since_last_visit": int(df.iloc[i]["days_since_last_visit"]),
        }
        for i in sample_idx
    ]

    centroid_points = [
        {
            "cluster": cid,
            "pca_x": round(float(centroids_2d[cid, 0]), 3),
            "pca_y": round(float(centroids_2d[cid, 1]), 3),
        }
        for cid in range(n_clusters)
    ]

    return {
        "n_clusters": n_clusters,
        "inertia": round(float(kmeans.inertia_), 2),
        "silhouette_score": round(sil_score, 4),
        "pca_explained_variance": [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_],
        "elbow_curve": elbow_data,
        "profiles": cluster_profiles,
        "scatter_points": scatter_points,
        "centroids": centroid_points,
    }
