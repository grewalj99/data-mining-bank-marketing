from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_best_k(
    X: pd.DataFrame, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> tuple[int, dict[int, float]]:
    """Sweep k and return the k with the best silhouette score, plus all scores."""
    ks = range(k_min, k_max + 1)
    scores: dict[int, float] = {}
    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state)
        labels = model.fit_predict(X)
        scores[k] = silhouette_score(X, labels)

    best_k = max(scores, key=scores.get)
    return best_k, scores


def run_kmeans(X: pd.DataFrame, k: int, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels
