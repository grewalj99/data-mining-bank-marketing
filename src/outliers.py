from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_outliers(
    X: pd.DataFrame, contamination: float = 0.05, random_state: int = 42
) -> tuple[IsolationForest, np.ndarray]:
    """Fit Isolation Forest and return (-1 = outlier, 1 = inlier) labels."""
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels
