from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def select_features_l1(
    X_train: pd.DataFrame, y_train: pd.Series, C: float = 0.1, random_state: int = 42
) -> tuple[np.ndarray, LogisticRegression]:
    """Fit an L1-penalized logistic regression and return a boolean mask of
    the features with a non-zero coefficient, plus the fitted model."""
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    coefs = model.coef_.ravel()
    mask = np.abs(coefs) > 1e-6

    if not mask.any():
        raise ValueError(
            f"L1 feature selection with C={C} zeroed out every coefficient — "
            "no features were selected. Try a larger C."
        )

    return mask, model
