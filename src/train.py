from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train, y_train)


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, rf_config: dict
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=rf_config["n_estimators"],
        max_depth=rf_config["max_depth"],
        min_samples_leaf=rf_config["min_samples_leaf"],
        random_state=rf_config["random_state"],
    )
    clf.fit(X_train, y_train)
    return clf


def run_hyperparameter_search(
    X: pd.DataFrame, y: pd.Series, search_config: dict
) -> RandomizedSearchCV:
    rf = RandomForestClassifier(random_state=search_config["random_state"])
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=search_config["param_distributions"],
        n_iter=search_config["n_iter"],
        scoring=search_config["scoring"],
        cv=search_config["cv"],
        random_state=search_config["random_state"],
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def train_all_variants(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_mask: np.ndarray,
    config: dict,
) -> dict[str, RandomForestClassifier]:
    """Train the four Random Forest variants compared in the README:
    baseline, +SMOTE, +feature selection, +feature selection +SMOTE.
    """
    rf_config = config["random_forest"]
    smote_seed = config["smote"]["random_state"]

    variants: dict[str, RandomForestClassifier] = {}

    variants["baseline"] = train_random_forest(X_train, y_train, rf_config)

    X_smote, y_smote = apply_smote(X_train, y_train, smote_seed)
    variants["smote"] = train_random_forest(X_smote, y_smote, rf_config)

    X_train_fs = X_train.loc[:, feature_mask]
    variants["feature_selection"] = train_random_forest(X_train_fs, y_train, rf_config)

    X_fs_smote, y_fs_smote = apply_smote(X_train_fs, y_train, smote_seed)
    variants["feature_selection_smote"] = train_random_forest(X_fs_smote, y_fs_smote, rf_config)

    return variants
