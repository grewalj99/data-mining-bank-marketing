from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_raw, resolve_path


def clean_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the dataset's 'unknown' string placeholders with real NaNs."""
    df_clean = df.copy()
    categorical_cols = df_clean.select_dtypes(include="object").columns.tolist()
    df_clean[categorical_cols] = df_clean[categorical_cols].replace("unknown", np.nan)
    return df_clean


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=numeric_impute_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def preprocess(
    df: pd.DataFrame,
    leakage_columns: list[str] | None = None,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Clean, impute, encode, and scale the raw dataset.

    `leakage_columns` (e.g. `duration`) are dropped before fitting since
    they aren't known until after a call has ended and would leak the
    outcome into the model.
    """
    leakage_columns = leakage_columns or []
    df_clean = clean_placeholders(df)

    X = df_clean.drop(columns=["y"])
    X_model = X.drop(columns=[c for c in leakage_columns if c in X.columns])
    y = df_clean["y"].map({"no": 0, "yes": 1})

    numeric_features = X_model.select_dtypes(include="number").columns.tolist()
    categorical_features = X_model.select_dtypes(include="object").columns.tolist()

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        numeric_impute_strategy,
        categorical_impute_strategy,
    )
    X_processed_array = preprocessor.fit_transform(X_model)
    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed_array, columns=feature_names, index=df_clean.index)
    processed_df = X_processed.assign(y=y.values)
    return processed_df, preprocessor


def run_preprocessing(config: dict) -> tuple[Path, pd.DataFrame]:
    """Load the raw CSV, preprocess it, and write the result to disk.

    This is the function the CLI (`python -m src.pipeline`) calls, and
    it's the same function the EDA notebook calls — there's a single
    implementation of the preprocessing logic.
    """
    raw_df = load_raw(config["data"]["raw_path"])
    processed_df, _ = preprocess(
        raw_df,
        leakage_columns=config["preprocessing"].get("leakage_columns", []),
        numeric_impute_strategy=config["preprocessing"].get("numeric_impute_strategy", "median"),
        categorical_impute_strategy=config["preprocessing"].get(
            "categorical_impute_strategy", "most_frequent"
        ),
    )
    processed_path = resolve_path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    return processed_path, processed_df
