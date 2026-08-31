from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


def resolve_path(path: str | Path) -> Path:
    """Resolve a path from config.yaml against the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_raw(raw_path: str | Path) -> pd.DataFrame:
    """Load the original semicolon-delimited UCI Bank Marketing CSV."""
    return pd.read_csv(resolve_path(raw_path), sep=";")


def load_processed(processed_path: str | Path) -> pd.DataFrame:
    """Load the preprocessed, comma-delimited dataset written by `run_preprocessing`."""
    return pd.read_csv(resolve_path(processed_path), sep=",")
