from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load pipeline configuration from a YAML file.

    Relative paths are resolved against the project root, so this works
    the same way whether it's called from the repo root or from a
    notebook in `notebooks/`.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)
