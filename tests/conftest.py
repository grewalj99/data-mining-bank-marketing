import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small synthetic stand-in for the raw bank-additional-full.csv schema:
    a mix of numeric/categorical columns, some 'unknown' placeholders, a
    leakage column (`duration`), and a yes/no target.
    """
    rng = np.random.default_rng(42)
    n = 80

    jobs = rng.choice(["admin.", "technician", "blue-collar", "unknown"], size=n)
    education = rng.choice(["primary", "secondary", "tertiary", "unknown"], size=n)
    default = rng.choice(["yes", "no", "unknown"], size=n, p=[0.1, 0.7, 0.2])
    housing = rng.choice(["yes", "no"], size=n)
    age = rng.integers(18, 90, size=n)
    campaign = rng.integers(1, 10, size=n)

    # y depends (weakly) on age/campaign rather than being pure noise, so
    # feature selection has a real signal to find instead of legitimately
    # zeroing out every coefficient.
    logits = 0.04 * (age - 54) - 0.3 * campaign - 1.0
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)
    y = np.where(y == 1, "yes", "no")

    return pd.DataFrame(
        {
            "age": age,
            "campaign": campaign,
            "duration": rng.integers(0, 3000, size=n),
            "job": jobs,
            "education": education,
            "default": default,
            "housing": housing,
            "y": y,
        }
    )


@pytest.fixture
def processed_df() -> pd.DataFrame:
    """A small synthetic stand-in for the already-preprocessed dataset:
    numeric feature columns plus a binary y column, with enough minority
    samples for SMOTE's default k_neighbors=5 to work.
    """
    rng = np.random.default_rng(42)
    n = 80
    n_features = 5

    X = rng.normal(size=(n, n_features))

    # y depends (weakly) on the first two features rather than being pure
    # noise, so feature selection has a real signal to find instead of
    # legitimately zeroing out every coefficient.
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] - 1.0
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["y"] = y
    return df
