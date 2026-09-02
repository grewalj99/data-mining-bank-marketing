import numpy as np
import pytest

from src.features import select_features_l1


def test_select_features_l1_mask_matches_feature_count(processed_df):
    X = processed_df.drop(columns=["y"])
    y = processed_df["y"]

    mask, model = select_features_l1(X, y, C=0.1, random_state=42)

    assert mask.dtype == bool
    assert len(mask) == X.shape[1]
    assert model.coef_.shape[1] == X.shape[1]


def test_select_features_l1_stronger_C_selects_at_least_as_many_features(processed_df):
    X = processed_df.drop(columns=["y"])
    y = processed_df["y"]

    strict_mask, _ = select_features_l1(X, y, C=0.1, random_state=42)
    lenient_mask, _ = select_features_l1(X, y, C=10.0, random_state=42)

    # Stronger regularization (smaller C) should never select more features
    # than weaker regularization (larger C).
    assert np.sum(strict_mask) <= np.sum(lenient_mask)


def test_select_features_l1_raises_when_everything_is_zeroed_out(processed_df):
    X = processed_df.drop(columns=["y"])
    y = processed_df["y"]

    with pytest.raises(ValueError, match="zeroed out every coefficient"):
        select_features_l1(X, y, C=1e-6, random_state=42)
