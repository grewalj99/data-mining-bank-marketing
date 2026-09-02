from src.outliers import detect_outliers


def test_detect_outliers_returns_valid_labels(processed_df):
    X = processed_df.drop(columns=["y"])

    _, labels = detect_outliers(X, contamination=0.1, random_state=42)

    assert len(labels) == len(X)
    assert set(labels) <= {-1, 1}


def test_detect_outliers_respects_contamination_rate(processed_df):
    X = processed_df.drop(columns=["y"])

    _, labels = detect_outliers(X, contamination=0.1, random_state=42)

    n_outliers = (labels == -1).sum()
    expected = round(0.1 * len(X))
    # IsolationForest's contamination is a target rate, not an exact count
    assert abs(n_outliers - expected) <= 2
