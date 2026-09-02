from src.clustering import find_best_k, run_kmeans


def test_find_best_k_returns_k_within_range(processed_df):
    X = processed_df.drop(columns=["y"])

    best_k, scores = find_best_k(X, k_min=2, k_max=5, random_state=42)

    assert 2 <= best_k <= 5
    assert set(scores.keys()) == {2, 3, 4, 5}
    assert all(-1 <= score <= 1 for score in scores.values())


def test_run_kmeans_labels_match_row_count(processed_df):
    X = processed_df.drop(columns=["y"])

    _, labels = run_kmeans(X, k=3, random_state=42)

    assert len(labels) == len(X)
    assert set(labels) <= {0, 1, 2}
