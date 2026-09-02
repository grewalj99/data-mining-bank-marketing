from sklearn.model_selection import train_test_split

from src.evaluate import compute_metrics, evaluate_model
from src.train import train_random_forest

RF_CONFIG = {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 2, "random_state": 42}

METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "auc"}


def test_compute_metrics_returns_expected_keys_and_ranges(processed_df):
    y_true = processed_df["y"].values
    y_pred = y_true.copy()
    y_proba = y_true.astype(float)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert set(metrics.keys()) == METRIC_KEYS
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_compute_metrics_perfect_predictions_score_one(processed_df):
    y_true = processed_df["y"].values
    metrics = compute_metrics(y_true, y_true, y_true.astype(float))

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc"] == 1.0


def test_evaluate_model_matches_compute_metrics(processed_df):
    X = processed_df.drop(columns=["y"])
    y = processed_df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = train_random_forest(X_train, y_train, RF_CONFIG)
    metrics = evaluate_model(clf, X_test, y_test)

    expected = compute_metrics(y_test, clf.predict(X_test), clf.predict_proba(X_test)[:, 1])
    assert metrics == expected
