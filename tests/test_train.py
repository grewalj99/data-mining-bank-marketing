from src.train import apply_smote, train_all_variants, train_random_forest

RF_CONFIG = {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 2, "random_state": 42}


def test_apply_smote_balances_classes(processed_df):
    X = processed_df.drop(columns=["y"])
    y = processed_df["y"]

    X_res, y_res = apply_smote(X, y, random_state=42)

    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) == len(y_res)


def test_train_random_forest_fits_and_predicts(processed_df):
    X = processed_df.drop(columns=["y"])
    y = processed_df["y"]

    clf = train_random_forest(X, y, RF_CONFIG)
    predictions = clf.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}


def test_train_all_variants_returns_all_four_models(processed_df):
    X = processed_df.drop(columns=["y"])
    y = processed_df["y"]
    feature_mask = X.columns.isin(X.columns[:3])  # keep the first 3 of 5 features

    config = {"random_forest": RF_CONFIG, "smote": {"random_state": 42}}
    variants = train_all_variants(X, y, feature_mask, config)

    assert set(variants.keys()) == {
        "baseline",
        "smote",
        "feature_selection",
        "feature_selection_smote",
    }
    # feature-selected variants should have been trained on 3 columns, not 5
    assert variants["feature_selection"].n_features_in_ == 3
    assert variants["baseline"].n_features_in_ == 5
