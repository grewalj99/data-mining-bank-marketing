from src import pipeline


def _write_config(tmp_path, raw_csv_path):
    processed_path = tmp_path / "processed.csv"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  raw_path: {raw_csv_path}
  processed_path: {processed_path}

preprocessing:
  leakage_columns: ["duration"]
  numeric_impute_strategy: median
  categorical_impute_strategy: most_frequent

split:
  test_size: 0.2
  random_state: 42

clustering:
  k_min: 2
  k_max: 3
  random_state: 42

outliers:
  contamination: 0.1
  random_state: 42

feature_selection:
  C: 1.0
  random_state: 42

random_forest:
  n_estimators: 20
  max_depth: 5
  min_samples_leaf: 2
  random_state: 42

smote:
  random_state: 42

hyperparameter_search:
  n_iter: 2
  cv: 2
  scoring: f1
  random_state: 42
  param_distributions:
    n_estimators: [10, 20]
    max_depth: [3, 5]
    min_samples_split: [2]
    min_samples_leaf: [1, 2]
    max_features: ["sqrt"]
    bootstrap: [true]
"""
    )
    return config_path


def test_pipeline_runs_all_stages_end_to_end(tmp_path, raw_df):
    raw_csv_path = tmp_path / "raw.csv"
    raw_df.to_csv(raw_csv_path, sep=";", index=False)
    config_path = _write_config(tmp_path, raw_csv_path)

    results = pipeline.run(str(config_path), pipeline.STAGES)

    assert set(results.keys()) == {"preprocess", "cluster", "outliers", "train", "tune"}

    assert results["preprocess"]["processed_path"]
    assert 2 <= results["cluster"]["best_k"] <= 3
    assert results["outliers"]["n_outliers"] >= 0

    assert set(results["train"].keys()) == {
        "baseline",
        "smote",
        "feature_selection",
        "feature_selection_smote",
    }
    for metrics in results["train"].values():
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "auc"}

    assert "best_params" in results["tune"]
    assert "metrics" in results["tune"]


def test_pipeline_respects_stage_selection(tmp_path, raw_df):
    raw_csv_path = tmp_path / "raw.csv"
    raw_df.to_csv(raw_csv_path, sep=";", index=False)
    config_path = _write_config(tmp_path, raw_csv_path)

    results = pipeline.run(str(config_path), ["preprocess", "train"])

    assert set(results.keys()) == {"preprocess", "train"}
