import numpy as np

from src.preprocessing import clean_placeholders, preprocess


def test_clean_placeholders_replaces_unknown_with_nan(raw_df):
    assert (raw_df["job"] == "unknown").any()

    cleaned = clean_placeholders(raw_df)

    assert not (cleaned["job"] == "unknown").any()
    assert cleaned["job"].isna().any()
    # numeric columns are untouched
    assert cleaned["age"].equals(raw_df["age"])


def test_preprocess_drops_leakage_column(raw_df):
    processed_df, _ = preprocess(raw_df, leakage_columns=["duration"])

    duration_cols = [c for c in processed_df.columns if "duration" in c]
    assert duration_cols == []


def test_preprocess_has_no_missing_values(raw_df):
    processed_df, _ = preprocess(raw_df, leakage_columns=["duration"])

    assert not processed_df.isna().any().any()


def test_preprocess_maps_target_to_binary(raw_df):
    processed_df, _ = preprocess(raw_df, leakage_columns=["duration"])

    assert set(processed_df["y"].unique()) <= {0, 1}
    assert processed_df["y"].dtype in (np.int64, np.int32, int)


def test_preprocess_row_count_is_preserved(raw_df):
    processed_df, _ = preprocess(raw_df, leakage_columns=["duration"])

    assert len(processed_df) == len(raw_df)


def test_preprocess_one_hot_encodes_categoricals(raw_df):
    processed_df, _ = preprocess(raw_df, leakage_columns=["duration"])

    # 'unknown' is imputed away with the most frequent real category, so the
    # one-hot columns should match the non-'unknown' categories seen in job.
    non_unknown_jobs = raw_df.loc[raw_df["job"] != "unknown", "job"].nunique()
    job_cols = [c for c in processed_df.columns if c.startswith("cat__job_")]
    assert len(job_cols) == non_unknown_jobs
