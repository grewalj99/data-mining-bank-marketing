from __future__ import annotations

import argparse
import json

from sklearn.model_selection import train_test_split

from src.clustering import find_best_k, run_kmeans
from src.config import load_config
from src.data import load_processed
from src.evaluate import evaluate_model
from src.features import select_features_l1
from src.outliers import detect_outliers
from src.preprocessing import run_preprocessing
from src.train import apply_smote, run_hyperparameter_search, train_all_variants

STAGES = ["preprocess", "cluster", "outliers", "train", "tune"]


def run(config_path: str, stages: list[str]) -> dict:
    config = load_config(config_path)
    results: dict = {}

    if "preprocess" in stages:
        processed_path, _ = run_preprocessing(config)
        results["preprocess"] = {"processed_path": str(processed_path)}

    df = load_processed(config["data"]["processed_path"])
    y = df["y"]
    X = df.drop(columns=["y"])

    if "cluster" in stages:
        cluster_cfg = config["clustering"]
        best_k, scores = find_best_k(
            X, cluster_cfg["k_min"], cluster_cfg["k_max"], cluster_cfg["random_state"]
        )
        run_kmeans(X, best_k, cluster_cfg["random_state"])
        results["cluster"] = {"best_k": best_k, "silhouette_by_k": scores}

    if "outliers" in stages:
        outlier_cfg = config["outliers"]
        _, labels = detect_outliers(X, outlier_cfg["contamination"], outlier_cfg["random_state"])
        n_outliers = int((labels == -1).sum())
        results["outliers"] = {
            "n_outliers": n_outliers,
            "outlier_pct": 100 * n_outliers / len(labels),
        }

    if "train" in stages or "tune" in stages:
        split_cfg = config["split"]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["random_state"],
            stratify=y,
        )

        fs_cfg = config["feature_selection"]
        feature_mask, _ = select_features_l1(X_train, y_train, fs_cfg["C"], fs_cfg["random_state"])

        variants = train_all_variants(X_train, y_train, feature_mask, config)

        metrics = {}
        for name, model in variants.items():
            X_eval = X_test.loc[:, feature_mask] if "feature_selection" in name else X_test
            metrics[name] = evaluate_model(model, X_eval, y_test)
        results["train"] = metrics

        if "tune" in stages:
            X_train_fs = X_train.loc[:, feature_mask]
            X_fs_smote, y_fs_smote = apply_smote(
                X_train_fs, y_train, config["smote"]["random_state"]
            )
            search = run_hyperparameter_search(
                X_fs_smote, y_fs_smote, config["hyperparameter_search"]
            )
            best_metrics = evaluate_model(
                search.best_estimator_, X_test.loc[:, feature_mask], y_test
            )
            results["tune"] = {"best_params": search.best_params_, "metrics": best_metrics}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bank marketing ML pipeline end to end.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=STAGES,
        choices=STAGES,
        help="Which pipeline stages to run (default: all)",
    )
    args = parser.parse_args()

    results = run(args.config, args.stages)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
