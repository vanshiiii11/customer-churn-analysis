"""
Training Module for Customer Churn Prediction.
Trains Logistic Regression and Random Forest models with imbalance handling,
tunes Random Forest via GridSearchCV, evaluates honest out-of-sample metrics,
picks the best pipeline by ROC-AUC, and serializes the winning pipeline and metrics.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocess import build_full_pipeline, prepare_data


def evaluate_pipeline(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics on unseen test data."""
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
    }
    return metrics


def train_and_evaluate(
    data_path: str = "data/customer_data.csv",
    output_dir: str = "models",
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Main training orchestration function:
    1. Prepares clean stratified train/test split (no leakage).
    2. Trains Logistic Regression baseline (class_weight='balanced').
    3. Performs GridSearchCV tuning on Random Forest (class_weight='balanced').
    4. Evaluates both models honestly on the test set.
    5. Saves best full pipeline to models/churn_model.joblib.
    6. Saves metrics report to models/metrics.json.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading data and creating stratified train/test split...")
    X_train, X_test, y_train, y_test = prepare_data(
        filepath=data_path,
        test_size=0.2,
        random_state=random_state,
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")

    results: Dict[str, Any] = {
        "dataset": {
            "total_samples": int(len(X_train) + len(X_test)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_churn_rate": round(float(y_test.mean()), 4),
        },
        "imbalance_handling": "class_weight='balanced'",
        "random_state": random_state,
        "models": {},
    }

    # 1. Logistic Regression
    print("\nTraining Logistic Regression (class_weight='balanced')...")
    lr_base = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    lr_pipe = build_full_pipeline(lr_base)
    lr_pipe.fit(X_train, y_train)

    lr_metrics = evaluate_pipeline(lr_pipe, X_test, y_test)
    results["models"]["LogisticRegression"] = {
        "metrics": lr_metrics,
        "parameters": {"max_iter": 1000, "class_weight": "balanced"},
    }
    print(f"Logistic Regression Test Metrics: {lr_metrics}")

    # 2. Random Forest with GridSearchCV Tuning
    print("\nTuning Random Forest with GridSearchCV (class_weight='balanced')...")
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
    )
    rf_pipe = build_full_pipeline(rf_base)

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 8, 12],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [2, 4],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=rf_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_rf_pipe: Pipeline = grid_search.best_estimator_
    clean_best_params = {
        k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()
    }
    rf_metrics = evaluate_pipeline(best_rf_pipe, X_test, y_test)

    results["models"]["RandomForestClassifier"] = {
        "metrics": rf_metrics,
        "best_cv_roc_auc": round(float(grid_search.best_score_), 4),
        "best_parameters": clean_best_params,
    }
    print(f"Best RF Parameters: {clean_best_params}")
    print(f"Random Forest Test Metrics: {rf_metrics}")

    # 3. Model Selection
    lr_auc = results["models"]["LogisticRegression"]["metrics"]["roc_auc"]
    rf_auc = results["models"]["RandomForestClassifier"]["metrics"]["roc_auc"]

    if rf_auc >= lr_auc:
        best_name = "RandomForestClassifier"
        best_pipeline = best_rf_pipe
        best_metrics = rf_metrics
    else:
        best_name = "LogisticRegression"
        best_pipeline = lr_pipe
        best_metrics = lr_metrics

    print(f"\nWinning Model by Test ROC-AUC: {best_name} (ROC-AUC: {best_metrics['roc_auc']})")

    # Extract Feature Importances if available
    try:
        classifier_step = best_pipeline.named_steps["classifier"]
        preprocessor_step = best_pipeline.named_steps["preprocessor"]
        feature_names = preprocessor_step.get_feature_names_out()

        if hasattr(classifier_step, "feature_importances_"):
            importances = classifier_step.feature_importances_
        elif hasattr(classifier_step, "coef_"):
            importances = np.abs(classifier_step.coef_[0])
        else:
            importances = None

        if importances is not None:
            feat_imp_pairs = sorted(
                [
                    (str(name).replace("num__", "").replace("cat__", ""), round(float(imp), 4))
                    for name, imp in zip(feature_names, importances)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            results["top_features"] = dict(feat_imp_pairs[:10])
    except Exception as exc:
        print(f"Could not compute feature importances: {exc}")
        results["top_features"] = {}

    results["best_model"] = best_name
    results["best_metrics"] = best_metrics

    # Save artifacts
    model_save_path = Path(output_dir) / "churn_model.joblib"
    metrics_save_path = Path(output_dir) / "metrics.json"

    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best pipeline to: {model_save_path}")

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_save_path}")

    return best_pipeline, results


if __name__ == "__main__":
    train_and_evaluate()
