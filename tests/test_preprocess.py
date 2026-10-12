"""
Tests for data cleaning, preprocessing, and feature engineering.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocess import (
    clean_data,
    engineer_features,
    FeatureEngineer,
    build_column_transformer,
    build_full_pipeline,
    prepare_data,
)
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "customerID": ["CUST1", "CUST2", "CUST1"],  # includes duplicate
            "gender": ["Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No"],
            "tenure": [12, 2, 12],
            "PhoneService": ["Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "DSL"],
            "OnlineSecurity": ["Yes", "No", "Yes"],
            "TechSupport": ["No", "No", "No"],
            "Contract": ["One year", "Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Credit card", "Electronic check"],
            "MonthlyCharges": [50.0, 80.0, 50.0],
            "TotalCharges": [600.0, np.nan, 600.0],  # includes missing
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_clean_data_deduplication(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    assert len(cleaned) == 2
    assert list(cleaned["customerID"]) == ["CUST1", "CUST2"]


def test_clean_data_missing_total_charges_imputation(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    # CUST2 has tenure=2, MonthlyCharges=80, TotalCharges was NaN -> should be imputed to 160.0
    cust2 = cleaned[cleaned["customerID"] == "CUST2"].iloc[0]
    assert cust2["TotalCharges"] == pytest.approx(160.0)


def test_engineer_features(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    featured = engineer_features(cleaned)

    assert "tenure_cohort" in featured.columns
    assert "num_services" in featured.columns
    assert "monthly_to_total_ratio" in featured.columns
    assert "is_month_to_month" in featured.columns
    assert "is_fiber" in featured.columns

    cust2 = featured[featured["customerID"] == "CUST2"].iloc[0]
    assert cust2["tenure_cohort"] == "0-12"
    assert cust2["is_month_to_month"] == 1
    assert cust2["is_fiber"] == 1


def test_feature_engineer_transformer(sample_raw_df):
    transformer = FeatureEngineer()
    res = transformer.transform(sample_raw_df)
    assert isinstance(res, pd.DataFrame)
    assert "num_services" in res.columns


def test_build_full_pipeline_fit_predict(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    X = cleaned.drop(columns=["customerID", "Churn"])
    y = (cleaned["Churn"] == "Yes").astype(int)

    pipeline = build_full_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=42)
    )
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_prepare_data_splits_and_leakage():
    X_train, X_test, y_train, y_test = prepare_data(
        filepath="data/customer_data.csv",
        test_size=0.2,
        random_state=42,
    )
    assert len(X_train) == 1600
    assert len(X_test) == 400
    assert "customerID" not in X_train.columns
    assert "Churn" not in X_train.columns
    assert len(y_train) == 1600
    assert len(y_test) == 400
    # Stratification check
    assert abs(y_train.mean() - y_test.mean()) < 0.02
