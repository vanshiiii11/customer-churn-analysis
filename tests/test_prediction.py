"""
Tests for single and batch predictions using the saved churn pipeline.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from app import get_risk_category, compute_top_reasons


@pytest.fixture(scope="module")
def loaded_pipeline():
    model_path = Path("models/churn_model.joblib")
    return joblib.load(model_path)


@pytest.fixture
def high_risk_customer():
    return pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0,
                "TotalCharges": 95.0,
            }
        ]
    )


@pytest.fixture
def low_risk_customer():
    return pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "TechSupport": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card",
                "MonthlyCharges": 45.0,
                "TotalCharges": 2700.0,
            }
        ]
    )


def test_single_prediction_outputs(loaded_pipeline, high_risk_customer):
    prob = loaded_pipeline.predict_proba(high_risk_customer)[0, 1]
    pred = loaded_pipeline.predict(high_risk_customer)[0]

    assert 0.0 <= prob <= 1.0
    assert pred in [0, 1]
    assert prob > 0.50, f"Expected high risk customer to have elevated prob, got {prob}"


def test_relative_risk_distinction(loaded_pipeline, high_risk_customer, low_risk_customer):
    high_prob = loaded_pipeline.predict_proba(high_risk_customer)[0, 1]
    low_prob = loaded_pipeline.predict_proba(low_risk_customer)[0, 1]

    assert high_prob > low_prob
    assert (high_prob - low_prob) >= 0.20, "Model should clearly distinguish high and low risk profiles"


def test_missing_totalcharges_handling(loaded_pipeline, high_risk_customer):
    df_missing = high_risk_customer.copy()
    df_missing["TotalCharges"] = np.nan

    # Should not raise exception because FeatureEngineer / clean_data handles NaN
    prob = loaded_pipeline.predict_proba(df_missing)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_batch_prediction(loaded_pipeline, high_risk_customer, low_risk_customer):
    batch = pd.concat([high_risk_customer, low_risk_customer], ignore_index=True)
    probs = loaded_pipeline.predict_proba(batch)[:, 1]

    assert len(probs) == 2
    assert probs[0] > probs[1]


def test_risk_categories_and_reasons(high_risk_customer):
    risk_info = get_risk_category(0.75)
    assert risk_info["label"] == "High Risk"

    reasons = compute_top_reasons(high_risk_customer.iloc[0], 0.75)
    assert isinstance(reasons, list)
    assert len(reasons) >= 1
    # Check that reasons mention month-to-month or tenure or fiber
    joined = " ".join(reasons).lower()
    assert "month-to-month" in joined or "tenure" in joined or "fiber" in joined
