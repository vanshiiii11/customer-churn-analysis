"""
Customer Churn Preprocessing and Feature Engineering Module.
Contains reusable data cleaning, transformation, and feature engineering utilities.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
    "monthly_to_total_ratio",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_cohort",
]

RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer churn data:
    - Removes duplicate customerID rows (keeps first).
    - Converts TotalCharges, MonthlyCharges, and tenure to numeric.
    - Imputes missing TotalCharges with estimated tenure * MonthlyCharges.
    - Fills any remaining NaNs with appropriate fallbacks.
    """
    df = df.copy()

    # Deduplicate if customerID present
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"], keep="first")

    # Numeric conversion & imputation
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Impute missing TotalCharges with tenure * MonthlyCharges
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            estimated_total = df["tenure"] * df["MonthlyCharges"]
            df["TotalCharges"] = df["TotalCharges"].fillna(estimated_total)
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int).astype(str)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-specific engineered features:
    - tenure_cohort: categorical duration bucket
    - num_services: count of subscribed services
    - monthly_to_total_ratio: monthly charge relative to total spend
    - is_month_to_month: binary indicator for flexible contract
    - is_fiber: binary indicator for fiber optic connection
    """
    df = df.copy()

    # Tenure cohort bucketing
    if "tenure" in df.columns:
        tenure_series = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        df["tenure_cohort"] = pd.cut(
            tenure_series,
            bins=[-1, 12, 24, 36, 48, 60, 120],
            labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72+"],
        ).astype(str)
    else:
        df["tenure_cohort"] = "0-12"

    # Count of active subscribed services
    service_indicators = []
    if "PhoneService" in df.columns:
        service_indicators.append((df["PhoneService"] == "Yes").astype(int))
    if "MultipleLines" in df.columns:
        service_indicators.append((df["MultipleLines"] == "Yes").astype(int))
    if "InternetService" in df.columns:
        service_indicators.append(df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int))
    if "OnlineSecurity" in df.columns:
        service_indicators.append((df["OnlineSecurity"] == "Yes").astype(int))
    if "TechSupport" in df.columns:
        service_indicators.append((df["TechSupport"] == "Yes").astype(int))

    if service_indicators:
        df["num_services"] = sum(service_indicators)
    else:
        df["num_services"] = 0

    # Monthly to total charges ratio
    if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
        mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        df["monthly_to_total_ratio"] = mc / (tc.replace(0, 1.0) + 1.0)
    else:
        df["monthly_to_total_ratio"] = 0.0

    if "Contract" in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    else:
        df["is_month_to_month"] = 0

    if "InternetService" in df.columns:
        df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    else:
        df["is_fiber"] = 0

    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer wrapper around engineer_features.
    Ensures safe integration inside a full scikit-learn Pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = clean_data(X)
        return engineer_features(cleaned)


def build_column_transformer(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Constructs an sklearn ColumnTransformer with StandardScaler for numerics
    and OneHotEncoder for categoricals. Includes imputation to handle edge cases.
    """
    num_cols = numeric_features or NUMERIC_FEATURES
    cat_cols = categorical_features or CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(
    classifier,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Pipeline:
    """
    Constructs an end-to-end sklearn Pipeline:
    [FeatureEngineer -> ColumnTransformer -> Classifier]
    Takes raw customer DataFrames directly and performs all transformations internally.
    """
    col_transformer = build_column_transformer(numeric_features, categorical_features)
    pipeline = Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("preprocessor", col_transformer),
            ("classifier", classifier),
        ]
    )
    return pipeline


def load_data(filepath: str = "data/customer_data.csv") -> pd.DataFrame:
    """Loads raw dataset from CSV."""
    return pd.read_csv(filepath)


def prepare_data(
    filepath: str = "data/customer_data.csv",
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw data, performs initial deduplication and basic cleaning,
    separates features and target, and splits into train and test sets
    with stratification to prevent train/test leakage.
    """
    df = load_data(filepath)
    df = clean_data(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = (df[target_col] == "Yes").astype(int)
    X = df.drop(columns=[target_col])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
