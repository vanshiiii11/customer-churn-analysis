"""
Customer Churn Analysis
------------------------
End-to-end analysis: data cleaning, EDA, cohort-style churn breakdown,
and a baseline churn prediction model.

Run: python3 analysis.py
Outputs: charts saved to images/, model metrics printed to console.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("data/customer_data.csv")
print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------
print("\nMissing values before cleaning:\n", df.isnull().sum()[df.isnull().sum() > 0])

# duplicates
dupes = df.duplicated(subset="customerID").sum()
print(f"\nDuplicate customerIDs found: {dupes}")
df = df.drop_duplicates(subset="customerID", keep="first")

# fill missing TotalCharges with tenure * MonthlyCharges estimate
df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

print(f"\nCleaned shape: {df.shape}")

# ---------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------
churn_rate = (df["Churn"] == "Yes").mean() * 100
print(f"\nOverall churn rate: {churn_rate:.1f}%")

# --- Chart 1: Overall churn distribution
plt.figure(figsize=(5, 5))
df["Churn"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"], startangle=90
)
plt.title("Overall Customer Churn Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("images/01_churn_distribution.png")
plt.close()

# --- Chart 2: Churn rate by contract type
plt.figure(figsize=(7, 5))
contract_churn = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
contract_churn = contract_churn.sort_values(ascending=False)
sns.barplot(x=contract_churn.index, y=contract_churn.values, palette="Blues_r")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Contract Type")
for i, v in enumerate(contract_churn.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("images/02_churn_by_contract.png")
plt.close()

# --- Chart 3: Tenure distribution by churn status
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, palette=["#4C72B0", "#DD8452"])
plt.title("Customer Tenure Distribution by Churn Status")
plt.xlabel("Tenure (months)")
plt.tight_layout()
plt.savefig("images/03_tenure_distribution.png")
plt.close()

# --- Chart 4: Monthly charges vs churn
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#4C72B0", "#DD8452"])
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.savefig("images/04_monthly_charges_vs_churn.png")
plt.close()

# --- Chart 5: Churn by internet service
plt.figure(figsize=(7, 5))
internet_churn = df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
internet_churn = internet_churn.sort_values(ascending=False)
sns.barplot(x=internet_churn.index, y=internet_churn.values, palette="Oranges_r")
plt.title("Churn Rate by Internet Service Type")
plt.ylabel("Churn Rate (%)")
for i, v in enumerate(internet_churn.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("images/05_churn_by_internet_service.png")
plt.close()

# --- Chart 6: Tenure cohort retention (bucketed)
df["tenure_group"] = pd.cut(
    df["tenure"], bins=[0, 12, 24, 36, 48, 60, 72],
    labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
)
cohort_churn = df.groupby("tenure_group", observed=True)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
plt.figure(figsize=(7, 5))
sns.barplot(x=cohort_churn.index, y=cohort_churn.values, palette="viridis")
plt.title("Churn Rate by Tenure Cohort (months)")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Tenure Group (months)")
for i, v in enumerate(cohort_churn.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("images/06_churn_by_tenure_cohort.png")
plt.close()

print("\nAll EDA charts saved to images/")

# ---------------------------------------------------------------
# 4. CHURN PREDICTION MODEL (baseline)
# ---------------------------------------------------------------
model_df = df.copy()
target = "Churn"

cat_cols = model_df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c not in ["customerID", target]]

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    le_dict[col] = le

model_df[target] = model_df[target].map({"Yes": 1, "No": 0})
model_df = model_df.drop(columns=["customerID", "tenure_group"])

X = model_df.drop(columns=[target])
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression baseline
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_probs = log_model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression ---")
print(classification_report(y_test, log_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, log_probs):.3f}")

# Random Forest (usually stronger baseline)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print(classification_report(y_test, rf_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.3f}")

# --- Chart 7: Confusion matrix (Random Forest)
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("images/07_confusion_matrix.png")
plt.close()

# --- Chart 8: ROC curve comparison
fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC={roc_auc_score(y_test, log_probs):.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, rf_probs):.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("images/08_roc_curve.png")
plt.close()

# --- Chart 9: Feature importance (Random Forest)
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, palette="mako")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("images/09_feature_importance.png")
plt.close()

print("\nModel charts saved to images/")
print("\nDone. See README.md for the full write-up.")
