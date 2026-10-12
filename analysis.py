"""
Customer Churn Analysis
------------------------
End-to-end analysis: data cleaning, EDA, cohort-style churn breakdown,
and churn prediction modeling using the modular src package.

Run: python3 analysis.py
Outputs: charts saved to images/, model metrics printed to console.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.preprocess import clean_data, prepare_data
from src.train import train_and_evaluate

# Configure visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
Path("images").mkdir(exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------------
raw_df = pd.read_csv("data/customer_data.csv")
print(f"Raw shape: {raw_df.shape}")
print("\nMissing values before cleaning:\n", raw_df.isnull().sum()[raw_df.isnull().sum() > 0])

df = clean_data(raw_df)
print(f"\nCleaned shape: {df.shape}")

# ---------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
churn_rate = (df["Churn"] == "Yes").mean() * 100
print(f"\nOverall churn rate: {churn_rate:.1f}%")

# --- Chart 1: Overall churn distribution
plt.figure(figsize=(5, 5))
df["Churn"].value_counts().plot.pie(
    autopct="%1.1f%%",
    colors=["#4C72B0", "#DD8452"],
    startangle=90,
    explode=(0, 0.05),
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
sns.barplot(x=contract_churn.index, y=contract_churn.values, palette="Blues_r", hue=contract_churn.index, legend=False)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Contract Type")
for i, v in enumerate(contract_churn.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.ylim(0, max(contract_churn.values) + 10)
plt.tight_layout()
plt.savefig("images/02_churn_by_contract.png")
plt.close()

# --- Chart 3: Tenure distribution by churn status
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, palette={"No": "#4C72B0", "Yes": "#DD8452"})
plt.title("Customer Tenure Distribution by Churn Status")
plt.xlabel("Tenure (months)")
plt.tight_layout()
plt.savefig("images/03_tenure_distribution.png")
plt.close()

# --- Chart 4: Monthly charges vs churn
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette={"No": "#4C72B0", "Yes": "#DD8452"}, hue="Churn", legend=False)
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.savefig("images/04_monthly_charges_vs_churn.png")
plt.close()

# --- Chart 5: Churn by internet service
plt.figure(figsize=(7, 5))
internet_churn = df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
internet_churn = internet_churn.sort_values(ascending=False)
sns.barplot(x=internet_churn.index, y=internet_churn.values, palette="Oranges_r", hue=internet_churn.index, legend=False)
plt.title("Churn Rate by Internet Service Type")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Internet Service")
for i, v in enumerate(internet_churn.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.ylim(0, max(internet_churn.values) + 10)
plt.tight_layout()
plt.savefig("images/05_churn_by_internet_service.png")
plt.close()

# --- Chart 6: Tenure cohort retention
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
)
cohort_churn = df.groupby("tenure_group", observed=True)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
plt.figure(figsize=(7, 5))
sns.barplot(x=cohort_churn.index, y=cohort_churn.values, palette="viridis", hue=cohort_churn.index, legend=False)
plt.title("Churn Rate by Tenure Cohort (months)")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Tenure Group (months)")
for i, v in enumerate(cohort_churn.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.ylim(0, max(cohort_churn.values) + 10)
plt.tight_layout()
plt.savefig("images/06_churn_by_tenure_cohort.png")
plt.close()

print("\nAll EDA charts saved to images/")

# ---------------------------------------------------------------
# 3. CHURN PREDICTION MODELING & EVALUATION
# ---------------------------------------------------------------
best_pipeline, metrics_data = train_and_evaluate(
    data_path="data/customer_data.csv",
    output_dir="models",
    random_state=42,
)

X_train, X_test, y_train, y_test = prepare_data(
    filepath="data/customer_data.csv",
    test_size=0.2,
    random_state=42,
)

# Test predictions for visualization
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

# --- Chart 7: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)
plt.title(f"Confusion Matrix - {metrics_data['best_model']}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("images/07_confusion_matrix.png")
plt.close()

# --- Chart 8: ROC curve comparison
fpr_best, tpr_best, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(
    fpr_best,
    tpr_best,
    label=f"{metrics_data['best_model']} (AUC={metrics_data['best_metrics']['roc_auc']:.2f})",
    color="#4C72B0",
    lw=2,
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC=0.50)")
plt.title("ROC Curve - Tuned Pipeline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/08_roc_curve.png")
plt.close()

# --- Chart 9: Top feature importances
if metrics_data.get("top_features"):
    top_feats = metrics_data["top_features"]
    feat_names = list(top_feats.keys())
    feat_values = list(top_feats.values())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=feat_values, y=feat_names, palette="mako", hue=feat_names, legend=False)
    plt.title(f"Top 10 Feature Importances ({metrics_data['best_model']})")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig("images/09_feature_importance.png")
    plt.close()

print("\nModel charts saved to images/")
print("\nDone. See README.md for the full write-up.")
