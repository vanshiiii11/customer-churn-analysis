"""
Generates a realistic (synthetic) telecom-style customer churn dataset.
Run this once to create data/customer_data.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

customer_id = [f"CUST{1000+i}" for i in range(n)]
gender = np.random.choice(["Male", "Female"], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.85, 0.15])
partner = np.random.choice(["Yes", "No"], n)
dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
tenure = np.random.randint(0, 72, n)
phone_service = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
multiple_lines = np.random.choice(["Yes", "No", "No phone service"], n, p=[0.4, 0.5, 0.1])
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
online_security = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.3, 0.5, 0.2])
tech_support = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.3, 0.5, 0.2])
contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
)
monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
total_charges = np.round(monthly_charges * tenure + np.random.uniform(0, 100, n), 2)

# churn probability influenced by contract type, tenure, monthly charges
churn_prob = (
    0.5 * (contract == "Month-to-month")
    + 0.1 * (contract == "One year")
    + 0.02 * (contract == "Two year")
    + 0.3 * (tenure < 12)
    + 0.2 * (monthly_charges > 80)
    - 0.15 * (dependents == "Yes")
)
churn_prob = np.clip(churn_prob, 0.02, 0.9)
churn = np.array(["Yes" if np.random.rand() < p else "No" for p in churn_prob])

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn
})

# introduce some messiness (real-world realism: nulls + duplicates)
missing_idx = np.random.choice(df.index, 30, replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan

dup_rows = df.sample(15, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("data/customer_data.csv", index=False)
print("Dataset created:", df.shape)
