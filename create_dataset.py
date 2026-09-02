import numpy as np
import pandas as pd


np.random.seed(42)

total_rows = 1000


# ==========================================
# CREATE SYNTHETIC CUSTOMER DATA
# ==========================================

income = np.random.randint(
    25000,
    150001,
    total_rows
)

loan_amount = np.random.randint(
    5000,
    80001,
    total_rows
)

credit_score = np.random.randint(
    300,
    851,
    total_rows
)

employment_years = np.random.randint(
    0,
    31,
    total_rows
)

existing_debt = np.random.randint(
    0,
    50001,
    total_rows
)

loan_term = np.random.choice(
    [12, 24, 36, 48, 60],
    total_rows
)


# ==========================================
# CALCULATE RATIOS
# ==========================================

debt_ratio = (
    existing_debt / income
)

loan_ratio = (
    loan_amount / income
)


# ==========================================
# DEMO LOAN APPROVAL RULE
# ==========================================

loan_status = (

    (credit_score >= 600)

    & (debt_ratio < 0.50)

    & (loan_ratio < 0.75)

    & (employment_years >= 1)

).astype(int)


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(
    {
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "existing_debt": existing_debt,
        "loan_term": loan_term,
        "loan_status": loan_status,
    }
)


# ==========================================
# SAVE DATASET
# ==========================================

df.to_csv(
    "data/loan_data.csv",
    index=False
)


print("Dataset created successfully.")

print()

print("Dataset Shape:")
print(df.shape)

print()

print("Loan Status Count:")
print(
    df["loan_status"].value_counts()
)