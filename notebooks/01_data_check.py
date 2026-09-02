from pathlib import Path

import pandas as pd


# ==========================================
# DATA FOLDER
# ==========================================

DATA_FOLDER = Path("data")


# ==========================================
# FIND CSV FILE
# ==========================================

csv_files = list(
    DATA_FOLDER.glob("*.csv")
)

if not csv_files:

    print(
        "No CSV file found inside data folder."
    )

    raise SystemExit


# First CSV file
csv_path = csv_files[0]

print(
    "Dataset:",
    csv_path.name
)


# ==========================================
# LOAD DATA
# ==========================================

try:

    df = pd.read_csv(
        csv_path
    )

except Exception as error:

    print(
        "Error loading dataset:"
    )

    print(error)

    raise SystemExit


# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ==========================================
# BASIC INFORMATION
# ==========================================

print("\n==============================")
print("DATASET BASIC CHECK")
print("==============================")


# Shape
print("\nShape:")
print(df.shape)


# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())


# Column names
print("\nColumns:")
print(
    df.columns.tolist()
)


# Data types
print("\nData Types:")
print(
    df.dtypes
)


# ==========================================
# MISSING VALUES
# ==========================================

print("\nMissing Values:")

print(
    df.isnull().sum()
)


# ==========================================
# DUPLICATE ROWS
# ==========================================

print("\nDuplicate Rows:")

print(
    df.duplicated().sum()
)


# ==========================================
# TARGET COLUMN CHECK
# ==========================================

if "loan_status" in df.columns:

    print("\nLoan Status Count:")

    print(
        df["loan_status"]
        .value_counts(
            dropna=False
        )
    )


    print("\nLoan Status Percentage:")

    print(
        (
            df["loan_status"]
            .value_counts(
                normalize=True,
                dropna=False
            )
            * 100
        ).round(2)
    )

else:

    print(
        "\nloan_status column not found."
    )

    print(
        "Available columns:"
    )

    print(
        df.columns.tolist()
    )


# ==========================================
# DATASET SUMMARY
# ==========================================

print("\nNumerical Summary:")

print(
    df.describe()
)


# ==========================================
# UNIQUE VALUES
# ==========================================

print("\n==============================")
print("UNIQUE VALUES")
print("==============================")

for column in df.columns:

    print(
        f"\n{column}:"
    )

    print(
        df[column].unique()
    )


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n==============================")
print("DATA CHECK COMPLETE")
print("==============================")