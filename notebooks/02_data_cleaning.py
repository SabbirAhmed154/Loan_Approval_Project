from pathlib import Path

import pandas as pd


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

DATA_FOLDER = PROJECT_FOLDER / "data"

OUTPUT_FILE = DATA_FOLDER / "loan_data_cleaned.csv"


# ==========================================
# FIND RAW CSV FILE
# ==========================================

csv_files = [
    file
    for file in DATA_FOLDER.glob("*.csv")
    if file.name != "loan_data_cleaned.csv"
]


if not csv_files:

    print("No raw CSV file found inside data folder.")

    raise SystemExit


csv_path = csv_files[0]


print("\n==============================")
print("LOAN DATA CLEANING")
print("==============================")


print(
    "\nDataset:",
    csv_path.name
)


# ==========================================
# LOAD DATASET
# ==========================================

try:

    df = pd.read_csv(
        csv_path
    )

except Exception as error:

    print("\nError loading dataset:")

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


print("\nColumns:")

print(
    df.columns.tolist()
)


# ==========================================
# BEFORE CLEANING
# ==========================================

print("\n==============================")
print("BEFORE CLEANING")
print("==============================")


print("\nShape:")

print(
    df.shape
)


print("\nMissing Values:")

print(
    df.isnull().sum()
)


print("\nDuplicate Rows:")

print(
    df.duplicated().sum()
)


# ==========================================
# CLEAN TEXT VALUES
# ==========================================

text_columns = df.select_dtypes(
    include="object"
).columns


for column in text_columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )


print("\nText values cleaned.")


# ==========================================
# REMOVE DUPLICATE ROWS
# ==========================================

duplicate_count = df.duplicated().sum()


df = df.drop_duplicates().copy()


print(
    "\nDuplicates Removed:",
    duplicate_count
)


# ==========================================
# REMOVE UNNECESSARY ID COLUMNS
# ==========================================

id_columns = [
    "loan_id",
    "id"
]


for column in id_columns:

    if column in df.columns:

        df = df.drop(
            columns=[column]
        )

        print(
            f"Removed ID column: {column}"
        )


# ==========================================
# TARGET COLUMN CHECK
# ==========================================

TARGET_COLUMN = "loan_status"


if TARGET_COLUMN not in df.columns:

    print(
        "\nERROR: loan_status column not found."
    )

    print(
        "\nAvailable columns:"
    )

    print(
        df.columns.tolist()
    )

    raise SystemExit


# ==========================================
# REMOVE MISSING TARGET VALUES
# ==========================================

missing_target = df[TARGET_COLUMN].isnull().sum()


if missing_target > 0:

    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    print(
        f"\nRemoved {missing_target} rows "
        "with missing loan_status."
    )


# ==========================================
# HANDLE MISSING NUMERICAL VALUES
# ==========================================

print("\n==============================")
print("HANDLING MISSING VALUES")
print("==============================")


numeric_columns = df.select_dtypes(
    include="number"
).columns


for column in numeric_columns:

    missing_count = df[column].isnull().sum()

    if missing_count > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )

        print(
            f"{column}: "
            f"{missing_count} missing values "
            f"filled with median {median_value}"
        )


# ==========================================
# HANDLE MISSING CATEGORICAL VALUES
# ==========================================

categorical_columns = df.select_dtypes(
    include="object"
).columns


for column in categorical_columns:

    if column == TARGET_COLUMN:
        continue

    missing_count = df[column].isnull().sum()

    if missing_count > 0:

        mode_values = df[column].mode()

        if not mode_values.empty:

            mode_value = mode_values.iloc[0]

            df[column] = df[column].fillna(
                mode_value
            )

            print(
                f"{column}: "
                f"{missing_count} missing values "
                f"filled with mode '{mode_value}'"
            )


# ==========================================
# CLEAN TARGET VALUES
# ==========================================

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)


print("\nLoan Status Values:")

print(
    df[TARGET_COLUMN].value_counts()
)


# ==========================================
# LOAN STATUS PERCENTAGE
# ==========================================

print("\nLoan Status Percentage:")

print(
    (
        df[TARGET_COLUMN]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ==========================================
# AFTER CLEANING
# ==========================================

print("\n==============================")
print("AFTER CLEANING")
print("==============================")


print("\nShape:")

print(
    df.shape
)


print("\nMissing Values:")

print(
    df.isnull().sum()
)


print("\nDuplicate Rows:")

print(
    df.duplicated().sum()
)


print("\nData Types:")

print(
    df.dtypes
)


# ==========================================
# SAVE CLEAN DATASET
# ==========================================

try:

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

except Exception as error:

    print(
        "\nError saving cleaned dataset:"
    )

    print(error)

    raise SystemExit


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n==============================")
print("DATA CLEANING COMPLETE")
print("==============================")


print(
    "\nClean dataset saved at:"
)

print(
    OUTPUT_FILE
)