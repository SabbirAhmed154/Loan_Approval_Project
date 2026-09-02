from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

DATA_FOLDER = PROJECT_FOLDER / "data"

MODEL_FOLDER = PROJECT_FOLDER / "models"

MODEL_FOLDER.mkdir(
    exist_ok=True
)


DATA_FILE = DATA_FOLDER / "loan_data_cleaned.csv"

TARGET_COLUMN = "loan_status"


# ==========================================
# LOAD CLEAN DATASET
# ==========================================

if not DATA_FILE.exists():

    print("Clean dataset not found.")

    print(
        "Run 02_data_cleaning.py first."
    )

    raise SystemExit


try:

    df = pd.read_csv(
        DATA_FILE
    )

except Exception as error:

    print(
        "Error loading dataset:"
    )

    print(error)

    raise SystemExit


print("\n==============================")
print("DATA PREPROCESSING")
print("==============================")


print(
    "\nDataset Shape:",
    df.shape
)


# ==========================================
# TARGET COLUMN CHECK
# ==========================================

if TARGET_COLUMN not in df.columns:

    print(
        "\nERROR: loan_status column not found."
    )

    print(
        "\nAvailable Columns:"
    )

    print(
        df.columns.tolist()
    )

    raise SystemExit


# ==========================================
# CLEAN TARGET VALUES
# ==========================================

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)


print("\nOriginal Target Values:")

print(
    df[TARGET_COLUMN]
    .value_counts(
        dropna=False
    )
)


# ==========================================
# CONVERT TARGET TO 0 AND 1
# ==========================================

target_mapping = {

    # Approved
    "approved": 1,
    "approve": 1,
    "yes": 1,
    "y": 1,
    "1": 1,

    # Rejected
    "rejected": 0,
    "reject": 0,
    "no": 0,
    "n": 0,
    "0": 0
}


df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .map(target_mapping)
)


# Remove unknown target values
unknown_target = (
    df[TARGET_COLUMN]
    .isnull()
    .sum()
)


if unknown_target > 0:

    print(
        f"\nRemoving {unknown_target} rows "
        "with unknown loan status."
    )

    df = df.dropna(
        subset=[TARGET_COLUMN]
    )


df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(int)
)


print("\nConverted Target Values:")

print(
    df[TARGET_COLUMN]
    .value_counts()
)


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=[TARGET_COLUMN]
)

y = df[TARGET_COLUMN]


print("\nFeature Shape:")

print(
    X.shape
)


print("\nTarget Shape:")

print(
    y.shape
)


# ==========================================
# DETECT NUMERICAL COLUMNS
# ==========================================

numerical_columns = (
    X.select_dtypes(
        include="number"
    )
    .columns
    .tolist()
)


print("\nNumerical Columns:")

print(
    numerical_columns
)


# ==========================================
# DETECT CATEGORICAL COLUMNS
# ==========================================

categorical_columns = (
    X.select_dtypes(
        exclude="number"
    )
    .columns
    .tolist()
)


print("\nCategorical Columns:")

print(
    categorical_columns
)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================")


print(
    "\nX Train:",
    X_train.shape
)

print(
    "X Test:",
    X_test.shape
)

print(
    "Y Train:",
    y_train.shape
)

print(
    "Y Test:",
    y_test.shape
)


# ==========================================
# NUMERICAL PREPROCESSOR
# ==========================================

numerical_transformer = StandardScaler()


# ==========================================
# CATEGORICAL PREPROCESSOR
# ==========================================

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)


# ==========================================
# COLUMN TRANSFORMER
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",
            numerical_transformer,
            numerical_columns
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_columns
        )

    ]

)


# ==========================================
# FIT PREPROCESSOR
# ==========================================

print("\nFitting preprocessor...")


preprocessor.fit(
    X_train
)


print(
    "Preprocessor fitted successfully."
)


# ==========================================
# TRANSFORM DATA
# ==========================================

X_train_processed = (
    preprocessor.transform(
        X_train
    )
)


X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


print("\nProcessed Training Shape:")

print(
    X_train_processed.shape
)


print("\nProcessed Testing Shape:")

print(
    X_test_processed.shape
)


# ==========================================
# SAVE PREPROCESSOR
# ==========================================

PREPROCESSOR_FILE = (
    MODEL_FOLDER /
    "preprocessor.joblib"
)


joblib.dump(
    preprocessor,
    PREPROCESSOR_FILE
)


print(
    "\nPreprocessor saved:"
)

print(
    PREPROCESSOR_FILE
)


# ==========================================
# SAVE TRAIN TEST DATA
# ==========================================

X_train.to_csv(
    DATA_FOLDER / "X_train.csv",
    index=False
)


X_test.to_csv(
    DATA_FOLDER / "X_test.csv",
    index=False
)


y_train.to_csv(
    DATA_FOLDER / "y_train.csv",
    index=False
)


y_test.to_csv(
    DATA_FOLDER / "y_test.csv",
    index=False
)


print("\nTrain/Test files saved.")


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n==============================")
print("PREPROCESSING COMPLETE")
print("==============================")


print(
    "\nReady for model training."
)