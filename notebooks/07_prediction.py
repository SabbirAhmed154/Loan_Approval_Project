from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

DATA_FOLDER = PROJECT_FOLDER / "data"
MODEL_FOLDER = PROJECT_FOLDER / "models"


# ==========================================
# FILE PATHS
# ==========================================

X_TRAIN_FILE = DATA_FOLDER / "X_train.csv"

MODEL_FILE = MODEL_FOLDER / "best_model.joblib"

PREPROCESSOR_FILE = (
    MODEL_FOLDER /
    "preprocessor.joblib"
)

MODEL_NAME_FILE = (
    MODEL_FOLDER /
    "best_model_name.txt"
)


# ==========================================
# CHECK REQUIRED FILES
# ==========================================

required_files = [
    X_TRAIN_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE
]


for file in required_files:

    if not file.exists():

        print(
            f"Missing file: {file}"
        )

        print(
            "Run previous project steps first."
        )

        raise SystemExit


# ==========================================
# LOAD DATA
# ==========================================

X_train = pd.read_csv(
    X_TRAIN_FILE
)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    MODEL_FILE
)


# ==========================================
# LOAD PREPROCESSOR
# ==========================================

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


# ==========================================
# LOAD MODEL NAME
# ==========================================

if MODEL_NAME_FILE.exists():

    with open(
        MODEL_NAME_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        model_name = file.read().strip()

else:

    model_name = "Best Model"


# ==========================================
# START
# ==========================================

print("\n====================================")
print("LOAN APPROVAL PREDICTION SYSTEM")
print("====================================")


print(
    "\nModel:",
    model_name
)


print(
    "\nPlease enter applicant information."
)


# ==========================================
# IDENTIFY COLUMN TYPES
# ==========================================

numerical_columns = (
    X_train
    .select_dtypes(
        include="number"
    )
    .columns
    .tolist()
)


categorical_columns = (
    X_train
    .select_dtypes(
        exclude="number"
    )
    .columns
    .tolist()
)


# ==========================================
# USER INPUT
# ==========================================

user_data = {}


for column in X_train.columns:

    print(
        "\n------------------------------------"
    )

    print(
        "Feature:",
        column
    )


    # ======================================
    # NUMERICAL INPUT
    # ======================================

    if column in numerical_columns:

        print(
            "Type: Numerical"
        )


        while True:

            value = input(
                f"Enter {column}: "
            )

            try:

                value = float(value)

                user_data[column] = value

                break

            except ValueError:

                print(
                    "Please enter a valid number."
                )


    # ======================================
    # CATEGORICAL INPUT
    # ======================================

    else:

        options = (
            X_train[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )


        print(
            "Available options:"
        )


        for number, option in enumerate(
            options,
            start=1
        ):

            print(
                f"{number}. {option}"
            )


        while True:

            value = input(
                f"Enter {column}: "
            ).strip().lower()


            valid_options = [
                str(option).lower()
                for option in options
            ]


            if value in valid_options:

                user_data[column] = value

                break


            try:

                choice = int(value)

                if (
                    1 <= choice <= len(options)
                ):

                    user_data[column] = (
                        options[
                            choice - 1
                        ]
                    )

                    break

            except ValueError:

                pass


            print(
                "Please enter a valid option."
            )


# ==========================================
# CREATE DATAFRAME
# ==========================================

input_df = pd.DataFrame(
    [user_data]
)


# Keep exact training column order

input_df = input_df[
    X_train.columns
]


print("\n====================================")
print("APPLICANT INFORMATION")
print("====================================")


print(
    input_df.to_string(
        index=False
    )
)


# ==========================================
# PREPROCESS INPUT
# ==========================================

try:

    processed_input = (
        preprocessor.transform(
            input_df
        )
    )

except Exception as error:

    print(
        "\nError preprocessing applicant data:"
    )

    print(error)

    raise SystemExit


# ==========================================
# PREDICT
# ==========================================

try:

    prediction = model.predict(
        processed_input
    )[0]

except Exception as error:

    print(
        "\nPrediction error:"
    )

    print(error)

    raise SystemExit


# ==========================================
# PREDICTION PROBABILITY
# ==========================================

approval_probability = None


if hasattr(
    model,
    "predict_proba"
):

    probabilities = (
        model.predict_proba(
            processed_input
        )[0]
    )


    classes = list(
        model.classes_
    )


    if 1 in classes:

        approved_index = (
            classes.index(1)
        )

        approval_probability = (
            probabilities[
                approved_index
            ]
        )


# ==========================================
# RESULT
# ==========================================

print("\n====================================")
print("PREDICTION RESULT")
print("====================================")


if prediction == 1:

    print(
        "\nLOAN APPROVED"
    )

else:

    print(
        "\nLOAN REJECTED"
    )


if approval_probability is not None:

    print(
        "\nApproval Probability:"
    )

    print(
        f"{approval_probability * 100:.2f}%"
    )


print("\n====================================")
print("PREDICTION COMPLETE")
print("====================================")