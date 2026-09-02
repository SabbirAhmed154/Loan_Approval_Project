from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

DATA_FOLDER = PROJECT_FOLDER / "data"
MODEL_FOLDER = PROJECT_FOLDER / "models"

MODEL_FOLDER.mkdir(
    exist_ok=True
)


# ==========================================
# FILE PATHS
# ==========================================

X_TRAIN_FILE = DATA_FOLDER / "X_train.csv"
X_TEST_FILE = DATA_FOLDER / "X_test.csv"

Y_TRAIN_FILE = DATA_FOLDER / "y_train.csv"
Y_TEST_FILE = DATA_FOLDER / "y_test.csv"

PREPROCESSOR_FILE = MODEL_FOLDER / "preprocessor.joblib"


# ==========================================
# CHECK FILES
# ==========================================

required_files = [
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    PREPROCESSOR_FILE
]


for file in required_files:

    if not file.exists():

        print(
            f"Missing file: {file}"
        )

        print(
            "Run 04_preprocessing.py first."
        )

        raise SystemExit


# ==========================================
# LOAD TRAIN TEST DATA
# ==========================================

print("\n==============================")
print("MODEL TRAINING")
print("==============================")


X_train = pd.read_csv(
    X_TRAIN_FILE
)

X_test = pd.read_csv(
    X_TEST_FILE
)

y_train = pd.read_csv(
    Y_TRAIN_FILE
).iloc[:, 0]

y_test = pd.read_csv(
    Y_TEST_FILE
).iloc[:, 0]


print("\nTraining Data:")

print(
    X_train.shape
)


print("\nTesting Data:")

print(
    X_test.shape
)


# ==========================================
# LOAD PREPROCESSOR
# ==========================================

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


print(
    "\nPreprocessor loaded successfully."
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


print(
    "\nTraining data transformed."
)

print(
    "Processed Train Shape:",
    X_train_processed.shape
)

print(
    "Processed Test Shape:",
    X_test_processed.shape
)


# ==========================================
# CREATE MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


# ==========================================
# STORE RESULTS
# ==========================================

results = []

trained_models = {}


# ==========================================
# TRAIN MODELS
# ==========================================

print("\n==============================")
print("TRAINING MODELS")
print("==============================")


for model_name, model in models.items():

    print(
        f"\nTraining: {model_name}"
    )


    # Train
    model.fit(
        X_train_processed,
        y_train
    )


    # Prediction
    y_pred = model.predict(
        X_test_processed
    )


    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    # Save result
    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    )


    trained_models[
        model_name
    ] = model


# ==========================================
# MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        by="Accuracy",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")


print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# FIND BEST MODEL
# ==========================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_accuracy = (
    results_df.iloc[0]["Accuracy"]
)


best_model = trained_models[
    best_model_name
]


print("\n==============================")
print("BEST MODEL")
print("==============================")


print(
    "Model:",
    best_model_name
)


print(
    f"Accuracy: {best_accuracy:.4f}"
)


print(
    f"Accuracy Percentage: "
    f"{best_accuracy * 100:.2f}%"
)


# ==========================================
# SAVE ALL MODELS
# ==========================================

joblib.dump(
    trained_models["Logistic Regression"],
    MODEL_FOLDER / "logistic_regression.joblib"
)


joblib.dump(
    trained_models["Decision Tree"],
    MODEL_FOLDER / "decision_tree.joblib"
)


joblib.dump(
    trained_models["Random Forest"],
    MODEL_FOLDER / "random_forest.joblib"
)


print(
    "\nAll models saved."
)


# ==========================================
# SAVE BEST MODEL
# ==========================================

BEST_MODEL_FILE = (
    MODEL_FOLDER / "best_model.joblib"
)


joblib.dump(
    best_model,
    BEST_MODEL_FILE
)


print(
    "\nBest model saved:"
)

print(
    BEST_MODEL_FILE
)


# ==========================================
# SAVE MODEL NAME
# ==========================================

BEST_MODEL_NAME_FILE = (
    MODEL_FOLDER /
    "best_model_name.txt"
)


with open(
    BEST_MODEL_NAME_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        str(best_model_name)
    )


# ==========================================
# SAVE RESULTS
# ==========================================

RESULT_FILE = (
    MODEL_FOLDER /
    "model_comparison.csv"
)


results_df.to_csv(
    RESULT_FILE,
    index=False
)


print(
    "\nModel comparison saved:"
)

print(
    RESULT_FILE
)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n==============================")
print("MODEL TRAINING COMPLETE")
print("==============================")


print(
    "\nReady for model evaluation."
)