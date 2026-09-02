
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

DATA_FOLDER = PROJECT_FOLDER / "data"
MODEL_FOLDER = PROJECT_FOLDER / "models"
OUTPUT_FOLDER = PROJECT_FOLDER / "evaluation_outputs"


OUTPUT_FOLDER.mkdir(
    exist_ok=True
)


# ==========================================
# FILE PATHS
# ==========================================

X_TEST_FILE = DATA_FOLDER / "X_test.csv"
Y_TEST_FILE = DATA_FOLDER / "y_test.csv"

PREPROCESSOR_FILE = MODEL_FOLDER / "preprocessor.joblib"
BEST_MODEL_FILE = MODEL_FOLDER / "best_model.joblib"
BEST_MODEL_NAME_FILE = MODEL_FOLDER / "best_model_name.txt"


# ==========================================
# CHECK FILES
# ==========================================

required_files = [
    X_TEST_FILE,
    Y_TEST_FILE,
    PREPROCESSOR_FILE,
    BEST_MODEL_FILE
]


for file in required_files:

    if not file.exists():

        print(
            f"Missing file: {file}"
        )

        print(
            "Run previous steps first."
        )

        raise SystemExit


# ==========================================
# LOAD TEST DATA
# ==========================================

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")


X_test = pd.read_csv(
    X_TEST_FILE
)


y_test = pd.read_csv(
    Y_TEST_FILE
).iloc[:, 0]


print(
    "\nX Test Shape:",
    X_test.shape
)


print(
    "Y Test Shape:",
    y_test.shape
)


# ==========================================
# LOAD PREPROCESSOR
# ==========================================

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


print(
    "\nPreprocessor loaded."
)


# ==========================================
# LOAD BEST MODEL
# ==========================================

best_model = joblib.load(
    BEST_MODEL_FILE
)


print(
    "Best model loaded."
)


# ==========================================
# LOAD BEST MODEL NAME
# ==========================================

if BEST_MODEL_NAME_FILE.exists():

    with open(
        BEST_MODEL_NAME_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        best_model_name = file.read().strip()

else:

    best_model_name = "Best Model"


print(
    "\nModel:",
    best_model_name
)


# ==========================================
# TRANSFORM TEST DATA
# ==========================================

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


print(
    "\nTest data transformed."
)


# ==========================================
# PREDICTION
# ==========================================

y_pred = best_model.predict(
    X_test_processed
)


print(
    "\nPrediction complete."
)


# ==========================================
# METRICS
# ==========================================

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


print("\n==============================")
print("EVALUATION RESULTS")
print("==============================")


print(
    f"\nAccuracy  : {accuracy:.4f}"
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


print(
    f"\nAccuracy Percentage: "
    f"{accuracy * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")


report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Rejected",
        "Approved"
    ],
    zero_division=0
)


print(
    report
)


# Save classification report

REPORT_FILE = (
    OUTPUT_FOLDER /
    "classification_report.txt"
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"Model: {best_model_name}\n\n"
    )

    file.write(
        report
    )


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")


print(
    cm
)


# ==========================================
# CONFUSION MATRIX GRAPH
# ==========================================

plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    cm
)


plt.title(
    f"Confusion Matrix - {best_model_name}"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.xticks(
    [0, 1],
    [
        "Rejected",
        "Approved"
    ]
)


plt.yticks(
    [0, 1],
    [
        "Rejected",
        "Approved"
    ]
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


plt.savefig(
    OUTPUT_FOLDER /
    "confusion_matrix.png"
)


plt.show()

plt.close()


# ==========================================
# METRICS GRAPH
# ==========================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


metric_values = [
    accuracy,
    precision,
    recall,
    f1
]


plt.figure(
    figsize=(8, 5)
)


plt.bar(
    metric_names,
    metric_values
)


plt.title(
    f"Model Performance - {best_model_name}"
)


plt.ylabel(
    "Score"
)


plt.ylim(
    0,
    1
)


for index, value in enumerate(
    metric_values
):

    plt.text(
        index,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )


plt.tight_layout()


plt.savefig(
    OUTPUT_FOLDER /
    "model_metrics.png"
)


plt.show()

plt.close()


# ==========================================
# SAVE PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": y_pred
    }
)


prediction_df.to_csv(
    OUTPUT_FOLDER /
    "test_predictions.csv",
    index=False
)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n==============================")
print("MODEL EVALUATION COMPLETE")
print("==============================")


print(
    "\nEvaluation files saved inside:"
)

print(
    OUTPUT_FOLDER
)