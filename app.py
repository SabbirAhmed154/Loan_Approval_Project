from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parent

DATA_FOLDER = PROJECT_FOLDER / "data"
MODEL_FOLDER = PROJECT_FOLDER / "models"


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
# CHECK FILES
# ==========================================

required_files = [
    X_TRAIN_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE
]


for file in required_files:

    if not file.exists():

        st.error(
            f"Missing file: {file.name}"
        )

        st.stop()


# ==========================================
# LOAD FILES
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    return model, preprocessor


@st.cache_data
def load_training_data():

    return pd.read_csv(
        X_TRAIN_FILE
    )


model, preprocessor = load_model()

X_train = load_training_data()


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

    model_name = "Machine Learning Model"


# ==========================================
# TITLE
# ==========================================

st.title(
    "🏦 Loan Approval Prediction System"
)

st.write(
    "Enter applicant information below to predict "
    "whether the loan may be approved or rejected."
)

st.info(
    f"Current Model: {model_name}"
)


# ==========================================
# COLUMN TYPES
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
# USER INPUT FORM
# ==========================================

user_data = {}


with st.form(
    "loan_prediction_form"
):

    st.subheader(
        "Applicant Information"
    )


    for column in X_train.columns:

        label = (
            column
            .replace("_", " ")
            .title()
        )


        # ==================================
        # NUMERICAL COLUMN
        # ==================================

        if column in numerical_columns:

            column_data = (
                X_train[column]
                .dropna()
            )


            if len(column_data) > 0:

                default_value = float(
                    column_data.median()
                )

            else:

                default_value = 0.0


            user_data[column] = (
                st.number_input(
                    label=label,
                    value=default_value
                )
            )


        # ==================================
        # CATEGORICAL COLUMN
        # ==================================

        elif column in categorical_columns:

            options = (
                X_train[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            if not options:

                options = [
                    "unknown"
                ]


            user_data[column] = (
                st.selectbox(
                    label=label,
                    options=options
                )
            )


    # ======================================
    # SUBMIT BUTTON
    # ======================================

    submit_button = (
        st.form_submit_button(
            "Predict Loan Status"
        )
    )


# ==========================================
# PREDICTION
# ==========================================

if submit_button:

    try:

        # ----------------------------------
        # Convert input to dataframe
        # ----------------------------------

        input_df = pd.DataFrame(
            [user_data]
        )


        # Same column order as training data
        input_df = input_df[
            X_train.columns
        ]


        # ----------------------------------
        # Preprocessing
        # ----------------------------------

        processed_input = (
            preprocessor.transform(
                input_df
            )
        )


        # ----------------------------------
        # Prediction
        # ----------------------------------

        prediction = model.predict(
            processed_input
        )[0]


        # ----------------------------------
        # Probability
        # ----------------------------------

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


        # ==================================
        # SHOW RESULT
        # ==================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        if prediction == 1:

            st.success(
                "✅ Loan Approved"
            )

        else:

            st.error(
                "❌ Loan Rejected"
            )


        # ==================================
        # SHOW PROBABILITY
        # ==================================

        if approval_probability is not None:

            probability_percentage = (
                approval_probability * 100
            )


            st.metric(
                label="Approval Probability",
                value=f"{probability_percentage:.2f}%"
            )


            st.progress(
                int(
                    probability_percentage
                )
            )


        # ==================================
        # SHOW INPUT DATA
        # ==================================

        with st.expander(
            "View Applicant Data"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.exception(
            error
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Loan Approval Prediction System | Machine Learning Portfolio Project"
)