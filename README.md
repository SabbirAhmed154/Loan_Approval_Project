# 🏦 Loan Approval Prediction System

A Machine Learning project that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant information.

This project includes data cleaning, exploratory data analysis, preprocessing, model training, model evaluation, real-time prediction, and a Streamlit web application.

---
## 🌐 Live Demo

https://sabbir-loan-approval-v2.streamlit.app
## 📌 Project Overview

The goal of this project is to build a Machine Learning system that can analyze loan applicant information and predict the loan approval status.

The project follows a complete end-to-end Machine Learning workflow:

* Data Loading
* Data Cleaning
* Missing Value Handling
* Exploratory Data Analysis
* Feature Preprocessing
* Train/Test Split
* Model Training
* Model Comparison
* Model Evaluation
* Loan Prediction
* Streamlit Web Application

---

## 🤖 Machine Learning Models

The following Machine Learning models were trained:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The models are compared using:

* Accuracy
* Precision
* Recall
* F1 Score

The best-performing model is automatically selected and saved for future predictions.

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit
* Git
* GitHub

---

## 📂 Project Structure

```text
Loan_Approval_Project/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── loan_data_cleaned.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── notebooks/
│   ├── 01_data_check.py
│   ├── 02_data_cleaning.py
│   ├── 03_eda.py
│   ├── 04_preprocessing.py
│   ├── 05_model_training.py
│   ├── 06_model_evaluation.py
│   └── 07_prediction.py
│
├── models/
│   ├── preprocessor.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── best_model.joblib
│   ├── best_model_name.txt
│   └── model_comparison.csv
│
├── eda_outputs/
│
└── evaluation_outputs/
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── model_metrics.png
    └── test_predictions.csv
```

---

## 🔍 Project Workflow

### 1. Data Checking

The dataset is checked for:

* Number of rows and columns
* Column names
* Data types
* Missing values
* Duplicate records
* Loan status distribution

File:

```text
notebooks/01_data_check.py
```

---

### 2. Data Cleaning

The cleaning process includes:

* Cleaning column names
* Removing duplicate rows
* Removing unnecessary ID columns
* Handling missing numerical values using median
* Handling missing categorical values using mode
* Cleaning text values

File:

```text
notebooks/02_data_cleaning.py
```

---

### 3. Exploratory Data Analysis

EDA is used to understand the dataset.

The project generates:

* Loan status distribution
* Numerical feature distributions
* Categorical feature distributions
* Correlation analysis
* Target vs feature analysis

File:

```text
notebooks/03_eda.py
```

---

### 4. Data Preprocessing

The preprocessing stage includes:

* Feature and target separation
* Target encoding
* Train/Test split
* Standard Scaling
* One-Hot Encoding
* Saving the fitted preprocessor

File:

```text
notebooks/04_preprocessing.py
```

---

### 5. Model Training

Three Machine Learning models are trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The best model is selected automatically based on model performance.

File:

```text
notebooks/05_model_training.py
```

---

### 6. Model Evaluation

The selected model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

File:

```text
notebooks/06_model_evaluation.py
```

---

### 7. Loan Prediction

The prediction script allows a user to enter applicant information from the terminal.

The trained model predicts:

```text
LOAN APPROVED
```

or

```text
LOAN REJECTED
```

File:

```text
notebooks/07_prediction.py
```

---

## 🌐 Streamlit Web Application

The project also includes an interactive Streamlit application.

Users can enter applicant information using a web form and receive a loan approval prediction.

Run the application using:

```bash
streamlit run app.py
```

If the command does not work, use:

```bash
python -m streamlit run app.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Enter the project directory

```bash
cd Loan_Approval_Project
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Complete Project

Run the scripts in this order:

```bash
python notebooks\01_data_check.py
```

```bash
python notebooks\02_data_cleaning.py
```

```bash
python notebooks\03_eda.py
```

```bash
python notebooks\04_preprocessing.py
```

```bash
python notebooks\05_model_training.py
```

```bash
python notebooks\06_model_evaluation.py
```

```bash
python notebooks\07_prediction.py
```

Then start the Streamlit application:

```bash
streamlit run app.py
```

---

## 📊 Model Performance

The project compares the trained models using several evaluation metrics.

Detailed results are stored in:

```text
models/model_comparison.csv
```

The final classification report is stored in:

```text
evaluation_outputs/classification_report.txt
```

The project automatically selects the best-performing model and stores it as:

```text
models/best_model.joblib
```

---

## ✨ Features

* Complete end-to-end Machine Learning workflow
* Automatic dataset preprocessing
* Numerical feature scaling
* Categorical feature encoding
* Multiple Machine Learning algorithms
* Automatic best-model selection
* Model evaluation
* Confusion matrix visualization
* Command-line prediction system
* Interactive Streamlit application
* Saved reusable ML model
* GitHub-ready project structure

---

## ⚠️ Disclaimer

This project is created for educational and portfolio purposes.

The prediction produced by this application should not be used as an actual financial lending decision without proper validation, regulatory review, fairness testing, and human oversight.

---

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* ROC-AUC analysis
* Feature importance analysis
* Class imbalance handling
* Improved UI design
* Model explainability using SHAP
* Cloud deployment
* Database integration

---

## 👨‍💻 Author

Machine Learning Portfolio Project

**Loan Approval Prediction System**
