# 🏦 Credit Risk / Loan Default Prediction

## Overview

This project implements an **end-to-end machine learning system** to predict the **probability of loan default** based on applicant financial and demographic data.

Unlike basic classification projects, this work focuses on **risk modeling**, emphasizing probability-based decisions, business tradeoffs, and model comparison rather than accuracy alone.

The project follows a complete ML lifecycle:

- Data understanding
- Feature engineering
- Model training
- Business-aware evaluation
- Risk scoring
- Model persistence and reuse

---

## Problem Statement

Financial institutions must decide whether to approve or reject loan applications.

**Goal:**  
Estimate the likelihood that a loan applicant will default and categorize applicants into **Low, Medium, or High risk** groups to support decision-making.

---

## Dataset

- **German Credit Dataset** (via OpenML)
- 1,000 loan applicants
- Mixed categorical and numerical features
- Target variable: `class`
  - `good` → loan repaid
  - `bad` → loan defaulted

---

## Project Structure

credit-risk-ml/
├── notebook/
│ ├── credit_risk_prediction.ipynb
│ └── reuse_model.ipynb
├── model/
│ └── credit_risk_model.pkl
├── README.md
├── requirements.txt
└── venv/

---

## Machine Learning Approach

### 1. Data Preparation

- Separated features (X) and target (y)
- Encoded categorical variables using one-hot encoding
- Applied stratified train/test split to preserve class distribution

### 2. Modeling

Two models were trained and compared:

- **Logistic Regression** — interpretable baseline
- **Random Forest Classifier** — captures non-linear patterns

### 3. Evaluation Metrics

Rather than relying solely on accuracy, the project emphasizes:

- Precision
- Recall
- F1-score
- Confusion matrix analysis

Special attention was given to **false negatives**, which represent missed loan defaults and carry higher financial risk.

### 4. Risk Scoring

Model outputs were converted from binary predictions into **probability-based risk scores**, then categorized into:

- Low Risk
- Medium Risk
- High Risk

---

## Results

- Random Forest achieved improved recall for high-risk borrowers
- Probability outputs enabled ranking applicants by risk
- Risk tiering translated ML predictions into decision-ready insights

---

## Model Reuse

The final model, preprocessing scaler, and feature columns were saved using `pickle` to support reusable inference on new applicants.

---

## Technologies Used

- Python 3.11
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Key Takeaways

- Built a business-aware ML classification system
- Compared multiple models and justified tradeoffs
- Applied probability-based decision-making
- Designed a reusable inference pipeline

---
