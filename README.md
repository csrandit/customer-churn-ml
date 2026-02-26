# 📊 Customer Churn Prediction (Telecom)
  An end-to-end production-style machine learning pipeline for telecom churn prediction.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Complete-success)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.83-brightgreen)

---

---

## 📈 Model Performance

### ROC Curve
![ROC Curve](reports/roc_curve.png)

### Confusion Matrix
![Confusion Matrix](reports/confusion_matrix.png)

---

## 🔝 Top Churn Drivers

### Increasing Churn
![Top Increasing](reports/top_10_increasing.png)

### Decreasing Churn
![Top Decreasing](reports/top_10_decreasing.png)

---
---
## 📌 Project Overview

Customer churn prediction is a critical problem in subscription-based industries.  
This project develops an end-to-end machine learning pipeline to identify customers at risk of leaving a telecom company, enabling proactive retention strategies.

The project focuses not only on predictive performance, but also on business interpretability and decision-making trade-offs.

---

## 🎯 Problem Definition

Binary Classification Task  

Target Variable: ChurnValue  
- 1 = Customer churned  
- 0 = Customer retained  

The objective is to maximize identification of at-risk customers while balancing the business cost of false positives and false negatives.

---

## 📂 Dataset

IBM Telco Customer Churn Dataset  

- 7,043 customers  
- 33 features  
- Includes demographic, contract, billing, and service-related attributes  

---

## 🔎 Exploratory Data Analysis (EDA)

Key Observations:

- ~27% of customers churned (moderate class imbalance)
- Customers on month-to-month contracts have significantly higher churn rates
- Customers with shorter tenure are more likely to churn
- Contract length strongly correlates with retention

All EDA visualizations are saved in the reports/ directory.

---

## 🛠 Feature Engineering & Preprocessing

- Cleaned and standardized column names
- Converted TotalCharges to numeric
- Removed data leakage variables (CLTV, ChurnScore, ChurnReason)
- Removed high-cardinality geographic features to prevent overfitting
- Implemented missing value imputation inside the ML pipeline
- One-hot encoded categorical variables
- Standardized numerical features

All preprocessing steps are handled using a ColumnTransformer inside a Scikit-learn Pipeline to prevent data leakage.

---

## 🧱 Technical Architecture

The project uses a structured ML pipeline:

1. Train/Test split with stratification
2. ColumnTransformer for preprocessing
3. Logistic Regression classifier
4. 5-fold Cross-Validation
5. Threshold tuning for recall optimization
6. Odds Ratio extraction for interpretability
7. Automated saving of feature importance reports

This ensures reproducibility, modularity, and production-style workflow.

---

## 🤖 Model Development

Baseline Model: Logistic Regression  

Why Logistic Regression?

- Interpretable coefficients
- Strong baseline for binary classification
- Probability outputs for threshold optimization
- Suitable for business explainability

---

## 📈 Model Evaluation

Metrics Used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 5-Fold Cross-Validation

### Performance

- ROC-AUC ≈ 0.83
- Recall improved from 53% → 73% after threshold tuning (0.5 → 0.3)

Threshold tuning was applied to prioritize recall, as false negatives (missed churners) are more costly than false positives in churn prediction scenarios.

---

## 🔍 Explainability

Model coefficients were transformed into Odds Ratios to identify key churn drivers.

### Strongest Churn Drivers

- Month-to-month contracts  
- No dependents  
- Fiber optic service  

### Strongest Churn Reducers

- Long tenure  
- Two-year contracts  
- Having dependents  

Feature importance reports and visualizations are saved in reports/.

---

## 💼 Business Impact

This model enables:

- Early identification of high-risk customers
- Targeted retention campaigns
- Improved customer lifetime value
- Optimized marketing budget allocation
- Data-driven decision-making

---

## ⚠ Model Limitations

- Dataset reflects U.S. telecom behavior and may not generalize globally without retraining.
- Logistic Regression assumes linear log-odds relationships.
- No cost-sensitive learning implemented.
- No hyperparameter optimization beyond baseline configuration.

---

## 🚀 Future Improvements

- Compare with tree-based models (Random Forest, XGBoost, Gradient Boosting)
- Perform hyperparameter tuning
- Introduce cost-sensitive learning
- Deploy model via FastAPI
- Containerize using Docker
- Simulate real-world retention budget constraints

---

## 🚀 How to Run

```bash
python src/pipeline.py