# Credit Card Fraud Detection using Machine Learning

## Overview
This project implements an end-to-end machine learning pipeline to detect fraudulent credit card transactions from a highly imbalanced dataset. A two-stage modeling approach is used to improve fraud detection performance while reducing false positives.

The project includes exploratory data analysis, preprocessing, feature engineering, model training, evaluation, threshold optimization, and cross-validation.

---

## Dataset
- Credit Card Fraud Detection Dataset (Kaggle)
- Approximately 284,000 transactions
- Approximately 0.17% fraudulent transactions

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.

The dataset is not included in this repository. Please download it from Kaggle using the link above.

---

## Exploratory Data Analysis (EDA)
EDA highlights the severe class imbalance in the dataset, the presence of outliers in transaction amounts, and strong correlations between certain anonymized features (V14, V17, V12, V10) and fraudulent behavior. Time-based transaction patterns also differ between fraud and non-fraud cases.

Visualizations include class distribution plots, boxplots for transaction amounts, correlation heatmaps, and feature distribution comparisons.

---

## Preprocessing
Numerical features are standardized using StandardScaler. The dataset is shuffled to remove ordering bias and split using stratified sampling to preserve class proportions. SMOTE is applied only to the training data to address class imbalance.

---

## Feature Engineering
The following features are engineered to capture behavioral and temporal patterns:
- Transaction hour derived from the time feature
- Log-transformed transaction amount
- Binary indicator for transactions occurring during unusual nighttime hours

---

## Modeling Approach

### Stage 1: Fraud Screening
An XGBoost classifier is trained as the first-stage model with the objective of maximizing recall. Each transaction receives a probability score indicating the likelihood of fraud.

Evaluation metrics include confusion matrix, precision, recall, F1-score, and ROC-AUC.

---

### Stage 2: Fraud Verification
A Random Forest classifier is trained on transactions flagged by Stage 1. It uses engineered features along with the Stage 1 probability score to refine predictions and reduce false positives.

---

## Threshold Optimization
Precision-recall analysis is used to identify the optimal classification threshold that maximizes the F1-score. This improves the trade-off between fraud detection and false alarms compared to the default threshold.

---

## Evaluation
Final evaluation is performed on unseen test data. The two-stage approach reduces false positives compared to a single-stage model. Model stability is assessed using 5-fold stratified cross-validation.

---

## How to Run

1. Clone the repository  
   git clone https://github.com/your-username/credit-card-fraud-detection.git  
   cd credit-card-fraud-detection  

2. Create and activate a virtual environment (Python 3.11 recommended)  
   python3.11 -m venv venv  
   source venv/bin/activate  

3. Install dependencies  
   pip install -r requirements.txt  

4. Add the dataset  
   Place `creditcard.csv` inside the `data/` directory  

5. Run the project  
   python main.py  

---

## Future Improvements
Potential improvements include hyperparameter tuning, cost-sensitive learning approaches, model explainability using SHAP values, and deployment as a real-time fraud detection system.

---

## Author
Gurnoor Singh Bagga  
Undergraduate student interested in Machine Learning and Data Science
