# ===============================
# Credit Card Fraud Detection ML
# GitHub-ready notebook
# ===============================

# 1️⃣ Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score
)
from imblearn.over_sampling import SMOTE

# ===============================
# 2️⃣ Load Dataset
# ===============================
df = pd.read_csv("data/creditcard.csv")

# ===============================
# 3️⃣ Exploratory Data Analysis (EDA)
# ===============================

# Outlier check
plt.figure(figsize=(8,4))
sns.boxplot(x=df['Amount'])
plt.title("Boxplot of Transaction Amounts")
plt.show()

# Class imbalance
fraud_counts = df['Class'].value_counts()
plt.figure(figsize=(6,4))
sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette="coolwarm")
plt.title("Distribution of Fraud (1) vs Non-Fraud (0) Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
fraud_percentage = (fraud_counts[1] / fraud_counts.sum()) * 100
print(f"Fraud cases: {fraud_counts[1]} ({fraud_percentage:.4f}% of total)")

# Correlation heatmap
df_scaled = df.copy()
scaler = StandardScaler()
df_scaled[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
corr_matrix = df_scaled.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of All Features")
plt.show()

# Top feature distributions
top_features = ['V14', 'V17', 'V12', 'V10']
plt.figure(figsize=(12,8))
for i, col in enumerate(top_features, 1):
    plt.subplot(2,2,i)
    sns.kdeplot(data=df_scaled, x=col, hue='Class', fill=True)
    plt.title(f'Distribution of {col} by Class')
plt.tight_layout()
plt.show()

# ===============================
# 4️⃣ Preprocessing + SMOTE
# ===============================

# Shuffle dataset
df_scaled = df_scaled.sample(frac=1, random_state=42).reset_index(drop=True)

# Features & target
X = df_scaled.drop('Class', axis=1)
y = df_scaled['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ===============================
# 5️⃣ Feature Engineering
# ===============================

# Transaction hour
df_scaled['Hour'] = (df_scaled['Time'] / 3600) % 24
# Log-transform amount
df_scaled['Log_Amount'] = np.log1p(df_scaled['Amount'])
# Binary flag for unusual hours
df_scaled['Is_Night'] = df_scaled['Hour'].apply(lambda x: 1 if (x<6 or x>22) else 0)

# ===============================
# 6️⃣ Stage 1 Models
# ===============================

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_res, y_train_res)
y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Results:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, digits=4))

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)
xgb_model.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Results:")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_xgb))

# Stage 1 scores
X_test_df = X_test.copy()
X_test_df['y_true'] = y_test
X_test_df['Stage1_Score'] = xgb_model.predict_proba(X_test)[:,1]
X_test_df['Stage1_Pred'] = (X_test_df['Stage1_Score'] >= 0.5).astype(int)

# ===============================
# 7️⃣ Stage 2 Model
# ===============================

# Prepare Stage 2 dataset
stage2_df = df_scaled[df_scaled['Class'].isin([1])].copy()  # only fraud? we can modify if needed
X_stage2 = stage2_df[['Stage1_Score','Log_Amount','Hour','Is_Night']]
y_stage2 = stage2_df['Class']

rf_stage2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_stage2.fit(X_stage2, y_stage2)

# Apply Stage 2 to flagged Stage 1 transactions
flagged = X_test_df[X_test_df['Stage1_Pred']==1].copy()
flagged['Log_Amount'] = np.log1p(flagged['Amount'])
flagged['Hour'] = (flagged['Time']/3600) % 24
flagged['Is_Night'] = flagged['Hour'].apply(lambda x: 1 if (x<6 or x>22) else 0)
flagged['Stage2_Pred'] = rf_stage2.predict(flagged[['Stage1_Score','Log_Amount','Hour','Is_Night']])

# Final prediction
X_test_df.loc[flagged.index, 'Final_Pred'] = flagged['Stage2_Pred']
X_test_df['Final_Pred'] = X_test_df['Final_Pred'].fillna(0)

# Final evaluation
print("Two-Stage Model Results:")
print(confusion_matrix(X_test_df['y_true'], X_test_df['Final_Pred']))
print(classification_report(X_test_df['y_true'], X_test_df['Final_Pred'], digits=4))

# ===============================
# 8️⃣ Threshold Optimization
# ===============================

y_prob = xgb_model.predict_proba(X_test)[:,1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2*(precisions*recalls)/(precisions+recalls)
best_idx = np.nanargmax(f1_scores)
best_thresh = thresholds[best_idx]
y_pred_opt = (y_prob >= best_thresh).astype(int)
print(f"Best threshold: {best_thresh:.3f}")
print(classification_report(y_test, y_pred_opt, digits=4))

# Plot precision, recall, F1 vs threshold
plt.figure(figsize=(8,6))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1', linewidth=2)
plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best F1 = {best_thresh:.2f}')
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall & F1 vs Threshold")
plt.legend()
plt.show()

# ===============================
# 9️⃣ Cross-Validation (Optional)
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
for train_idx, test_idx in cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    xgb_model.fit(X_tr, y_tr)
    y_prob = xgb_model.predict_proba(X_te)[:,1]
    auc_scores.append(roc_auc_score(y_te, y_prob))
print("CV AUC scores:", auc_scores)
print("Mean CV AUC:", np.mean(auc_scores))
