"""
Advanced Bank Churn Prediction Model
Using Random Forest with Hyperparameter Tuning and Robust Evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import os

warnings.filterwarnings('ignore')

# --- 1. SETUP & STYLE ---
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def log(msg):
    print(f"[MODELING] {msg}")

# --- 2. DATA LOADING & PREPROCESSING ---
log("Loading dataset: Bank_Churn.csv")
df = pd.read_csv('Bank_Churn.csv')

# Drop irrelevant identifiers
df = df.drop(['CustomerId', 'Surname'], axis=1)

# Feature Engineering
log("Performing feature engineering...")
df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Adult', 'Senior', 'Elderly'])
df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['TenurePerAge'] = df['Tenure'] / df['Age']
df['CreditScorePerAge'] = df['CreditScore'] / df['Age']

# Identify features
target = 'Exited'
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                    'BalanceSalaryRatio', 'TenurePerAge', 'CreditScorePerAge']
categorical_features = ['Geography', 'Gender', 'Age_Bin']

# --- 3. PIPELINE CONSTRUCTION ---
log("Building preprocessing pipeline...")
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# --- 4. DATA SPLIT ---
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
log(f"Data split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")

# --- 5. HYPERPARAMETER TUNING ---
log("Starting Hyperparameter Tuning (Randomized Search)...")
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, 
    n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1, scoring='f1'
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

log(f"Best Parameters: {random_search.best_params_}")

# --- 6. PREDICTIONS & EVALUATION ---
log("Evaluating model on test set...")
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Classification Report
report = classification_report(y_test, y_pred)
print("\n" + "="*20 + " CLASSIFICATION REPORT " + "="*20)
print(report)

# --- 7. VISUALIZATIONS ---
log("Generating performance visualizations...")

# A. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: Predicted vs Actual')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('advanced_confusion_matrix.png', dpi=300)

# B. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('advanced_roc_curve.png', dpi=300)

# C. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP={avg_precision:.2f})')
plt.savefig('advanced_pr_curve.png', dpi=300)

# D. Feature Importance
log("Extracting feature importance...")
# Get feature names from preprocessor
ohe_feature_names = best_model.named_steps['preprocessor'].transformers_[1][1]\
    .named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_feature_names)

importances = best_model.named_steps['classifier'].feature_importances_
feat_importances = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=feat_importances.values[:15], y=feat_importances.index[:15], palette="magma")
plt.title('Top 15 Predictive Features for Bank Churn')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('advanced_feature_importance.png', dpi=300)

log("Modeling process complete. Visuals saved to disk.")
