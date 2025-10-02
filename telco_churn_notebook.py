# telco_churn_notebook.py
# Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost (optional), joblib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------- 1. Load ----------
DATA_PATH = "data:telco_customer_churn.csv"
df = pd.read_csv(DATA_PATH)
print("Initial shape:", df.shape)
df.head()

# ---------- 2. Clean ----------
# Some versions have whitespace in TotalCharges -> convert to numeric
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan), errors='coerce')
    # Impute missing TotalCharges as MonthlyCharges * tenure (approx) or median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    
# Drop customerID (identifier)
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# ---------- 3. Feature engineering ----------
# Tenure bucket
df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-1, 6, 12, 24, 48, 72, 1000],
                            labels=['0-6','7-12','13-24','25-48','49-72','73+'])
# Count of services subscribed
service_cols = ['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
               'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
df['num_services'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

# Payment method one-hot later; Contract mapping
df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)

# ---------- 4. Select features ----------
# Identify categorical and numeric
target = 'Churn'
drop_cols = []  # add any to drop
X = df.drop(columns=[target] + drop_cols)
y = df[target]

numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

# Remove numeric columns that are actually encoded as object
# (e.g., 'SeniorCitizen' might be int; keep it)
print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

# ---------- 5. Preprocessing pipeline ----------
from sklearn.preprocessing import OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

# ---------- 6. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- 7. Baseline model: Logistic Regression ----------
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("Logistic Regression results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ---------- 8. Random Forest with simple tuning ----------
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:,1]

print("Random Forest results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1:", f1_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf))

# Feature importance (approx): need to map to preprocessor columns
# Get feature names from preprocessor after OHE
ohe_cols = rf_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['ohe'].get_feature_names_out(categorical_cols)
all_features = list(numeric_cols) + list(ohe_cols)
importances = rf_pipeline.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(20)
print(feat_imp)
feat_imp.plot(kind='barh')
plt.title("Top 20 Feature importances (Random Forest)")
plt.gca().invert_yaxis()
plt.show()

# ---------- 9. Save model ----------
os.makedirs("models", exist_ok=True)
joblib.dump(rf_pipeline, "models/telco_churn_rf.pkl")
print("Saved model to models/telco_churn_rf.pkl")

# ---------- 10. Create churn-risk segments ----------
X_test_copy = X_test.copy()
X_test_copy['actual_churn'] = y_test.values
X_test_copy['pred_proba'] = y_proba_rf
segments = X_test_copy.groupby(pd.cut(X_test_copy['pred_proba'], bins=[0,0.2,0.5,0.7,1],
                                      labels=['Low','Medium','High','Very High']))['actual_churn'].agg(['count','mean']).rename(columns={'mean':'churn_rate'})
print(segments.sort_values(by='churn_rate', ascending=False))
