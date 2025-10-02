# Telco Churn Prediction

**Description:**  
Developed a machine learning model to predict customer churn for a telecom company using Python (pandas, scikit-learn). The project demonstrates end-to-end data science workflow including data cleaning, feature engineering, modeling, evaluation, and visualization.

**Dataset:**  
- ~7,000 customer records with 21 features  
- Features include customer demographics, contract type, service usage, and tenure  
- Source: Kaggle Telco Customer Churn dataset

**Features & Methods:**  
- **Data preprocessing:** Handled missing values, encoded categorical variables, scaled numeric features  
- **Feature engineering:** Created aggregated service usage metrics, tenure buckets, and churn indicators  
- **Modeling:** Logistic Regression (baseline), Random Forest, Gradient Boosting (optional)  
- **Evaluation metrics:**  
  - Accuracy: 73.9%  
  - Precision: 50.5%  
  - Recall: 80.2%  
  - F1 Score: 61.9%  
  - ROC-AUC: 0.845  
- **Visualization:** Confusion matrix, feature importance, and churn probability distributions using matplotlib/seaborn  
