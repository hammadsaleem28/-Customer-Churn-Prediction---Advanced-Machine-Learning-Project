# Customer-Churn-Prediction-- (Advanced-Machine-Learning-Project)

## Project Summary
This project predicts customer churn for a telecom company using advanced machine learning techniques.  
It uses Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning, cross-validation, and explainability via SHAP values.  
The goal is to identify customers likely to leave, enabling proactive retention strategies.

## Dataset
The dataset used is the "Telco Customer Churn" dataset from Kaggle:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Features
- Customer demographics  
- Account information  
- Services subscribed  
- Tenure, Monthly Charges, Total Charges  

## Key Techniques
- Advanced exploratory data analysis (EDA)  
- Feature engineering (tenure bins, encoding)  
- Pipelines for preprocessing and modeling  
- GridSearchCV for hyperparameter tuning  
- Cross-validation (Stratified KFold)  
- Model explainability using SHAP  

## Results
- Logistic Regression, Random Forest, and XGBoost models trained and evaluated  
- Best model chosen based on ROC-AUC score  
- Visualization of ROC curves and feature importance  

## Setup Instructions
1. Clone the repo  
2. Install dependencies:  
   ```bash
   pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
