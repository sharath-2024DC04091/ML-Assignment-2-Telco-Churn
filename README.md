# ML Assignment 2 - Telco Customer Churn Prediction

## 1. Problem Statement

The objective of this project is to predict customer churn for a telecom company using multiple machine learning classification models. Customer churn refers to whether a customer has left the company during the current quarter.

The task involves implementing six classification models, evaluating them using multiple performance metrics, and deploying the final solution as an interactive Streamlit web application.

---

## 2. Dataset Description

The dataset used is the "Telco Customer Churn: IBM Dataset" obtained from Kaggle:
https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

- Total Instances: 7043
- Total Features Used: 19 predictors
- After one-hot encoding, the total number of features increased to 30.
- Target Variable: Churn Value (1 = Customer churned, 0 = Customer retained)

The dataset includes customer demographic information, service subscription details, billing information, and contract type.

Data preprocessing steps included:
- Removal of identifier and leakage columns
- Conversion of Total Charges to numeric
- Handling missing values using median imputation
- One-hot encoding of categorical variables
- Stratified train-test split (80:20)
- The dataset was split into 80% training data and 20% testing data using stratified sampling to maintain class distribution.

---

## 3. Models Used and Performance Comparison

The following models were implemented:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes Classifier - Gaussian
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.7999 | 0.8485 | 0.6377 | 0.5695 | 0.6017 | 0.4699 |
| Decision Tree | 0.7296 | 0.6549 | 0.4908 | 0.4973 | 0.4940 | 0.3095 |
| kNN | 0.7608 | 0.7786 | 0.5510 | 0.5348 | 0.5427 | 0.3809 |
| Naive Bayes | 0.6884 | 0.8165 | 0.4536 | 0.8503 | 0.5916 | 0.4241 |
| Random Forest | 0.7899 | 0.8348 | 0.6250 | 0.5214 | 0.5685 | 0.4343 |
| XGBoost | 0.8055 | 0.8542 | 0.6613 | 0.5481 | 0.5994 | 0.4761 |

---

## 4. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Performed strongly with high AUC and balanced MCC, indicating a significant linear relationship between features and churn. |
| Decision Tree | Showed lower AUC and MCC compared to ensemble methods, likely due to overfitting and high variance. |
| kNN | Delivered moderate performance. Sensitive to feature scaling and distance-based decision boundaries. |
| Naive Bayes | Achieved very high recall but lower precision, indicating aggressive churn prediction. Suitable when minimizing false negatives is important. |
| Random Forest (Ensemble) | Improved stability compared to a single decision tree. Delivered strong overall performance with balanced metrics. |
| XGBoost (Ensemble) | Best performing model overall. Achieved highest Accuracy, AUC, and MCC, effectively capturing complex patterns in the dataset. |


---

## 5. Deployment

The trained models were saved and integrated into a Streamlit web application. The app allows:

- Uploading test data (CSV)
- Selecting a model
- Viewing evaluation metrics
- Displaying confusion matrix

The application is deployed on Streamlit Community Cloud.
