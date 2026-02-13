# 💳 Credit Card Fraud Detection Using Machine Learning

## Machine Learning Assignment - 2  
M.Tech (AIML) – BITS Pilani  

---

# 1️⃣ Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models for detecting fraudulent credit card transactions.

Credit card fraud detection is a critical real-world business problem where the goal is to correctly identify fraudulent transactions while minimizing false alarms.

This project includes:
- Implementation of 6 classification models
- Evaluation using multiple performance metrics
- Development of an interactive Streamlit web application
- Deployment on Streamlit Community Cloud

---

# 2️⃣ Dataset Description

Dataset Name: **Credit Card Fraud Detection Dataset**  
Source: Kaggle  

Link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

### Dataset Characteristics:

- Total Instances: 284,807 transactions  
- Total Features: 30 numerical features  
- Target Variable: `Class`  
    - 0 → Legitimate Transaction  
    - 1 → Fraudulent Transaction  
- Highly Imbalanced Dataset  

Features V1–V28 are PCA-transformed features.  
Additional features include:
- `Time`
- `Amount`

The dataset is highly imbalanced, making evaluation metrics like AUC and MCC important.

---

# 3️⃣ Models Used and Evaluation Metrics

The following 6 machine learning models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

# 4️⃣ Evaluation Metrics Used

For each model, the following metrics were calculated:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

# 5️⃣ Model Comparison Table

| ML Model Name        | Accuracy | AUC  | Precision | Recall | F1 Score | MCC |
|----------------------|----------|------|-----------|--------|----------|-----|
| Logistic Regression  | 0.975457 | 0.972063 | 0.060811 | 0.918367 | 0.114068 | 0.232876 |
| Decision Tree        | 0.998929 | 0.861946 | 0.676190 | 0.724490 | 0.699507 | 0.699389 |
| KNN                  | 0.999544 | 0.943744 | 0.918605 | 0.806122 | 0.858696 | 0.860305 |
| Naive Bayes          | 0.976405 | 0.963248 | 0.058782 | 0.846939 | 0.109934 | 0.219519 |
| Random Forest        | 0.999508 | 0.952909 | 0.960526 | 0.744898 | 0.839080 | 0.845645 |
| XGBoost              | 0.999438 | 0.938952 | 0.866667 | 0.795918 | 0.829787 | 0.830261 |

---

# 6️⃣ Observations on Model Performance

| ML Model | Observation |
|----------|------------|
| Logistic Regression | Performs well despite simplicity. Balanced performance across metrics. |
| Decision Tree | Prone to overfitting. Performance slightly unstable on imbalanced dataset. |
| KNN | Moderate performance. Sensitive to scaling and computationally expensive. |
| Naive Bayes | Lower performance due to independence assumption among features. |
| Random Forest | Strong performance due to ensemble learning. Better handling of imbalance. |
| XGBoost | Best overall performance with high AUC and MCC. Robust and efficient ensemble model. |

---

# 7️⃣ Streamlit Web Application Features

The deployed Streamlit app includes:

- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization
- Interactive user interface

Live App Link:
https://credit-card-fraud-ml-app-epnd9k4vfittjhikuwhjcw.streamlit.app/

---

# 8️⃣ GitHub Repository

Repository Link:
https://github.com/suhanibhargava4402/credit-card-fraud-ml-streamlit/tree/main

---

# 9️⃣ Conclusion

Ensemble models such as Random Forest and XGBoost outperform individual models in fraud detection due to their ability to handle complex decision boundaries and imbalanced datasets effectively.

The project demonstrates a complete end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment.

Although multiple models achieved high accuracy due to dataset imbalance, KNN achieved the highest Matthews Correlation Coefficient (MCC = 0.8603) and strong F1-score performance.

Since MCC is a balanced metric suitable for imbalanced datasets, KNN is considered the best performing model in this study.

Ensemble models such as Random Forest and XGBoost also showed strong performance, but KNN achieved the best overall balance between precision and recall.





