# ===============================
# Import Required Libraries
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
# App Title
# ===============================

st.title("💳 Credit Card Fraud Detection - ML Models")
st.write("Machine Learning Assignment 2 - BITS Pilani")

# ===============================
# Upload Dataset
# ===============================

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    if "Class" not in df.columns:
        st.error("Dataset must contain 'Class' column.")
    else:
        
        # Separate features and target
        X = df.drop("Class", axis=1)
        y = df["Class"]
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # ===============================
        # Model Selection
        # ===============================
        
        st.sidebar.header("Select Model")
        model_name = st.sidebar.selectbox(
            "Choose Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            ]
        )
        
        if model_name == "Logistic Regression":
            model = LogisticRegression(class_weight="balanced", max_iter=1000)
        
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(class_weight="balanced")
        
        elif model_name == "KNN":
            model = KNeighborsClassifier()
        
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        
        elif model_name == "Random Forest":
            model = RandomForestClassifier(class_weight="balanced", n_estimators=100)
        
        elif model_name == "XGBoost":
            model = XGBClassifier(eval_metric="logloss")
        
        # Train Model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        
        # ===============================
        # Metrics
        # ===============================
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.subheader("📊 Evaluation Metrics")
        st.write("Accuracy:", round(accuracy, 4))
        st.write("AUC:", round(auc, 4))
        st.write("Precision:", round(precision, 4))
        st.write("Recall:", round(recall, 4))
        st.write("F1 Score:", round(f1, 4))
        st.write("MCC:", round(mcc, 4))
        
        # ===============================
        # Confusion Matrix
        # ===============================
        
        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)