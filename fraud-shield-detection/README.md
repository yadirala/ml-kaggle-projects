```markdown
# Fraud Shield: Real-time Fraud Detection API

## Overview

This notebook demonstrates the end-to-end process of building a real-time fraud detection API using a financial transactions dataset. It covers data loading, exploratory data analysis, data preprocessing, model training with XGBoost, and deploying the model as a FastAPI service.

## Table of Contents

1.  [Setup](#1-setup)
2.  [Data Loading and Initial Exploration](#2-data-loading-and-initial-exploration)
3.  [Data Preprocessing](#3-data-preprocessing)
4.  [Model Training (XGBoost)](#4-model-training-xgboost)
5.  [Model Evaluation](#5-model-evaluation)
6.  [API Deployment with FastAPI](#6-api-deployment-with-fastapi)
7.  [Testing the API](#7-testing-the-api)

## 1. Setup

This section installs necessary libraries and loads the dataset.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Install FastAPI and Uvicorn
!pip install fastapi uvicorn nest-asyncio pyngrok -q
```

## 2. Data Loading and Initial Exploration

The dataset `fraud.csv` is loaded into a pandas DataFrame, and basic information like `df.info()` and `df.head()` are displayed to understand its structure and content.

```python
df = pd.read_csv('fraud.csv')
df.head()
df.info()
```

We also examine the distribution of fraudulent transactions:

```python
print(df['isFraud'].value_counts())
print(f"\nFraud percentage: {df['isFraud'].mean() * 100:.2f}%")
```

Fraudulent transactions are primarily found in `CASH_OUT` and `TRANSFER` transaction types:

```python
print(df.groupby('type')['isFraud'].sum())
```

## 3. Data Preprocessing

To focus on relevant transactions, the dataset is filtered to include only `CASH_OUT` and `TRANSFER` types. New features (`errorBalanceOrig` and `errorBalanceDest`) are engineered to capture discrepancies in account balances, which are strong indicators of fraud.

```python
df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])]
print(f"Remaining rows: {len(df)}")
print(f"Fraud percentage after filter: {df['isFraud'].mean() * 100:.2f}%")

df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
df['type'] = df['type'].map({'CASH_OUT': 0, 'TRANSFER': 1})

print(df[['errorBalanceOrig', 'errorBalanceDest', 'type']].head())
```

The data is then split into training and testing sets, with stratification to ensure a balanced representation of fraudulent transactions in both sets.

```python
from sklearn.model_selection import train_test_split

features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest',
            'errorBalanceOrig', 'errorBalanceDest']

X = df[features]
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Fraud in training: {y_train.sum()}")
print(f"Fraud in testing: {y_test.sum()}")
```

## 4. Model Training (XGBoost)

An XGBoost Classifier is trained on the preprocessed data. Due to the high class imbalance, `scale_pos_weight` is used to give more importance to the minority class (fraudulent transactions) during training.

```python
from xgboost import XGBClassifier

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

model.fit(X_train, y_train)
print("Model trained successfully!")
```

## 5. Model Evaluation

The trained model's performance is evaluated using AUC-ROC score, classification report, and confusion matrix.

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

The model is saved for later deployment, and feature importances are visualized to understand which features contribute most to fraud detection.

```python
import joblib
joblib.dump(model, 'fraud_shield_model.pkl')
print("Model saved!")

plt.figure(figsize=(10, 6))
feat_importance = model.feature_importances_
feat_names = features
plt.barh(feat_names, feat_importance)
plt.title('Fraud Shield - Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

## 6. API Deployment with FastAPI

A FastAPI application (`main.py`) is created to serve the trained fraud detection model. It defines a `/predict` endpoint that accepts transaction details and returns a fraud prediction, probability, and risk level.

```python
# main.py content
app_code = """
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Shield API")

# Load model
model = joblib.load('fraud_shield_model.pkl')

# Input schema
class Transaction(BaseModel):
    type: int  # 0=CASH_OUT, 1=TRANSFER
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    errorBalanceOrig: float
    errorBalanceDest: float

@app.get("/")
def home():
    return {"message": "Fraud Shield API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    features = [[
        transaction.type,
        transaction.amount,
        transaction.oldbalanceOrg,
        transaction.newbalanceOrig,
        transaction.oldbalanceDest,
        transaction.newbalanceDest,
        transaction.errorBalanceOrig,
        transaction.errorBalanceDest
    ]]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    }
"""

with open('main.py', 'w') as f:
    f.write(app_code)
print("FastAPI app created!")
```

## 7. Testing the API

The FastAPI application is run in the background, and its functionality is tested using sample fraudulent and legitimate transactions. The responses demonstrate the API's ability to classify transactions and provide fraud probabilities.

```python
import nest_asyncio
import uvicorn
import threading
import requests
import time

nest_asyncio.apply()

config = uvicorn.Config("main:app", port=8000, log_level="error")
server = uvicorn.Server(config)

thread = threading.Thread(target=server.run)
thread.daemon = True
thread.start()

time.sleep(2) # Give the server a moment to start
print("Fraud Shield API is running!")

# Example of a fraudulent transaction
fraud_transaction = {
    "type": 1,
    "amount": 500000.0,
    "oldbalanceOrg": 500000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "errorBalanceOrig": 0.0,
    "errorBalanceDest": 500000.0
}

response = requests.post("http://localhost:8000/predict", json=fraud_transaction)
print("\nFraudulent Transaction Test:")
print(response.json())

# Example of a legitimate transaction
legit_transaction = {
    "type": 0,
    "amount": 1000.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4000.0,
    "oldbalanceDest": 2000.0,
    "newbalanceDest": 3000.0,
    "errorBalanceOrig": 0.0,
    "errorBalanceDest": 0.0
}

response = requests.post("http://localhost:8000/predict", json=legit_transaction)
print("\nLegitimate Transaction Test:")
print(response.json())
```

This README provides a comprehensive guide to understanding and replicating the fraud detection solution presented in the notebook.
