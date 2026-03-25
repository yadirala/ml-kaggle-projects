Heart Disease Prediction Project
Project Description
This project aims to predict the presence of heart disease using various patient attributes. Machine learning models are trained and evaluated to achieve the best possible prediction accuracy.

Dataset
The dataset used in this project is heart_cleveland_upload.csv, which contains information about patients, including:

age: Age of the patient
sex: Sex of the patient (1 = male; 0 = female)
cp: Chest pain type (0-3)
trestbps: Resting blood pressure
chol: Serum cholestoral in mg/dl
fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
restecg: Resting electrocardiographic results (0-2)
thalach: Maximum heart rate achieved
exang: Exercise induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise relative to rest
slope: The slope of the peak exercise ST segment
ca: Number of major vessels (0-3) colored by flourosopy
thal: Thallium stress test result (0 = normal; 1 = fixed defect; 2 = reversible defect)
condition: Diagnosis of heart disease (angiographic disease status) (0 = no disease; 1 = disease)
Data Preprocessing
Feature Scaling: Continuous features (age, trestbps, chol, thalach, oldpeak) were scaled using StandardScaler.
One-Hot Encoding: Categorical features (cp, restecg, slope, ca, thal) were transformed using OneHotEncoder.
Train-Test Split: The data was split into 80% training and 20% testing sets.
Models Trained
The following machine learning models were trained and evaluated:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Random Forest Classifier
XGBoost Classifier
Model Evaluation
Models were evaluated based on their accuracy, classification reports, and cross-validation recall scores. Hyperparameter tuning was performed using GridSearchCV for Random Forest and XGBoost to optimize their performance, focusing on recall.

Best Performing Models (based on cross-validation recall)
Logistic Regression: Best recall of 0.81
XGBoost: Best recall of 0.83 (after hyperparameter tuning)
Random Forest: Best recall of 0.83 (after hyperparameter tuning)
An ROC curve was plotted for the optimized XGBoost model, and an optimal threshold was determined to balance precision and recall.
