# Heart Disease Prediction

Binary classification project to predict the presence of heart disease using patient clinical data.

---

## Dataset

**Source:** `heart_cleveland_upload.csv` (UCI Cleveland Heart Disease Dataset)  
**Size:** 297 patients · 13 features · 1 target variable  
**Class Balance:** 55% no disease / 45% disease

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia type (0 = normal, 1 = fixed, 2 = reversible) |
| `condition` | **Target** — 0 = no disease, 1 = disease |

---

## Pipeline

```
EDA → Train/Test Split → Preprocessing → Model Training → K-Fold CV → GridSearchCV → ROC-AUC
```

> **Note:** Train/Test split is performed **before** preprocessing to prevent data leakage.

---

## Preprocessing

| Feature Type | Features | Transformation |
|---|---|---|
| Continuous | age, trestbps, chol, thalach, oldpeak | StandardScaler |
| Categorical | cp, restecg, slope, ca, thal | OneHotEncoder (drop='first') |
| Binary | sex, fbs, exang | No transformation |

---

## Models Trained

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Random Forest Classifier
- XGBoost Classifier

---

## Results

Models evaluated using **Recall** as the primary metric — minimizing False Negatives (missed disease patients) is the clinical priority.

### Cross-Validation Recall (K-Fold, k=5)

| Model | CV Recall |
|---|---|
| XGBoost (tuned) | **0.825** |
| Random Forest (tuned) | **0.825** |
| Logistic Regression | 0.81 |
| KNN | 0.79 |
| Naive Bayes | 0.73 |
| Decision Tree | 0.69 |

### Hyperparameter Tuning (GridSearchCV)

**XGBoost best params:** `learning_rate=0.2`, `max_depth=3`, `n_estimators=100`  
**Random Forest best params:** `max_depth=5`, `max_features='log2'`, `n_estimators=300`

### ROC-AUC

XGBoost (tuned) AUC = **0.77**  
Optimal threshold determined from ROC curve to balance Precision and Recall.

---

## Key Findings

- Logistic Regression outperformed complex models on default settings — small datasets (297 rows) favor simpler models
- XGBoost improved from 0.68 (single split) → 0.825 (after K-Fold CV + GridSearchCV)
- Strongest predictors: `thal` (r=0.52) and `thalach` (r=-0.42)
- No missing values found; no class imbalance treatment required

---

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
