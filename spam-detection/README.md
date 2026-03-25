# SMS Spam Detection

NLP binary classification project to detect spam messages using text preprocessing and machine learning.

---

## Dataset

**Source:** UCI SMS Spam Collection Dataset (`spam.csv`)  
**Size:** 5572 messages · 2 features · 1 target variable  
**Class Balance:** 87% ham / 13% spam (imbalanced)

| Column | Description |
|---|---|
| `text` | Raw SMS message |
| `target` | 0 = ham, 1 = spam |

---

## Pipeline
```
Load → Clean Text → Train/Test Split → TF-IDF Vectorization → Model Training → K-Fold CV
```

> **Note:** TF-IDF vectorizer fit on training data only to prevent data leakage.

---

## Text Preprocessing

Applied in order:

1. **Lowercase** — normalize case
2. **Remove punctuation/numbers/special chars** — regex `[^a-zA-Z\s]`
3. **Remove stop words** — NLTK English stop words
4. **Stemming** — Porter Stemmer

---

## Vectorization

**TF-IDF** chosen over Bag of Words — automatically downweights common words and upweights rare but meaningful spam signals like "winner", "prize", "free".

Vocabulary size: **7424 unique words**

---

## Models Trained

- Multinomial Naive Bayes
- Logistic Regression
- SVM (SVC)
- Random Forest Classifier
- XGBoost Classifier

---

## Results

Primary metric: **Precision** — minimizing False Positives (legitimate emails wrongly blocked) is the priority for a spam filter.

### Cross-Validation Results (K-Fold, k=5)

| Model | Precision | Recall |
|---|---|---|
| Naive Bayes | 1.000 | 0.701 |
| SVM | 0.996 | 0.758 |
| **Random Forest** | **0.998** | **0.779** |
| XGBoost | 0.929 | 0.823 |
| Logistic Regression | 0.978 | 0.590 |

### Selected Model: Random Forest
Best balance of precision (0.998) and recall (0.779) — near-zero false alarms with highest spam catch rate among reliable models.

---

## Key Findings

- Naive Bayes achieved perfect precision (1.0) — classic choice for spam detection
- SVM strong on high-dimensional text (7424 features) as expected
- XGBoost highest recall but lowest precision — catches most spam but risks blocking legitimate emails
- No SMOTE needed — 13% minority class sufficient for models to learn

---

## Tech Stack

- Python, Pandas, NumPy
- NLTK (stopwords, Porter Stemmer)
- Scikit-learn (TF-IDF, models)
- XGBoost
