readme = """# House Prices Prediction

## Problem
Predict sale price of residential homes based on 79 features.

## Dataset
Kaggle House Prices Competition
- Train: 1460 rows, 81 columns
- Test: 1459 rows, 80 columns

## Approach
- Full EDA (scatter plots, boxplots, correlation heatmap)
- Missing value handling (median, mode, ordinal encoding)
- Feature Engineering (TotalSF, TotalBath, HouseAge, YearsSinceRemod, IsRemodeled)
- Log Transform (target + skewed features)
- One Hot Encoding (32 categorical columns)
- Trained 4 models (Linear, Ridge, Lasso, XGBoost)
- GridSearchCV hyperparameter tuning
- K-Fold Cross Validation

## Results
| Model | Val R² | Val RMSE | Gap |
|-------|--------|----------|-----|
| Linear Regression | 0.925 | 0.1181 | high |
| Ridge (tuned) | 0.924 | 0.1193 | medium |
| Lasso (tuned) | 0.926 | 0.1176 | low ✅ |
| XGBoost (tuned) | 0.905 | 0.1333 | medium |

## Best Model: Lasso (alpha=0.001)
- Val R² = 0.926
- Val RMSE = 0.1176
- Kaggle Public Score = 0.13089 (top 25%) 🏆

## Key Features
- OverallQual (0.82 correlation)
- TotalSF (0.81 correlation) - engineered feature
- GrLivArea (0.73 correlation)
- TotalBath (0.67 correlation) - engineered feature

## Setup
Download dataset from Kaggle and run notebook cells in order.
"""

with open('/content/ml-kaggle-projects/1-house-prices-regression/README.md', 'w') as f:
    f.write(readme)

print("README created!")
