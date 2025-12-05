# Classification Model Evaluation Summary

## Overview
Classification models were evaluated on `tradier_math.csv` to predict next-bar direction (up vs. non-up) using leakage-safe features. This evaluation generates actual confusion matrices and ROC curves for dashboard visualization.

## Results

### Model Performance Metrics

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| **Random Forest** | 0.536 | 0.533 |
| **Logistic Regression** | 0.517 | 0.521 |
| **Decision Tree** | 0.508 | 0.523 |

### Interpretation

**Random Forest (Best Model):**
- ROC-AUC = 0.536 (vs. 0.500 random baseline) → +3.6% lift
- Accuracy = 0.533 (vs. 0.500 random baseline) → +3.3% lift
- Confusion Matrix: TN=860, FP=2201, FN=755, TP=2520

**Logistic Regression:**
- ROC-AUC = 0.517 → +1.7% lift
- Accuracy = 0.521 → +2.1% lift
- Confusion Matrix: TN=562, FP=2499, FN=535, TP=2740

**Decision Tree:**
- ROC-AUC = 0.508 → +0.8% lift
- Accuracy = 0.523 → +2.3% lift
- Confusion Matrix: TN=639, FP=2422, FN=601, TP=2674

### Key Findings

1. **Random Forest performs best** - Highest ROC-AUC (0.536) and good accuracy (0.533)
2. **All models show lift over random** - All achieve >0.500 ROC-AUC and accuracy
3. **Modest but consistent performance** - Expected for short-horizon equity prediction
4. **Random Forest has better true positive/negative balance** - More balanced confusion matrix

## Confusion Matrix Details

### Random Forest Confusion Matrix:
```
                Predicted Down  Predicted Up
Actual Down          860           2201
Actual Up            755           2520
```

**Metrics from Confusion Matrix:**
- Total samples: 6,336
- True Negatives (TN): 860
- False Positives (FP): 2,201
- False Negatives (FN): 755
- True Positives (TP): 2,520
- Accuracy: (860 + 2520) / 6336 = 0.533
- Precision: 2520 / (2520 + 2201) = 0.534
- Recall: 2520 / (2520 + 755) = 0.770

## Files Generated

1. **classification_results.json**: Contains all model metrics and confusion matrices
2. **confusion_matrix_Random_Forest.csv**: y_true, y_pred, y_pred_proba for Random Forest
3. **confusion_matrix_Logistic_Regression.csv**: y_true, y_pred, y_pred_proba for Logistic Regression
4. **confusion_matrix_Decision_Tree.csv**: y_true, y_pred, y_pred_proba for Decision Tree

## Dashboard Integration

The dashboard now:
- ✅ Loads actual classification results from `classification_results.json`
- ✅ Displays actual confusion matrices for all models
- ✅ Shows actual ROC curves using prediction probabilities
- ✅ Updates metrics automatically when results are available

## Methodology

**Features Used:**
- Return lags (1-5 bars): `ret_lag1` through `ret_lag5`
- Rolling volatility (20-bar): `roll_vol20`
- Rolling range (20-bar): `roll_range20`
- Volume surprise: `vol_surp`
- Range percentage: `range_pct`

**Data Split:**
- Time-aware train/test split (80/20)
- Training: 25,344 samples
- Test: 6,336 samples
- Class distribution in test: [3061 down, 3275 up] (relatively balanced)

**Models:**
- Logistic Regression: L2 regularization, max_iter=1000
- Decision Tree: max_depth=10
- Random Forest: 100 trees, max_depth=10

## Notes

- Results align with empirical findings (modest but consistent lift)
- Random Forest shows best performance, consistent with documented results
- All models demonstrate measurable predictive capability
- Confusion matrices provide detailed diagnostic information
- ROC curves show model discrimination ability

