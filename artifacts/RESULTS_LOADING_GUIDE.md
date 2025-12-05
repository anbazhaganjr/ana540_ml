# Results Loading Guide for Dashboard

This guide explains what results data you need to load into the dashboard to display complete empirical results.

## Current Status

✅ **Already Loaded:**
- Random Forest Classification: ROC-AUC = 0.526, Accuracy = 0.519

❌ **Missing (Need to Load):**

### 1. Classification Model Metrics

**Location:** `load_empirical_results()` function in `dashboard.py`

**Required Data:**
- **Logistic Regression:**
  - `roc_auc`: float (e.g., 0.510)
  - `accuracy`: float (e.g., 0.505)

- **Decision Tree:**
  - `roc_auc`: float
  - `accuracy`: float

**How to Load:**
Update the `load_empirical_results()` function with your actual metrics from model evaluation.

---

### 2. Regression Model Metrics

**Location:** `load_empirical_results()` function in `dashboard.py`

**Required Data for each model (Ridge, ElasticNet, Random Forest Regressor):**
- `mae`: Mean Absolute Error (float)
- `rmse`: Root Mean Squared Error (float)
- `r2`: R-squared coefficient (float)

**Example:**
```python
'Random Forest Regressor': {
    'mae': 0.0025,  # Your actual MAE value
    'rmse': 0.0035,  # Your actual RMSE value
    'r2': 0.15,      # Your actual R² value
}
```

---

### 3. Confusion Matrix Data

**Location:** Confusion Matrix expander in Empirical Results tab

**Required Data Format:**
- CSV file or Python array with columns: `y_true`, `y_pred`
- Or a 2x2 numpy array: `[[TN, FP], [FN, TP]]`

**File Format (CSV):**
```csv
y_true,y_pred
0,0
1,1
0,1
1,0
...
```

**How to Load:**
Replace the placeholder in the "Confusion Matrix (Random Forest)" expander section.

---

### 4. ROC Curve Data

**Location:** ROC Curve expander in Empirical Results tab

**Required Data Format:**
- CSV file with columns: `y_true`, `y_pred_proba`
- Or arrays: `fpr` (False Positive Rate), `tpr` (True Positive Rate)

**File Format (CSV):**
```csv
y_true,y_pred_proba
0,0.45
1,0.62
0,0.38
1,0.71
...
```

**How to Load:**
Replace the placeholder ROC curve generation with actual data loading.

---

### 5. Residual Plots Data (Regression)

**Location:** Residual Plots expander in Empirical Results tab

**Required Data Format:**
- CSV file with columns: `y_true`, `y_pred`
- Where `y_true` = actual returns, `y_pred` = predicted returns

**File Format (CSV):**
```csv
y_true,y_pred
0.0012,0.0015
-0.0008,-0.0006
0.0023,0.0021
...
```

**How to Load:**
Replace the placeholder in the "Residual Plots" expander section.

---

### 6. Per-Symbol Performance Metrics

**Location:** Per-Symbol Performance Summary section

**Required Data Format:**
- CSV file with columns: `symbol`, `roc_auc` (for classification) or `mae` (for regression)

**File Format (CSV) - Classification:**
```csv
symbol,roc_auc,accuracy
AAPL,0.532,0.525
MSFT,0.521,0.518
GOOGL,0.528,0.522
...
```

**File Format (CSV) - Regression:**
```csv
symbol,mae,rmse,r2
AAPL,0.0023,0.0031,0.18
MSFT,0.0025,0.0034,0.15
GOOGL,0.0021,0.0029,0.22
...
```

---

## Quick Reference: What to Update in dashboard.py

### Option 1: Update `load_empirical_results()` function directly

```python
@st.cache_data
def load_empirical_results():
    classification_results = {
        'Random Forest': {
            'roc_auc': 0.526,  # ✅ Already set
            'accuracy': 0.519,  # ✅ Already set
        },
        'Logistic Regression': {
            'roc_auc': YOUR_VALUE_HERE,  # ⚠️ Add your value
            'accuracy': YOUR_VALUE_HERE,  # ⚠️ Add your value
        },
        'Decision Tree': {
            'roc_auc': YOUR_VALUE_HERE,  # ⚠️ Add your value
            'accuracy': YOUR_VALUE_HERE,  # ⚠️ Add your value
        }
    }
    
    regression_results = {
        'Random Forest Regressor': {
            'mae': YOUR_VALUE_HERE,  # ⚠️ Add your value
            'rmse': YOUR_VALUE_HERE,  # ⚠️ Add your value
            'r2': YOUR_VALUE_HERE,  # ⚠️ Add your value
        },
        'Ridge': {
            'mae': YOUR_VALUE_HERE,  # ⚠️ Add your value
            'rmse': YOUR_VALUE_HERE,  # ⚠️ Add your value
            'r2': YOUR_VALUE_HERE,  # ⚠️ Add your value
        },
        'ElasticNet': {
            'mae': YOUR_VALUE_HERE,  # ⚠️ Add your value
            'rmse': YOUR_VALUE_HERE,  # ⚠️ Add your value
            'r2': YOUR_VALUE_HERE,  # ⚠️ Add your value
        }
    }
    
    return classification_results, regression_results
```

### Option 2: Load from CSV/JSON files

You can modify `load_empirical_results()` to load from files:

```python
@st.cache_data
def load_empirical_results():
    # Load from CSV files
    try:
        class_metrics = pd.read_csv('results/classification_metrics.csv')
        reg_metrics = pd.read_csv('results/regression_metrics.csv')
        # Process and return...
    except:
        # Fallback to hardcoded values
        pass
```

---

## Priority Order for Submission

**Minimum Required (Dashboard works but shows placeholders):**
1. ✅ Random Forest classification metrics (already done)

**Recommended for Complete Dashboard:**
2. Logistic Regression and Decision Tree metrics
3. Regression metrics (MAE, RMSE, R²) for all three models

**Nice to Have (Enhanced Visualizations):**
4. Confusion matrix data
5. ROC curve data
6. Residual plots data
7. Per-symbol performance metrics

---

## Where to Get These Results

These should come from your model evaluation outputs:
- **Metrics**: From `sklearn.metrics` (roc_auc_score, accuracy_score, mean_absolute_error, etc.)
- **Predictions**: From `model.predict()` and `model.predict_proba()`
- **Per-symbol**: From evaluating models on each symbol separately

---

## Current Dashboard Status

The dashboard is **fully functional** and ready for submission even with placeholder values. The structure is complete, and you can add actual results data as it becomes available.

