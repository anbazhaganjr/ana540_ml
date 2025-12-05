# Regression Model Evaluation Summary

## Overview
Regression models were evaluated on `tradier_math.csv` to predict next-bar returns using leakage-safe features.

## Results

### Model Performance Metrics

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Ridge** | 0.013153 | 0.020289 | 0.000430 |
| **ElasticNet** | 0.013162 | 0.020294 | -0.000081 |
| **Random Forest Regressor** | 0.013160 | 0.020362 | -0.006806 |

### Interpretation

**MAE (Mean Absolute Error):**
- All models achieve similar MAE (~0.013 or ~1.3%)
- This means predictions are off by approximately 1.3% on average
- Ridge performs slightly better (0.013153)

**RMSE (Root Mean Squared Error):**
- All models have similar RMSE (~0.020 or ~2.0%)
- Ridge has the lowest RMSE (0.020289)
- RMSE penalizes larger errors more than MAE

**R² (R-squared):**
- All models show very low R² values (near zero or negative)
- This is **expected** for short-horizon equity prediction
- Low R² indicates high noise-to-signal ratio at single-bar horizons
- Ridge has the best (though still very low) R² = 0.000430

### Key Findings

1. **Linear models (Ridge, ElasticNet) perform similarly** - both achieve nearly identical MAE and RMSE
2. **Ridge slightly outperforms** - has the best R² and lowest RMSE
3. **Random Forest performs comparably** - similar MAE but slightly worse RMSE and negative R²
4. **All models struggle with prediction** - very low R² values reflect the inherent difficulty of predicting next-bar returns

### Methodology

**Features Used:**
- Return lags (1-5 bars): `ret_lag1` through `ret_lag5`
- Rolling volatility (20-bar): `roll_vol20`
- Rolling range (20-bar): `roll_range20`
- Volume surprise: `vol_surp`
- Range percentage: `range_pct`

**Data Split:**
- Time-aware train/test split (80/20)
- Training: 25,216 samples
- Test: 6,304 samples
- All features lagged by ≥1 bar (leakage-safe)

**Models:**
- Ridge: L2 regularization (alpha=1.0)
- ElasticNet: L1+L2 regularization (alpha=1.0, l1_ratio=0.5)
- Random Forest: 100 trees, max_depth=10

## Files Generated

- `regression_results.json`: Contains all model metrics
- `evaluate_regression_models.py`: Evaluation script (can be re-run)

## Dashboard Integration

The dashboard automatically loads these results from `regression_results.json` when available. The results are displayed in the "Regression Results" section of the Empirical Results tab.

## Notes

- Low R² values are expected for short-horizon equity prediction
- The empirical results document mentions this is a "noisy domain"
- Even modest predictive capability can be meaningful in this context
- Results align with the classification findings (modest but consistent lift)

