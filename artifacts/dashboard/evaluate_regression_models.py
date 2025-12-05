"""
Regression Model Evaluation Script
Evaluates Ridge, ElasticNet, and Random Forest Regressor on tradier_math.csv
to predict next-bar returns.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create leakage-safe features for regression"""
    df = df.copy()
    df = df.sort_values(['Symbol', 'timestamp'])
    
    # Calculate returns (lagged appropriately)
    df['ret'] = df.groupby('Symbol')['close'].pct_change()
    
    # Create lagged returns (all lagged by at least 1 bar)
    for lag in [1, 2, 3, 4, 5]:
        df[f'ret_lag{lag}'] = df.groupby('Symbol')['ret'].shift(lag)
    
    # Rolling statistics (20-bar window)
    df['roll_vol20'] = df.groupby('Symbol')['ret'].transform(
        lambda x: x.rolling(window=20, min_periods=1).std()
    ).shift(1)  # Lag by 1
    
    df['roll_range20'] = df.groupby('Symbol')['high'].transform(
        lambda x: x.rolling(window=20, min_periods=1).max()
    ) - df.groupby('Symbol')['low'].transform(
        lambda x: x.rolling(window=20, min_periods=1).min()
    )
    df['roll_range20'] = df.groupby('Symbol')['roll_range20'].shift(1)
    
    # Volume features
    df['vol_surp'] = df.groupby('Symbol')['volume'].transform(
        lambda x: (x - x.rolling(window=20, min_periods=1).mean()) / (x.rolling(window=20, min_periods=1).mean() + 1e-8)
    ).shift(1)
    
    # Price-based features
    df['range_pct'] = ((df['high'] - df['low']) / df['close']).shift(1)
    
    # Target: next-bar return (what we're predicting)
    df['next_ret'] = df.groupby('Symbol')['ret'].shift(-1)
    
    return df

def prepare_data(df, test_size=0.2):
    """Prepare data with time-aware train/test split"""
    df = df.copy()
    df = df.sort_values(['Symbol', 'timestamp'])
    
    # Remove rows with NaN in target or key features
    feature_cols = ['ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag4', 'ret_lag5',
                    'roll_vol20', 'roll_range20', 'vol_surp', 'range_pct']
    
    df = df.dropna(subset=['next_ret'] + feature_cols)
    
    if len(df) == 0:
        return None, None, None, None
    
    # Time-aware split: use earlier data for training, later for testing
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Prepare features and target
    X_train = train_df[feature_cols].values
    y_train = train_df['next_ret'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['next_ret'].values
    
    # Remove any remaining NaN
    train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    return X_train, X_test, y_train, y_test

def evaluate_models(X_train, X_test, y_train, y_test):
    """Evaluate regression models and return metrics"""
    results = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge Regression
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    results['Ridge'] = {
        'mae': mean_absolute_error(y_test, y_pred_ridge),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'r2': r2_score(y_test, y_pred_ridge)
    }
    
    # ElasticNet
    print("Training ElasticNet...")
    elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=1000)
    elastic.fit(X_train_scaled, y_train)
    y_pred_elastic = elastic.predict(X_test_scaled)
    
    results['ElasticNet'] = {
        'mae': mean_absolute_error(y_test, y_pred_elastic),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_elastic)),
        'r2': r2_score(y_test, y_pred_elastic)
    }
    
    # Random Forest Regressor
    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    y_pred_rf = rf.predict(X_test)
    
    results['Random Forest Regressor'] = {
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'r2': r2_score(y_test, y_pred_rf)
    }
    
    return results

def main():
    print("Loading data...")
    df = pd.read_csv('tradier_math.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['Symbol', 'timestamp'])
    
    print(f"Loaded {len(df)} records for {df['Symbol'].nunique()} symbols")
    
    print("Creating features...")
    df_features = create_features(df)
    
    print("Preparing train/test split...")
    X_train, X_test, y_train, y_test = prepare_data(df_features, test_size=0.2)
    
    if X_train is None or len(X_train) == 0:
        print("Error: No valid data after preprocessing")
        return
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nEvaluating models...")
    results = evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("REGRESSION MODEL RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R²:   {metrics['r2']:.6f}")
    
    # Save results to JSON for dashboard
    import json
    with open('regression_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Results saved to regression_results.json")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = main()

