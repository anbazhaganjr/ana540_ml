"""
Classification Model Evaluation Script
Evaluates Logistic Regression, Decision Tree, and Random Forest on tradier_math.csv
to predict next-bar direction (up vs. non-up) and generates confusion matrix data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create leakage-safe features for classification"""
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
    
    # Target: next-bar direction (1 = up, 0 = non-up)
    df['next_ret'] = df.groupby('Symbol')['ret'].shift(-1)
    df['next_direction'] = (df['next_ret'] > 0).astype(int)
    
    return df

def prepare_data(df, test_size=0.2):
    """Prepare data with time-aware train/test split"""
    df = df.copy()
    df = df.sort_values(['Symbol', 'timestamp'])
    
    # Remove rows with NaN in target or key features
    feature_cols = ['ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag4', 'ret_lag5',
                    'roll_vol20', 'roll_range20', 'vol_surp', 'range_pct']
    
    df = df.dropna(subset=['next_direction'] + feature_cols)
    
    if len(df) == 0:
        return None, None, None, None
    
    # Time-aware split: use earlier data for training, later for testing
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Prepare features and target
    X_train = train_df[feature_cols].values
    y_train = train_df['next_direction'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['next_direction'].values
    
    # Remove any remaining NaN
    train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    return X_train, X_test, y_train, y_test

def evaluate_models(X_train, X_test, y_train, y_test):
    """Evaluate classification models and return metrics and predictions"""
    results = {}
    predictions = {}
    
    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    results['Logistic Regression'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba_lr),
        'accuracy': accuracy_score(y_test, y_pred_lr)
    }
    predictions['Logistic Regression'] = {
        'y_true': y_test,
        'y_pred': y_pred_lr,
        'y_pred_proba': y_pred_proba_lr
    }
    
    # Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)  # DT doesn't need scaling
    y_pred_dt = dt.predict(X_test)
    y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]
    
    results['Decision Tree'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba_dt),
        'accuracy': accuracy_score(y_test, y_pred_dt)
    }
    predictions['Decision Tree'] = {
        'y_true': y_test,
        'y_pred': y_pred_dt,
        'y_pred_proba': y_pred_proba_dt
    }
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf),
        'accuracy': accuracy_score(y_test, y_pred_rf)
    }
    predictions['Random Forest'] = {
        'y_true': y_test,
        'y_pred': y_pred_rf,
        'y_pred_proba': y_pred_proba_rf
    }
    
    return results, predictions

def save_confusion_matrices(predictions):
    """Save confusion matrices for each model"""
    confusion_matrices = {}
    
    for model_name, pred_data in predictions.items():
        cm = confusion_matrix(pred_data['y_true'], pred_data['y_pred'])
        confusion_matrices[model_name] = cm.tolist()
        
        # Also save as CSV for easy loading
        cm_df = pd.DataFrame({
            'y_true': pred_data['y_true'],
            'y_pred': pred_data['y_pred'],
            'y_pred_proba': pred_data['y_pred_proba']
        })
        cm_df.to_csv(f'confusion_matrix_{model_name.replace(" ", "_")}.csv', index=False)
        print(f"Saved predictions to confusion_matrix_{model_name.replace(' ', '_')}.csv")
    
    return confusion_matrices

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
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    print("\nEvaluating models...")
    results, predictions = evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("CLASSIFICATION MODEL RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.6f}")
        print(f"  Accuracy: {metrics['accuracy']:.6f}")
    
    # Save confusion matrices
    print("\nSaving confusion matrices and predictions...")
    confusion_matrices = save_confusion_matrices(predictions)
    
    # Save results to JSON for dashboard
    import json
    output = {
        'results': results,
        'confusion_matrices': confusion_matrices
    }
    
    with open('classification_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*60)
    print("Results saved to classification_results.json")
    print("Confusion matrix CSV files saved for each model")
    print("="*60)
    
    return results, predictions

if __name__ == "__main__":
    results, predictions = main()

