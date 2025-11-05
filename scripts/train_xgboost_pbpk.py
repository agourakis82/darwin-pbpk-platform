#!/usr/bin/env python3
"""
🌲 XGBOOST BASELINE PARA ENSEMBLE HETEROGÊNEO
==============================================

Treina XGBoost para PBPK prediction como baseline clássico de ML.
XGBoost é conhecido por funcionar bem em datasets pequenos/missing data.

Author: Dr. Demetrios Chiuratto Agourakis
Date: October 28, 2025
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, Tuple
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

# ==================== PATHS ====================
KEC_DATA = BASE_DIR / 'data' / 'processed' / 'kec_dataset_split.pkl'
EMBEDDINGS = BASE_DIR / 'data' / 'embeddings' / 'rich_embeddings_788d.npz'
OUTPUT_DIR = BASE_DIR / 'results' / 'ensemble_heterogeneous' / 'xgboost'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🌲 XGBOOST PBPK BASELINE")
print("="*80)


# ==================== DATA LOADING ====================

def load_data():
    """Load KEC dataset with embeddings"""
    print("\n📁 Loading data...")
    
    # Load dataset
    with open(KEC_DATA, 'rb') as f:
        data = pickle.load(f)
    
    train_df = data['train']['dataframe']
    val_df = data['val']['dataframe']
    test_df = data['test']['dataframe']
    
    # Load embeddings
    embs = np.load(EMBEDDINGS)
    all_embeddings = embs['embeddings']
    
    n_train = len(train_df)
    n_val = len(val_df)
    
    train_emb = all_embeddings[:n_train]
    val_emb = all_embeddings[n_train:n_train+n_val]
    test_emb = all_embeddings[n_train+n_val:]
    
    print(f"  Train: {len(train_df)} samples, {train_emb.shape[1]} features")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Extract targets (no transforms for XGBoost - works better with raw values)
    def extract_targets(df):
        targets = df[['fu', 'vd', 'clearance']].values
        masks = ~df[['fu', 'vd', 'clearance']].isna().values
        
        # Replace NaN with -999 (XGBoost can handle this as missing)
        targets = np.where(np.isnan(targets), -999, targets)
        
        return targets, masks
    
    y_train, mask_train = extract_targets(train_df)
    y_val, mask_val = extract_targets(val_df)
    y_test, mask_test = extract_targets(test_df)
    
    return {
        'train': (train_emb, y_train, mask_train),
        'val': (val_emb, y_val, mask_val),
        'test': (test_emb, y_test, mask_test)
    }


# ==================== TRAINING ====================

def train_xgboost_model(X_train, y_train, mask_train, param_idx, param_name):
    """Train XGBoost for single parameter"""
    print(f"\n🌲 Training XGBoost for {param_name}...")
    
    # Filter valid samples
    valid_mask = mask_train[:, param_idx].astype(bool)
    X_valid = X_train[valid_mask]
    y_valid = y_train[valid_mask, param_idx]
    
    print(f"  Valid samples: {len(y_valid)} / {len(y_train)}")
    
    if len(y_valid) < 10:
        print(f"  ⚠️  Too few samples, skipping")
        return None
    
    # XGBoost parameters optimized for small datasets
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    # Train
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_valid,
        y_valid,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    # Evaluate on training set
    y_pred_train = model.predict(X_valid)
    r2_train = r2_score(y_valid, y_pred_train)
    
    print(f"  Train R²: {r2_train:.4f}")
    
    return model


def evaluate_model(models, X, y, masks, split_name="Val"):
    """Evaluate XGBoost models"""
    print(f"\n📊 Evaluating on {split_name}...")
    
    results = {}
    param_names = ['fu', 'vd', 'clearance']
    
    for i, (param_name, model) in enumerate(zip(param_names, models)):
        if model is None:
            print(f"  {param_name}: No model trained")
            results[param_name] = {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'n_samples': 0}
            continue
        
        # Filter valid samples
        valid_mask = masks[:, i].astype(bool)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask, i]
        
        if len(y_valid) < 5:
            print(f"  {param_name}: Too few samples")
            results[param_name] = {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'n_samples': len(y_valid)}
            continue
        
        # Predict
        y_pred = model.predict(X_valid)
        
        # Metrics
        r2 = r2_score(y_valid, y_pred)
        mae = mean_absolute_error(y_valid, y_pred)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        
        results[param_name] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'n_samples': int(len(y_valid))
        }
        
        print(f"  {param_name:10s}: R²={r2:.4f}, MAE={mae:.4f}, N={len(y_valid)}")
    
    # Average R²
    valid_r2 = [v['r2'] for v in results.values() if not np.isnan(v['r2'])]
    avg_r2 = np.mean(valid_r2) if len(valid_r2) > 0 else np.nan
    
    print(f"\n  📈 Average R²: {avg_r2:.4f}")
    
    return avg_r2, results


# ==================== MAIN ====================

def main():
    """Main execution"""
    
    # Load data
    data = load_data()
    X_train, y_train, mask_train = data['train']
    X_val, y_val, mask_val = data['val']
    X_test, y_test, mask_test = data['test']
    
    # Train separate models for each parameter
    param_names = ['fu', 'vd', 'clearance']
    models = []
    
    for i, param_name in enumerate(param_names):
        model = train_xgboost_model(X_train, y_train, mask_train, i, param_name)
        models.append(model)
    
    # Evaluate
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    
    val_r2, val_results = evaluate_model(models, X_val, y_val, mask_val, "Validation")
    test_r2, test_results = evaluate_model(models, X_test, y_test, mask_test, "Test")
    
    # Save models
    print(f"\n💾 Saving models...")
    for i, (param_name, model) in enumerate(zip(param_names, models)):
        if model is not None:
            model_path = OUTPUT_DIR / f'xgboost_{param_name}.json'
            model.save_model(model_path)
            print(f"  ✓ Saved: {model_path}")
    
    # Save results
    results = {
        'model': 'XGBoost',
        'validation': {
            'avg_r2': float(val_r2),
            'parameters': val_results
        },
        'test': {
            'avg_r2': float(test_r2),
            'parameters': test_results
        }
    }
    
    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved: {results_path}")
    
    # Summary
    print("\n" + "="*80)
    print("🎉 XGBOOST TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ Validation R²: {val_r2:.4f}")
    print(f"✅ Test R²:       {test_r2:.4f}")
    
    print("\n📊 Per-parameter (Test):")
    for param, metrics in test_results.items():
        print(f"  {param:10s}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, N={metrics['n_samples']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

