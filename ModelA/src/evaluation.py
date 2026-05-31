"""
Evaluation Metrics
==================

Compute metrics for Q2DM models with emphasis on variance ratio (primary diagnostic)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def compute_metrics(y_true, y_pred, verbose=True):
    """
    Compute all evaluation metrics
    
    PRIMARY: variance_ratio (must be > 0.5 to avoid collapse to mean!)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    # PRIMARY DIAGNOSTIC: Variance ratio
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    metrics['variance_ratio'] = var_pred / var_true if var_true > 0 else 0.0
    
    # Standard regression metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Physics compliance
    metrics['violation_rate'] = (y_pred < 0).mean()  # Should be 0%
    metrics['mean_error'] = np.mean(y_pred - y_true)
    metrics['std_error'] = np.std(y_pred - y_true)
    
    # Distribution info
    metrics['var_true'] = var_true
    metrics['var_pred'] = var_pred
    metrics['mean_true'] = np.mean(y_true)
    metrics['mean_pred'] = np.mean(y_pred)
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"\n PRIMARY DIAGNOSTIC:")
        print(f"  Variance Ratio: {metrics['variance_ratio']:.4f}")
        
        if metrics['variance_ratio'] < 0.1:
            print(f"   COLLAPSED TO MEAN! (ratio < 0.1)")
        elif metrics['variance_ratio'] < 0.5:
            print(f"    Weak variance (ratio < 0.5)")
        else:
            print(f"   Good variance (ratio >= 0.5)")
        
        print(f"\n Regression Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f} eV/atom")
        print(f"  MAE: {metrics['mae']:.4f} eV/atom")
        print(f"  R: {metrics['r2']:.4f}")
        
        print(f"\n  Physics Compliance:")
        print(f"  Violation Rate: {100*metrics['violation_rate']:.1f}% (negative predictions)")
        
        print(f"\n Distributions:")
        print(f"  True: mean={metrics['mean_true']:.4f}, var={metrics['var_true']:.4f}")
        print(f"  Pred: mean={metrics['mean_pred']:.4f}, var={metrics['var_pred']:.4f}")
        print("="*60 + "\n")
    
    return metrics


def check_collapse_to_mean(y_pred, threshold=0.01):
    """
    Quick check if predictions collapsed to mean
    
    Returns True if collapsed (variance ratio < threshold)
    """
    var_pred = np.var(y_pred)
    return var_pred < threshold


def compute_feature_importance_shap(model, X, feature_names=None):
    """
    Compute SHAP feature importance (placeholder)
    
    TODO: Implement SHAP analysis for quantum models
    """
    # Placeholder
    print("Feature importance analysis not yet implemented")
    return None
