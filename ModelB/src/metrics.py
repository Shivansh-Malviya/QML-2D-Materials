import numpy as np
import torch
from typing import Dict, List


def compute_mse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((preds - targets) ** 2))


def compute_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


def compute_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1 - ss_res / ss_tot)


def compute_mape(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((targets - preds) / (np.abs(targets) + eps)))) * 100


def compute_variance_ratio(preds: np.ndarray, targets: np.ndarray) -> float:
    std_t = np.std(targets)
    if std_t < 1e-9:
        return 0.0
    return float(np.std(preds) / std_t)


def compute_gradient_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Per-component gradient L2 norms for monitoring barren plateaus."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = float(param.grad.norm().item())
    return norms


def evaluate_all(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute all metrics at once."""
    return {
        'mse': compute_mse(preds, targets),
        'mae': compute_mae(preds, targets),
        'r2': compute_r2(preds, targets),
        'mape': compute_mape(preds, targets),
        'var_ratio': compute_variance_ratio(preds, targets),
    }
