import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_training_curves(history: Dict[str, List], save_dir: str, fold: int = None):
    """Generate epoch-evolving training plots. Called every epoch for live tracking."""
    suffix = f'_fold{fold}' if fold is not None else ''
    os.makedirs(save_dir, exist_ok=True)
    epochs = history['epoch']

    # Loss curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Progress{" (Fold " + str(fold) + ")" if fold else ""}', fontsize=14)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax.plot(epochs, history['val_mse'], 'r-', label='Val MSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Loss')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['val_mae'], 'g-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, history['val_r2'], 'm-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('R² Score')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history['val_mape'], 'c-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Mean Abs Percentage Error')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history['var_ratio'], 'orange')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Target = 1.0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Var Ratio')
    ax.set_title('Variance Ratio (std_pred / std_true)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient norms (if available)
    ax = axes[1, 2]
    if 'grad_norm_classical' in history:
        ax.plot(epochs, history['grad_norm_classical'], 'b-', label='Classical')
        ax.plot(epochs, history['grad_norm_quantum'], 'r-', label='Quantum')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_title('Gradient Norms')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Grad norms\nnot tracked', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Gradient Norms')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves{suffix}.png'), dpi=150)
    plt.close()


def plot_predictions(preds: np.ndarray, targets: np.ndarray, save_dir: str, target_name: str, fold: int = None):
    """Scatter plot of predictions vs ground truth."""
    suffix = f'_fold{fold}' if fold is not None else ''
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, label='y=x')
    ax.set_xlabel(f'True {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'Predictions vs Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pred_vs_true{suffix}.png'), dpi=150)
    plt.close()
