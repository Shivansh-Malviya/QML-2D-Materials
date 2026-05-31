"""
Evaluate Specific Checkpoint
============================
Loads a specific checkpoint (e.g. Epoch 50) to bypass "Best Loss" selection.
Generates Parity Plot, Residual Plot, and Metrics.
"""

import sys

sys.path.append("../shared")
import os

# PROACTIVE FIX: Handle OpenMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

try:
    from .mo_dataset import MoDataset
    from .model_a import ModelA
    from .config_a import CONFIG
    from .evaluation import compute_metrics
except ImportError:  # pragma: no cover - script-style fallback
    from mo_dataset import MoDataset
    from model_a import ModelA
    from config_a import CONFIG
    from evaluation import compute_metrics
from scipy.stats import spearmanr

# Override config to match training (15 qubits)
CONFIG["n_qubits"] = 15
CONFIG["n_total_features"] = 15
CONFIG["use_pca"] = True
CONFIG["pca_components"] = 15

CHECKPOINT_PATH = "results/checkpoints/checkpoint_epoch_50.pt"
OUTPUT_DIR = "results"


def evaluate_checkpoint():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # Load dataset
    dataset = MoDataset(cache=True)
    _, test_materials = dataset.get_train_test_split(
        test_ratio=0.2, seed=CONFIG["seed"]
    )

    # Load model
    model = ModelA(config=CONFIG)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from Epoch {checkpoint['epoch']}")

    # Predict
    predictions = []
    targets = []

    with torch.no_grad():
        for material in test_materials:
            pred, _ = model.forward(material)
            predictions.append(pred.item())
            targets.append(material["exfoliation_energy"])

    preds = np.array(predictions)
    targs = np.array(targets)

    # Compute Metrics
    metrics = compute_metrics(targs, preds, verbose=True)
    spearman = spearmanr(targs, preds)
    metrics["spearman_r"] = spearman.correlation
    print(f"Spearman R: {spearman.correlation:.4f}")

    # --- PLOTS ---

    # 1. Parity Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(targs, preds, alpha=0.7, c="blue", edgecolors="k")

    # Perfect line
    lims = [min(min(targs), min(preds)), max(max(targs), max(preds))]
    plt.plot(lims, lims, "r--", alpha=0.75, zorder=0)

    plt.xlabel("True Exfoliation Energy (eV/atom)")
    plt.ylabel("Predicted Energy (eV/atom)")
    plt.title(
        f"Model A (Epoch 50) - Parity Plot\nR²={metrics['r2']:.2f}, VarRatio={metrics['variance_ratio']:.2f}"
    )
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.savefig(f"{OUTPUT_DIR}/parity_plot_epoch50.png", dpi=300)
    print("Parity plot saved.")

    # 2. Residual Plot
    residuals = preds - targs
    plt.figure(figsize=(8, 4))
    plt.scatter(targs, residuals, alpha=0.7, c="green", edgecolors="k")
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("True Value")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residuals Analysis")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/residuals_plot_epoch50.png", dpi=300)
    print("Residuals plot saved.")


if __name__ == "__main__":
    evaluate_checkpoint()
