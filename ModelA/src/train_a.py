"""
Model A Training Script
=======================

Trains Graph + 28-Feature Hybrid + Angle Encoding model on 86 Mo materials
With checkpointing, variance ratio monitoring, and physics-informed loss
"""

import os

# Legacy OpenMP workaround retained from the original training script.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import time

try:
    from .mo_dataset import MoDataset
    from .model_a import ModelA, physics_informed_loss
    from .evaluation import compute_metrics
    from .config_a import CONFIG
except ImportError:  # pragma: no cover - script-style fallback
    from mo_dataset import MoDataset
    from model_a import ModelA, physics_informed_loss
    from evaluation import compute_metrics
    from config_a import CONFIG


def setup_directories(config):
    """Create output directories"""
    for dir_path in [
        config["output_dir"],
        config["checkpoint_dir"],
        config["plots_dir"],
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def train_epoch(model, materials, optimizer, config):
    """Train for one epoch"""
    model.train()

    epoch_loss = 0.0
    predictions = []
    targets = []
    loss_components = {"mse": 0.0, "positivity_violation": 0.0, "vdw_violation": 0.0}

    for material in materials:
        # Forward pass
        pred, features = model.forward(material)

        # Target
        target = torch.tensor(material["exfoliation_energy"], dtype=torch.float32)
        vdw_energy = torch.tensor(material["vdw_energy"], dtype=torch.float32)

        # Loss
        loss, components = physics_informed_loss(
            pred, target, vdw_energy, config=config
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v

        predictions.append(pred.item())
        targets.append(target.item())

    # Average losses
    n = len(materials)
    epoch_loss /= n
    for k in loss_components:
        loss_components[k] /= n

    return epoch_loss, np.array(predictions), np.array(targets), loss_components


def evaluate(model, materials, config):
    """Evaluate on test set"""
    model.eval()

    predictions = []
    targets = []
    features_list = []

    with torch.no_grad():
        for material in materials:
            pred, features = model.forward(material)

            predictions.append(pred.item())
            targets.append(material["exfoliation_energy"])
            features_list.append(features.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    return predictions, targets, np.array(features_list)


def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    # Regular checkpoint
    if epoch % config["checkpoint_every"] == 0:
        path = Path(config["checkpoint_dir"]) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"   Checkpoint saved: {path.name}")

    # Best model
    if is_best:
        path = Path(config["output_dir"]) / "best_model_a.pt"
        torch.save(checkpoint, path)
        print(f"   New best model saved!")


def plot_training_curves(train_losses, val_losses, train_vars, val_vars, config):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, label="Train Loss", marker="o")
    ax1.plot(epochs, val_losses, label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Variance ratio
    ax2.plot(epochs, train_vars, label="Train Var Ratio", marker="o")
    ax2.plot(epochs, val_vars, label="Val Var Ratio", marker="s")
    ax2.axhline(y=0.5, color="r", linestyle="--", label="Target (0.5)")
    ax2.axhline(y=0.1, color="orange", linestyle="--", label="Collapse threshold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Variance Ratio")
    ax2.set_title("Variance Ratio (PRIMARY DIAGNOSTIC)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(config["plots_dir"]) / "training_curves.png", dpi=150)
    print(f"   Training curves saved")


def main():
    """Main training loop"""
    print("\n" + "=" * 60)
    print("MODEL A TRAINING - Graph + 26-Feature Hybrid")
    print("=" * 60)

    # Setup
    setup_directories(CONFIG)

    # Load dataset
    print("\n Loading dataset...")
    dataset = MoDataset(cache=True)
    train_materials, test_materials = dataset.get_train_test_split(
        test_ratio=0.2, seed=CONFIG["seed"]
    )

    print(f"Train: {len(train_materials)}, Test: {len(test_materials)}")

    # Initialize model
    print("\n Initializing Model A...")
    model = ModelA(config=CONFIG)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_var_ratio": [],
        "val_var_ratio": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    # Training loop
    print(f"\n Starting training for {CONFIG['epochs']} epochs...")
    print(f"  WATCH FOR: Variance ratio > 0.5 (if < 0.1  collapse to mean!)\n")

    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"Epoch {epoch}/{CONFIG['epochs']}")

        # Train
        train_loss, train_preds, train_targets, loss_comp = train_epoch(
            model, train_materials, optimizer, CONFIG
        )

        # Evaluate
        val_preds, val_targets, _ = evaluate(model, test_materials, CONFIG)
        val_loss = np.mean((val_preds - val_targets) ** 2)

        # Compute variance ratios (PRIMARY DIAGNOSTIC!)
        train_var_ratio = (
            np.var(train_preds) / np.var(train_targets)
            if np.var(train_targets) > 0
            else 0
        )
        val_var_ratio = (
            np.var(val_preds) / np.var(val_targets) if np.var(val_targets) > 0 else 0
        )

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_var_ratio"].append(train_var_ratio)
        history["val_var_ratio"].append(val_var_ratio)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"  Train Var Ratio: {train_var_ratio:.4f} | Val Var Ratio: {val_var_ratio:.4f}"
        )

        # Check for collapse
        if val_var_ratio < 0.1:
            print(
                f"    WARNING: Variance ratio too low! Model may be collapsing to mean!"
            )
        elif val_var_ratio > 0.5:
            print(f"   Good variance ratio!")

        # Best model
        is_best = val_loss < history["best_val_loss"]
        if is_best:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch

        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch,
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_var_ratio": val_var_ratio,
            },
            CONFIG,
            is_best=is_best,
        )

        print()

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed / 60:.1f} minutes")
    print(
        f" Best epoch: {history['best_epoch']} (val loss: {history['best_val_loss']:.4f})"
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best model
    best_checkpoint = torch.load(Path(CONFIG["output_dir"]) / "best_model_a.pt")
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_preds, val_targets, features = evaluate(model, test_materials, CONFIG)

    metrics = compute_metrics(val_targets, val_preds, verbose=True)

    # Save results
    results = {
        "config": CONFIG,
        "history": history,
        "final_metrics": metrics,
        "predictions": val_preds.tolist(),
        "targets": val_targets.tolist(),
    }

    with open(Path(CONFIG["output_dir"]) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions & targets
    np.save(Path(CONFIG["output_dir"]) / "predictions.npy", val_preds)
    np.save(Path(CONFIG["output_dir"]) / "targets.npy", val_targets)
    np.save(Path(CONFIG["output_dir"]) / "features.npy", features)

    # Plot training curves
    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        history["train_var_ratio"],
        history["val_var_ratio"],
        CONFIG,
    )

    print(f"\n All results saved to: {CONFIG['output_dir']}")

    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if metrics["variance_ratio"] < 0.1:
        print(" FAILURE: Model collapsed to mean (var ratio < 0.1)")
        print("    Check if vdW energy loaded correctly")
        print("    Verify all 26 features present")
    elif metrics["variance_ratio"] < 0.5:
        print("  PARTIAL SUCCESS: Low variance (0.1-0.5)")
        print("    Model learning but not capturing full variance")
    else:
        print(" SUCCESS: Good variance ratio (> 0.5)!")
        rmse_target = 0.12
        if metrics["rmse"] < rmse_target:
            print(f" RMSE below target ({metrics['rmse']:.4f} < {rmse_target})")
        if metrics["r2"] > 0.7:
            print(f" Strong R ({metrics['r2']:.4f})")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
