
# ============================================================
# Experimental module guard.
# This file is in ModelB/experimental/ and is untested.
# The training pipeline has a known data-leakage defect
# (PC2: normalization before split).
# Main retained implementation: ModelB/src/.
# ============================================================
import sys as _sys
if __name__ == "__main__":
    raise RuntimeError(
        "Experimental VQC module with known defects. See ModelB/experimental/README.md."
    )
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import os
import copy
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from .data import MaterialDataset, compute_normalization_stats, prepare_pyg_dataset
from .features import PhysicsFeatureExtractor
from .model import ModelB
from .metrics import evaluate_all, compute_gradient_norms
from .plotting import plot_training_curves, plot_predictions


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _build_optimizer(model: ModelB, config: dict) -> torch.optim.Adam:
    """Separate parameter groups: no weight decay on quantum rotations."""
    classical = [p for n, p in model.named_parameters() if "q_weights" not in n]
    quantum = [p for n, p in model.named_parameters() if "q_weights" in n]
    return torch.optim.Adam(
        [
            {"params": classical, "weight_decay": config["weight_decay"]},
            {"params": quantum, "weight_decay": 0.0},
        ],
        lr=config["lr"],
    )


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    history,
    path,
    best_loss=float("inf"),
    patience_counter=0,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": dict(history),
            "best_loss": best_loss,
            "patience_counter": patience_counter,
        },
        path,
    )


def _load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    history = defaultdict(list, ckpt["history"])
    best_loss = ckpt.get("best_loss", float("inf"))
    patience_counter = ckpt.get("patience_counter", 0)
    return ckpt["epoch"], history, best_loss, patience_counter


def train_fold(
    config: dict,
    train_graphs: list,
    val_graphs: list,
    phys_mean: torch.Tensor,
    phys_std: torch.Tensor,
    run_dir: str,
    fold: int = None,
    resume: bool = False,
):
    """Train a single fold using batched DataLoader.

    Args:
        train_graphs: list of pre-computed PyG Data objects (from prepare_pyg_dataset)
        val_graphs: same for validation
        phys_mean, phys_std: normalization stats
    """
    target_name = config["target_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config.get("batch_size", 32)

    model = ModelB(config, phys_mean.to(device), phys_std.to(device)).to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    fold_tag = f"fold{fold}" if fold is not None else "single"
    ckpt_path = os.path.join(config["checkpoint_dir"], f"checkpoint_{fold_tag}.pt")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    history = defaultdict(list)
    start_epoch = 1
    best_loss = float("inf")
    best_model_wts = None
    patience_counter = 0

    if resume and os.path.exists(ckpt_path):
        start_epoch, history, best_loss, patience_counter = _load_checkpoint(
            ckpt_path, model, optimizer, scheduler
        )
        start_epoch += 1
        print(
            f"[TRAIN] Resumed from epoch {start_epoch - 1} (best_loss={best_loss:.4f}, patience={patience_counter})"
        )

    # Build DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    n_train = len(train_graphs)
    n_val = len(val_graphs)
    n_batches = len(train_loader)
    print(
        f"[TRAIN] {fold_tag} | {device} | {target_name} | train={n_train} val={n_val} | batch={batch_size} ({n_batches} batches/epoch)"
    )
    print(f"[MODEL] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        grad_norms = {}

        for batch in train_loader:
            batch = batch.to(device)
            B = batch.y.shape[0]

            optimizer.zero_grad()
            preds = model(batch)  # [B, 1]
            loss = criterion(preds.squeeze(-1), batch.y.squeeze(-1))
            loss.backward()

            grad_norms = compute_gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item() * B
            n_samples += B

        epoch_loss /= max(n_samples, 1)

        # Aggregate gradient norms for logging
        classical_norm = float(
            np.mean([v for k, v in grad_norms.items() if "q_weights" not in k] or [0])
        )
        quantum_norm = float(
            np.mean([v for k, v in grad_norms.items() if "q_weights" in k] or [0])
        )

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch)
                all_preds.append(preds.squeeze(-1).cpu().numpy())
                all_targets.append(batch.y.squeeze(-1).cpu().numpy())

        preds_arr = np.concatenate(all_preds)
        targets_arr = np.concatenate(all_targets)
        val_metrics = (
            evaluate_all(preds_arr, targets_arr)
            if len(targets_arr) > 0
            else {
                "mse": 999.0,
                "mae": 999.0,
                "r2": 0.0,
                "mape": 999.0,
                "var_ratio": 0.0,
            }
        )

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_mape"].append(val_metrics["mape"])
        history["var_ratio"].append(val_metrics["var_ratio"])
        history["grad_norm_classical"].append(classical_norm)
        history["grad_norm_quantum"].append(quantum_norm)

        scheduler.step(val_metrics["mse"])

        print(
            f"E{epoch:03d} | loss={epoch_loss:.4f} | MSE={val_metrics['mse']:.4f} "
            f"| MAE={val_metrics['mae']:.4f} | R┬▓={val_metrics['r2']:.3f} "
            f"| VR={val_metrics['var_ratio']:.3f} | g_cl={classical_norm:.4f} g_q={quantum_norm:.4f}"
        )

        # Per-epoch training curves (overwrite each time for live tracking)
        plot_training_curves(history, run_dir, fold)

        # Prediction scatter less frequently (more expensive to render)
        if epoch % 5 == 0 or epoch == config["epochs"]:
            plot_predictions(preds_arr, targets_arr, run_dir, target_name, fold)

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                ckpt_path,
                best_loss,
                patience_counter,
            )

        # Early stopping
        if val_metrics["mse"] < best_loss:
            best_loss = val_metrics["mse"]
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"[TRAIN] Early stopping at epoch {epoch}")
                break

    # Save best and final checkpoint
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        history,
        ckpt_path,
        best_loss,
        patience_counter,
    )
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        torch.save(
            model.state_dict(), os.path.join(run_dir, f"best_model_{fold_tag}.pt")
        )

    # Final prediction plot
    if len(preds_arr) > 0:
        plot_predictions(preds_arr, targets_arr, run_dir, target_name, fold)

    # Save history
    with open(os.path.join(run_dir, f"history_{fold_tag}.json"), "w") as f:
        json.dump(dict(history), f, indent=2)

    return val_metrics


def train_model(config: dict):
    """Main entry point: k-fold CV or single split."""
    _set_seed(config["seed"])

    raw_path = config.get(
        "raw_data_path", os.path.join("data", "raw", "2dmatpedia_full.json")
    )
    cache_path = config["data_path"]

    if os.path.exists(cache_path):
        ds = MaterialDataset(cache_path=cache_path)
    else:
        ds = MaterialDataset(raw_path=raw_path)
        ds.save_cache(cache_path)

    # Always filter for valid target values (cache may hold all materials)
    ds.filter_valid(config["target_name"])

    # Element filter (opt-in via --filter_elements)
    if config.get("filter_elements"):
        ds.filter_by_elements(config["filter_elements"])

    ds.print_stats(verbose=config.get("verbose", True))

    # Pre-compute normalization stats from all materials
    extractor = PhysicsFeatureExtractor(config.get("feature_switches"))
    p_mean, p_std = compute_normalization_stats(
        ds.materials, config["target_name"], extractor
    )

    run_tag = f"{config['target_name']}_q{config['n_qubits']}_{int(time.time())}"
    run_dir = os.path.join(config["run_dir_base"], run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    k = config.get("k_folds", 1)
    if k > 1:
        folds = ds.get_kfold_splits(k=k, seed=config["seed"])
        all_metrics = []
        for i, (train_mats, val_mats) in enumerate(folds):
            print(f'\n{"="*60}\n  Fold {i+1}/{k}\n{"="*60}')
            # Pre-compute PyG datasets per fold
            print("[DATA] Pre-computing train graphs...")
            train_graphs = prepare_pyg_dataset(
                train_mats, config["target_name"], extractor, config["cutoff"]
            )
            print("[DATA] Pre-computing val graphs...")
            val_graphs = prepare_pyg_dataset(
                val_mats, config["target_name"], extractor, config["cutoff"]
            )

            metrics = train_fold(
                config,
                train_graphs,
                val_graphs,
                p_mean,
                p_std,
                run_dir,
                fold=i + 1,
                resume=config.get("resume", False),
            )
            all_metrics.append(metrics)

        # Summary
        print(f'\n{"="*60}\n  K-Fold Summary\n{"="*60}')
        for key in ["mse", "mae", "r2", "mape", "var_ratio"]:
            vals = [m[key] for m in all_metrics]
            print(f"  {key}: {np.mean(vals):.4f} ┬▒ {np.std(vals):.4f}")

        with open(os.path.join(run_dir, "kfold_summary.json"), "w") as f:
            json.dump(
                {
                    k: {
                        "mean": float(np.mean([m[k] for m in all_metrics])),
                        "std": float(np.std([m[k] for m in all_metrics])),
                    }
                    for k in all_metrics[0]
                },
                f,
                indent=2,
            )
    else:
        train_mats, val_mats = ds.get_split(seed=config["seed"])
        print("[DATA] Pre-computing train graphs...")
        train_graphs = prepare_pyg_dataset(
            train_mats, config["target_name"], extractor, config["cutoff"]
        )
        print("[DATA] Pre-computing val graphs...")
        val_graphs = prepare_pyg_dataset(
            val_mats, config["target_name"], extractor, config["cutoff"]
        )

        train_fold(
            config,
            train_graphs,
            val_graphs,
            p_mean,
            p_std,
            run_dir,
            resume=config.get("resume", False),
        )

    print(f"\n[DONE] Results saved to {run_dir}")


if __name__ == "__main__":
    from src.config import get_default_config

    config = get_default_config()
    train_model(config)

