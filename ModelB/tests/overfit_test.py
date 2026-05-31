"""R3 â€” Overfit Micro-Batch Test

Proves the VQC ansatz can learn by overfitting on 2 materials.
Pass criterion: RÂ² > 0.99, MSE < 1e-4 within 50 epochs.
If this fails, the architecture is fundamentally incapable of learning.
"""
import sys
import os
import json
import numpy as np
import torch

# Ensure parent is importable for src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import get_default_config
from src.data import MaterialDataset, compute_normalization_stats, prepare_pyg_dataset
from src.features import PhysicsFeatureExtractor
from src.model import ModelB
from src.metrics import evaluate_all


def overfit_test(config_overrides: dict = None):
    """Run the overfit test: 2 samples, 50 epochs, must memorize perfectly."""
    config = get_default_config()
    config.update({
        'epochs': 50,
        'lr': 1e-3,
        'patience': 9999,       # No early stopping
        'accum_steps': 1,       # No grad accumulation for 2 samples
        'n_qubits': 8,          # Increased for 10 samples
        'n_vqc_layers': 3,
        'n_features_compressed': 16,
        'schnet_hidden': 32,    # Tiny SchNet for speed
        'schnet_filters': 16,
        'schnet_interactions': 2,
    })
    if config_overrides:
        config.update(config_overrides)

    target = config['target_name']
    n_samples = config.get('n_samples', 2)
    print(f'\n{"="*60}')
    print(f'  OVERFIT TEST â€” Target: {target}')
    print(f'  Goal: R2 > 0.99, MSE < 1e-4 on {n_samples} samples in 50 epochs')
    print(f'{"="*60}\n')

    # Load data (just need 2 valid materials)
    raw_path = config.get('raw_data_path')
    cache_path = config.get('data_path')
    if os.path.exists(cache_path):
        ds = MaterialDataset(cache_path=cache_path)
    elif os.path.exists(raw_path):
        ds = MaterialDataset(raw_path=raw_path)
    else:
        print(f'[OVERFIT] SKIP: No dataset found at {raw_path} or {cache_path}. Dataset-backed check skipped.')
        return None

    # Find N materials with valid and DISTINCT target values
    micro_batch = []
    seen_vals = set()
    for mat in ds.materials:
        val = mat.get(target)
        if val is not None:
            try:
                v = float(val)
                if not np.isnan(v) and v not in seen_vals:
                    micro_batch.append(mat)
                    seen_vals.add(v)
                if len(micro_batch) >= n_samples:
                    break
            except (TypeError, ValueError):
                continue

    if len(micro_batch) < n_samples:
        print(f'[OVERFIT] ERROR: Could not find {n_samples} materials with valid distinct {target}')
        return False

    target_vals = [float(m[target]) for m in micro_batch]
    print(f'[OVERFIT] Micro-batch target values: {target_vals}')

    # Build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = PhysicsFeatureExtractor(config.get('feature_switches'))
    p_mean, p_std = compute_normalization_stats(micro_batch, target, extractor)
    model = ModelB(config, p_mean.to(device), p_std.to(device)).to(device)

    # Build PyG graphs for micro-batch (model requires PyG Data, not raw dicts)
    cutoff = config.get('cutoff', 5.0)
    micro_graphs = prepare_pyg_dataset(micro_batch, target, extractor, cutoff=cutoff)
    if len(micro_graphs) == 0:
        print('[OVERFIT] ERROR: prepare_pyg_dataset returned empty list')
        return False

    # Separate param groups (R2: no weight decay on quantum params)
    classical = [p for n, p in model.named_parameters() if 'q_weights' not in n]
    quantum = [p for n, p in model.named_parameters() if 'q_weights' in n]
    optimizer = torch.optim.Adam([
        {'params': classical, 'weight_decay': 1e-3},
        {'params': quantum, 'weight_decay': 0.0},
    ], lr=config['lr'])
    criterion = torch.nn.MSELoss()

    # Train using PyG graphs (not raw dicts)
    best_r2 = -999.0
    best_mse = 999.0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for graph in micro_graphs:
            graph = graph.to(device)
            t = graph.y.view(-1, 1).to(device)
            pred = model(graph)
            loss = criterion(pred, t)
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()

        # Evaluate
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for graph in micro_graphs:
                graph = graph.to(device)
                p = model(graph).item()
                preds.append(p)
                tgts.append(graph.y.item())

        metrics = evaluate_all(np.array(preds), np.array(tgts))
        r2 = metrics['r2']
        mse = metrics['mse']
        if r2 > best_r2:
            best_r2 = r2
        if mse < best_mse:
            best_mse = mse

        if epoch % 10 == 0 or epoch == 1:
            print(f'  E{epoch:03d} | loss={total_loss/2:.6f} | MSE={mse:.6f} | R2={r2:.4f}')

        # Early pass
        if r2 > 0.99 and mse < 1e-4:
            print(f'\n  [PASS] OVERFIT TEST PASSED at epoch {epoch}')
            print(f'     R2={r2:.4f}, MSE={mse:.6f}')
            return True

    # Didn't pass
    print(f'\n  [FAIL] OVERFIT TEST FAILED after {config["epochs"]} epochs')
    print(f'     Best R2={best_r2:.4f}, Best MSE={best_mse:.6f}')
    print(f'     Required: R2 > 0.99, MSE < 1e-4')
    print(f'\n  This means the architecture CANNOT learn even {n_samples} samples.')
    print(f'  Root causes to investigate:')
    print(f'    1. Barren plateau (check gradient norms)')
    print(f'    2. Compressor saturation (tanh dying gradients)')
    print(f'    3. VQC expressivity insufficient for this target')
    return False


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Q2DM_v2 Overfit Micro-Batch Test')
    p.add_argument('--target_name', type=str, default='bandgap')
    p.add_argument('--raw_data_path', type=str, default='data/raw/2dmatpedia_full.json')
    p.add_argument('--data_path', type=str, default='data/processed/dataset.npz')
    p.add_argument('--n_qubits', type=int, default=8)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--n_samples', type=int, default=2)
    args = p.parse_args()

    passed = overfit_test(vars(args))
    sys.exit(0 if passed is None or passed else 1)

