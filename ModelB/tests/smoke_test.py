"""Smoke test: compact end-to-end training on a tiny PyG batch."""

import sys, os, random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import numpy as np
import pytest
from torch_geometric.loader import DataLoader

from src.config import get_default_config
from src.data import MaterialDataset, compute_normalization_stats, prepare_pyg_dataset
from src.model import ModelB
from src.features import PhysicsFeatureExtractor
from src.metrics import evaluate_all

config = get_default_config()
config["n_qubits"] = 2
config["n_vqc_layers"] = 1
config["epochs"] = 3
config["target_name"] = "bandgap"

repo_root = Path(__file__).resolve().parent.parent.parent
data_path = repo_root / "ModelB" / "data" / "raw" / "2dmatpedia_full.json"
if not data_path.exists():
    pytest.skip("ModelB raw dataset is absent; dataset-backed smoke training skipped.", allow_module_level=True)
ds = MaterialDataset(raw_path=str(data_path))
ds.filter_valid("bandgap")
ds.print_stats()

random.seed(42)
sample = random.sample(ds.materials, 20)
train, val = sample[:15], sample[15:]

ext = PhysicsFeatureExtractor()
pm, ps = compute_normalization_stats(train, "bandgap", ext)
model = ModelB(config, pm, ps)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model params: {param_count:,}")

train_graphs = prepare_pyg_dataset(train, "bandgap", ext, cutoff=config["cutoff"])
val_graphs = prepare_pyg_dataset(val, "bandgap", ext, cutoff=config["cutoff"])
train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(1, 4):
    model.train()
    ep_loss = 0.0
    for batch in train_loader:
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ep_loss += loss.item()
    avg = ep_loss / max(1, len(train_loader))
    print(f"Epoch {ep}: loss={avg:.4f}")

model.eval()
preds, targs = [], []
with torch.no_grad():
    for batch in val_loader:
        batch_preds = model(batch).view(-1).cpu().numpy()
        batch_targs = batch.y.view(-1).cpu().numpy()
        preds.extend(batch_preds.tolist())
        targs.extend(batch_targs.tolist())

m = evaluate_all(np.array(preds), np.array(targs))
print(f"Val: MSE={m['mse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.3f}")

# Real assertions
assert all(np.isfinite(preds)), "Predictions must be finite"
assert all(np.isfinite(targs)), "Targets must be finite"
assert m["mse"] < 1e6, f"MSE unreasonably large: {m['mse']}"
assert np.isfinite(m["r2"]), f"R² is not finite: {m['r2']}"
print("SMOKE TEST PASSED")
