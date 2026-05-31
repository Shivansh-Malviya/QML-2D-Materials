
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
import argparse
from typing import Dict


def get_default_config() -> Dict:
    return {
        # Data
        "raw_data_path": "data/raw/2dmatpedia_full.json",
        "data_path": "data/processed/dataset.npz",
        "target_name": "bandgap",
        # SchNet
        "schnet_hidden": 128,
        "schnet_filters": 64,
        "schnet_interactions": 3,
        "cutoff": 5.0,
        # Features
        "n_features_compressed": 32,
        "feature_switches": None,  # Dict[str, bool] or None for all-on
        # Quantum ΓÇö conservative baseline within Cerezo BP bound
        "n_qubits": 8,  # 2^8=256 Hilbert space; safe per McClean 2018
        "n_vqc_layers": 3,  # = log2(8) per Cerezo 2021 local-cost bound
        "quantum_backend": "simulator",  # simulator|lightning_gpu|lightning_cpu
        "diff_method": "backprop",  # auto-overridden in model.py per backend
        # Training
        "lr": 1e-3,
        "weight_decay": 1e-3,  # Classical params only (quantum gets 0.0)
        "epochs": 60,
        "patience": 15,
        "batch_size": 32,  # Graph batch size for DataLoader
        "grad_clip": 1.0,
        "k_folds": 5,
        "seed": 42,
        "resume": False,
        "verbose": True,  # Print dataset stats on load
        "overfit_test": False,  # R3: run overfit microbatch test instead of training
        "filter_elements": None,  # List[str] ΓÇö keep only materials containing ALL listed elements
        # Paths
        "checkpoint_dir": "models",
        "run_dir_base": "Runs",
    }


def parse_args() -> Dict:
    """Parse CLI args, filling defaults from config."""
    config = get_default_config()
    p = argparse.ArgumentParser(description="Q2DM_v2 ΓÇö Hybrid Graph-QNN Training")

    p.add_argument("--target_name", type=str, default=config["target_name"])
    p.add_argument("--n_qubits", type=int, default=config["n_qubits"])
    p.add_argument("--n_vqc_layers", type=int, default=config["n_vqc_layers"])
    p.add_argument("--lr", type=float, default=config["lr"])
    p.add_argument("--epochs", type=int, default=config["epochs"])
    p.add_argument("--k_folds", type=int, default=config["k_folds"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=config["seed"])
    p.add_argument(
        "--quantum_backend",
        type=str,
        default=config["quantum_backend"],
        choices=["simulator", "lightning_cpu", "lightning_gpu", "ibm", "ionq"],
    )
    p.add_argument(
        "--diff_method",
        type=str,
        default=config["diff_method"],
        choices=["backprop", "parameter-shift", "adjoint"],
    )
    p.add_argument("--data_path", type=str, default=config["data_path"])
    p.add_argument("--raw_data_path", type=str, default=config["raw_data_path"])
    p.add_argument(
        "--filter_elements",
        type=str,
        default=None,
        help="Comma-separated elements to filter by (e.g. Mo,S)",
    )
    p.add_argument("--schnet_hidden", type=int, default=config["schnet_hidden"])
    p.add_argument("--schnet_filters", type=int, default=config["schnet_filters"])
    p.add_argument(
        "--schnet_interactions", type=int, default=config["schnet_interactions"]
    )
    p.add_argument(
        "--n_features_compressed", type=int, default=config["n_features_compressed"]
    )
    p.add_argument("--weight_decay", type=float, default=config["weight_decay"])
    p.add_argument("--patience", type=int, default=config["patience"])

    args = p.parse_args()
    config.update(vars(args))

    # Parse comma-separated filter_elements into a list
    if config.get("filter_elements") and isinstance(config["filter_elements"], str):
        config["filter_elements"] = [
            e.strip() for e in config["filter_elements"].split(",")
        ]
    return config

