"""
Model A Configuration
====================

Retained snapshot configuration for the intermediate Model A line.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_A1 = {
    # Model name
    "name": "ModelA_Config1_NoPCA_28qubits",
    # Dataset
    "n_materials": 86,  # Mo family
    "train_ratio": 0.8,  # 68 train, 18 test
    "seed": 42,
    # Graph construction
    "bond_cutoff": 5.0,  # Ångströms
    "max_pbc_neighbors": 1,  # Periodic images to check
    # GNN architecture (for geometric features)
    "gnn_type": "simple",  # Simple message passing GNN
    "gnn_hidden_dim": 32,
    "gnn_layers": 3,
    "gnn_output_dim": 12,  # 12 geometric features
    # Physics features (16 total)
    "physics_features": [
        # DFT (4)
        "vdw_energy",  # CRITICAL!
        "energy_per_atom",
        "decomposition_energy",
        # 'formation_energy',  # 0% coverage - skip
        # Electronic (4)
        "bandgap",
        "magnetization",
        # Note: cbm_vbm, work_func not available in current dataset
        # Structural (8)
        "mo_coordination",
        "packing_efficiency",  # #1 predictor
        "bond_density",
        "layer_thickness",
        "lattice_strain",
        "mean_electronegativity",
        "d_band_center",  # Placeholder
        "interlayer_distance",
    ],
    # Total features
    "n_geometric_features": 12,
    "n_physics_features": 13,  # Actually available (14 - 1 missing)
    "n_total_features": 25,  # 12 + 13 = 25
    # PCA (Config A1 - NO PCA)
    "use_pca": False,
    "pca_components": None,
    # Quantum circuit
    "n_qubits": 25,  # Match total features (updated)
    "encoding": "angle",  # AngleEncoding
    "n_vqc_layers": 3,
    "ansatz": "ring",  # Ring topology for symmetry
    # Training
    "epochs": 50,
    "batch_size": 8,  # Small dataset
    "learning_rate": 0.01,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    # Physics-informed loss
    "loss_weights": {
        "mse": 1.0,
        "positivity": 0.1,
        "vdw_bound": 0.1,
        "bandgap_auxiliary": 0.3,  # Multi-task
    },
    # Checkpointing
    "checkpoint_every": 10,  # epochs
    "save_best_only": True,
    # Device
    "device": "cpu",  # Use CPU for quantum simulation (GPU doesn't help)
    # Compact-repo local paths
    "output_dir": str(BASE_DIR / "artifacts"),
    "checkpoint_dir": str(BASE_DIR / "artifacts" / "checkpoints"),
    "plots_dir": str(BASE_DIR / "artifacts"),
}

# Config A2: With PCA (if A1 too slow)
CONFIG_A2 = CONFIG_A1.copy()
CONFIG_A2.update(
    {
        "name": "ModelA_Config2_PCA20_20qubits",
        "use_pca": True,
        "pca_components": 20,
        "n_total_features": 20,
        "n_qubits": 20,
    }
)

# Config A3: Aggressive PCA (if A2 still slow)
CONFIG_A3 = CONFIG_A1.copy()
CONFIG_A3.update(
    {
        "name": "ModelA_Config3_PCA15_15qubits",
        "use_pca": True,
        "pca_components": 15,
        "n_total_features": 15,
        "n_qubits": 15,
    }
)

# Default config
# Default config
CONFIG = CONFIG_A3
