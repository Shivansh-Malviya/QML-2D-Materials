"""
Model A: Graph + 28-Feature Hybrid + Angle Encoding
====================================================

Complete implementation with:
- Graph construction (PBC-aware)
- GNN for geometric features
- Physics feature extraction
- 26-feature hybrid (12 geom + 14 physics)
- Angle encoding to 26 qubits
- Variational quantum circuit
- Physics-informed loss
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

try:
    from .graph_constructor import create_graph_with_pbc
    from .physics_features import PhysicsFeatureExtractor
    from .config_a import CONFIG
except ImportError:  # pragma: no cover - script-style fallback
    from graph_constructor import create_graph_with_pbc
    from physics_features import PhysicsFeatureExtractor
    from config_a import CONFIG


class SimpleGNN(MessagePassing):
    """
    Simple Graph Neural Network for extracting geometric features

    Outputs 12-dimensional global geometric representation
    """

    def __init__(self, hidden_dim=32, num_layers=3, output_dim=12):
        super().__init__(aggr="mean")

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Input: atomic number (1 feature)
        self.node_encoder = nn.Linear(1, hidden_dim)

        # Message passing layers
        self.convs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Global pooling & output
        self.pool = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [n_nodes, 1] (atomic numbers)
            edge_index: Edge connectivity [2, n_edges]
            batch: Batch assignment (for multiple graphs)

        Returns:
            Global graph features [batch_size, 12]
        """
        # Encode nodes
        x = self.node_encoder(x)

        # Message passing
        for conv in self.convs:
            x = self.activation(conv(x))
            x = self.propagate(edge_index, x=x)

        # Global pooling
        if batch is None:
            # Single graph
            x_global = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            # Multiple graphs - pool per batch
            x_global = torch.zeros(batch.max().item() + 1, self.hidden_dim)
            for i in range(batch.max().item() + 1):
                mask = batch == i
                x_global[i] = x[mask].mean(dim=0)

        # Output projection
        out = self.pool(x_global)  # [batch_size, 12]

        return out

    def message(self, x_j):
        """Message from neighbor j to node i"""
        return x_j


class ModelA(nn.Module):
    """
    Complete Model A Architecture

    Flow:
        Material → Graph → GNN → 12 geom features
                         ↓
                  Physics extraction → 14 physics features
                         ↓
                  Concatenate → 26 total features
                         ↓
                  Angle Encoding → 26 qubits
                         ↓
                  VQC (3 layers) → Measurement
                         ↓
                  Linear readout → E_exfoliation
    """

    def __init__(self, config=CONFIG):
        super().__init__()

        self.config = config

        # GNN for geometric features
        self.gnn = SimpleGNN(
            hidden_dim=config["gnn_hidden_dim"],
            num_layers=config["gnn_layers"],
            output_dim=config["n_geometric_features"],
        )

        # Physics feature extractor
        self.physics_extractor = PhysicsFeatureExtractor(
            bond_cutoff=config["bond_cutoff"]
        )

        # Quantum circuit
        self.n_qubits = config["n_qubits"]
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Initialize quantum weights
        self.q_weights = nn.Parameter(
            torch.randn(config["n_vqc_layers"], self.n_qubits, 3) * 0.1
        )

        # Feature compressor (if features > qubits)
        self.compress_features = config["n_total_features"] > self.n_qubits
        if self.compress_features:
            self.compressor = nn.Linear(config["n_total_features"], self.n_qubits)
            # Initialize close to identity/PCA-like to start well
            nn.init.orthogonal_(self.compressor.weight)

        # Linear readout
        self.readout = nn.Linear(self.n_qubits, 1)

    def quantum_circuit(self, features, weights):
        """
        Variational Quantum Circuit

        - Angle Encoding (26 features → 26 qubits)
        - 3-layer ansatz with ring topology
        """
        # Angle Encoding: Each feature → RY rotation on corresponding qubit
        for i in range(self.n_qubits):
            qml.RY(features[i], wires=i)

        # Variational layers (ring topology for symmetry)
        for layer_weights in weights:
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(layer_weights[i, 0], wires=i)
                qml.RZ(layer_weights[i, 1], wires=i)
                qml.RY(layer_weights[i, 2], wires=i)

            # Entangling layer (ring)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, material):
        """
        Forward pass

        Args:
            material: Material dict from MoDataset

        Returns:
            prediction: Exfoliation energy prediction
            features: 26-dim feature vector (for analysis)
        """
        # 1. Build graph
        graph = create_graph_with_pbc(
            material,
            cutoff=self.config["bond_cutoff"],
            max_neighbors=self.config["max_pbc_neighbors"],
        )

        # 2. GNN forward (geometric features)
        geom_features = self.gnn(graph.x, graph.edge_index)  # [1, 12]

        # 3. Extract physics features
        physics_dict = self.physics_extractor.extract_all_features(material)

        # Build physics feature vector (14 features)
        physics_features = torch.tensor(
            [
                # DFT (3 available)
                material.get("vdw_energy", 0.0),  # CRITICAL!
                material.get("energy_per_atom", 0.0),
                material.get("decomposition_energy", 0.0)
                if material.get("decomposition_energy") is not None
                else 0.0,
                # Electronic (2 available)
                material.get("bandgap", 0.0)
                if material.get("bandgap") is not None
                else 0.0,
                material.get("magnetization", 0.0),
                # Structural (8)
                physics_dict["mo_coordination"],
                physics_dict["packing_efficiency"],
                physics_dict["bond_density"],
                physics_dict["layer_thickness"],
                physics_dict["lattice_strain"],
                physics_dict["mean_electronegativity"],
                physics_dict["d_band_center"],
                physics_dict["interlayer_distance"],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 14]

        # 4. Concatenate features
        # 4. Concatenate features
        features = torch.cat(
            [geom_features, physics_features], dim=1
        )  # [1, 26] -> [1, 25]

        # Compress if needed (e.g. 25 -> 15 qubits)
        if self.compress_features:
            features_for_q = (
                torch.tanh(self.compressor(features)) * np.pi
            )  # Scale to [-pi, pi] for angle encoding
        else:
            features_for_q = features

        # 5. Quantum circuit
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(feat, weights):
            return self.quantum_circuit(feat, weights)

        # Run quantum circuit
        # Use compressed features for quantum part
        q_out = circuit(features_for_q.squeeze(), self.q_weights)
        q_out = torch.stack(q_out).unsqueeze(0)  # [1, 26]

        # 6. Linear readout
        # Cast to float32 (PennyLane returns double)
        q_out = q_out.to(torch.float32)
        prediction = self.readout(q_out)  # [1, 1]

        return prediction.squeeze(), features.squeeze()

    def predict_batch(self, materials):
        """Predict for batch of materials"""
        predictions = []
        features_list = []

        for material in materials:
            pred, feat = self.forward(material)
            predictions.append(pred.item())
            features_list.append(feat.detach().numpy())

        return np.array(predictions), np.array(features_list)


def physics_informed_loss(pred, target, vdw_energy, bandgap=None, config=CONFIG):
    """
    Physics-Informed Loss Function

    Components:
    1. MSE (primary)
    2. Positivity penalty (E_exf > 0)
    3. vdW upper bound (E_exf < |E_vdW|)
    4. Bandgap auxiliary task (optional multi-task)
    """
    weights = config["loss_weights"]

    # 1. Primary MSE
    mse = torch.nn.functional.mse_loss(pred, target)

    # 2. Positivity constraint
    positivity_violation = torch.relu(-pred)  # Penalize negative

    # 3. vdW bound constraint
    vdw_abs = torch.abs(vdw_energy)
    vdw_violation = torch.relu(pred - vdw_abs)  # Penalize exceeding vdW

    # Total loss
    loss = (
        weights["mse"] * mse
        + weights["positivity"] * positivity_violation.mean()
        + weights["vdw_bound"] * vdw_violation.mean()
    )

    # 4. Bandgap auxiliary (if provided)
    if bandgap is not None and "bandgap_auxiliary" in weights:
        # Multi-task: predict both exfoliation AND bandgap
        # (Not implementing full multi-task head for now - placeholder)
        pass

    return loss, {
        "mse": mse.item(),
        "positivity_violation": positivity_violation.mean().item(),
        "vdw_violation": vdw_violation.mean().item(),
    }


if __name__ == "__main__":
    print("Testing Model A on single material...")

    from mo_dataset import MoDataset

    dataset = MoDataset(cache=True)

    model = ModelA()
    material = dataset[0]

    print(f"Material: {material['formula']}")
    print("\nForward pass...")

    pred, features = model.forward(material)

    print(f"Features extracted: {features.shape}")
    print(f"Prediction: {pred.item():.4f} eV/atom")
    print(f"Actual: {material['exfoliation_energy']:.4f} eV/atom")

    print("\n✅ Model A architecture working!")
