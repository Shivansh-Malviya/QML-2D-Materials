import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import Dict
from torch_geometric.nn import SchNet, global_mean_pool
from .features import PhysicsFeatureExtractor
from .data import create_graph_data


# Supported quantum backends
QUANTUM_BACKENDS = {
    'simulator': 'default.qubit',
    'lightning_cpu': 'lightning.qubit',
    'lightning_gpu': 'lightning.gpu',
    'ibm': 'qiskit.ibmq',
    'ionq': 'ionq.simulator',
}


class _PBCInteractionGraph(nn.Module):
    """Returns pre-computed PBC edge_index and edge_weight.

    SchNet calls ``self.interaction_graph(pos, batch)`` during forward.
    This module stores the PBC-computed edges and returns them, bypassing
    SchNet's built-in RadiusInteractionGraph (which ignores PBC).
    """

    def __init__(self):
        super().__init__()
        self._edge_index = None
        self._edge_weight = None

    def set_edges(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        self._edge_index = edge_index
        self._edge_weight = edge_weight

    def forward(self, pos, batch):
        return self._edge_index, self._edge_weight


class ModelB(nn.Module):
    """Hybrid Graph-QNN: SchNet encoder + physics features + VQC.

    Architecture described in docs/architecture.md.
    Accepts batched PyG Data with pre-computed physics features attached.
    """

    def __init__(self, config: Dict, phys_mean: torch.Tensor, phys_std: torch.Tensor):
        super().__init__()
        self.config = config
        self.register_buffer('phys_mean', phys_mean)
        self.register_buffer('phys_std', phys_std)

        # PBC-aware interaction graph for SchNet
        self.pbc_graph = _PBCInteractionGraph()

        # SchNet encoder (uses our PBC graph instead of RadiusInteractionGraph)
        self.gnn = SchNet(
            hidden_channels=config['schnet_hidden'],
            num_filters=config['schnet_filters'],
            num_interactions=config['schnet_interactions'],
            cutoff=config['cutoff'],
            interaction_graph=self.pbc_graph,
            readout='mean',
        )

        # Forward hook to capture per-atom hidden states before readout MLP
        self._gnn_hidden = None
        self.gnn.lin1.register_forward_pre_hook(self._capture_hidden)

        n_physics = phys_mean.shape[0]
        self.n_qubits = config['n_qubits']

        # Feature dimensions: SchNet(hidden) + Physics(active features)
        n_combined = config['schnet_hidden'] + n_physics

        # Classical compressor (information bottleneck)
        self.compressor = nn.Linear(n_combined, config['n_features_compressed'])
        nn.init.orthogonal_(self.compressor.weight)
        nn.init.zeros_(self.compressor.bias)

        # FIX-1: LayerNorm before angle projection prevents tanh saturation
        self.pre_q_norm = nn.LayerNorm(config['n_features_compressed'])

        # Quantum projection: compressed -> n_qubits angles
        self.q_projection = nn.Linear(config['n_features_compressed'], self.n_qubits)
        nn.init.xavier_uniform_(self.q_projection.weight)

        # Quantum device — fallback chain: requested → lightning.qubit → default.qubit
        import warnings
        backend_key = config.get('quantum_backend', 'simulator')
        backend_name = QUANTUM_BACKENDS.get(backend_key, backend_key)
        user_dm = config.get('diff_method', 'backprop')

        if backend_key in ('ibm', 'ionq'):
            diff_method = 'parameter-shift'
        elif backend_key in ('lightning_cpu', 'lightning_gpu'):
            diff_method = 'adjoint'
        else:
            diff_method = user_dm

        _FALLBACK_CHAIN = [
            (backend_name, diff_method),
            ('lightning.qubit', 'adjoint'),
            ('default.qubit', 'backprop'),
        ]
        for _bn, _dm in _FALLBACK_CHAIN:
            try:
                self.dev = qml.device(_bn, wires=self.n_qubits)
                diff_method = _dm
                if _bn != backend_name:
                    warnings.warn(f'[MODEL] Backend fallback: {backend_name!r} → {_bn!r}')
                print(f'[MODEL] Quantum: {_bn} | diff={_dm}')
                break
            except Exception as e:
                warnings.warn(f'[MODEL] Cannot create {_bn}: {e}')
                continue

        # VQC weights
        self.q_weights = nn.Parameter(
            torch.randn(config['n_vqc_layers'], self.n_qubits, 3) * 0.1
        )

        # Define QNode once
        @qml.qnode(self.dev, interface='torch', diff_method=diff_method)
        def _qnode(features, weights):
            for i in range(self.n_qubits):
                qml.RX(features[i], wires=i)
            for layer in range(weights.shape[0]):
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                for i in range(self.n_qubits):
                    qml.Rot(*weights[layer, i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._qnode = _qnode

        # FIX-2: LayerNorm after VQC re-centers bounded [-1,1] PauliZ expectations
        self.post_q_norm = nn.LayerNorm(self.n_qubits)

        # Readout with learnable scale+shift for arbitrary target ranges
        self.readout = nn.Linear(self.n_qubits, 1)
        nn.init.uniform_(self.readout.weight, -0.5, 0.5)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_shift = nn.Parameter(torch.tensor(0.0))

    def _capture_hidden(self, module, args):
        """Forward pre-hook: capture per-atom embeddings before readout MLP."""
        self._gnn_hidden = args[0]

    def forward(self, batch) -> torch.Tensor:
        """Batched forward pass.

        Args:
            batch: PyG Batch object with .z, .pos, .batch, .edge_index,
                   .edge_attr, .physics (pre-computed features).

        Returns:
            predictions: [batch_size, 1] tensor.
        """
        device = next(self.parameters()).device

        # ── Batched SchNet ───────────────────────────────────────────────
        self.pbc_graph.set_edges(batch.edge_index, batch.edge_attr)
        _ = self.gnn(batch.z, batch.pos, batch.batch)  # triggers hook
        g_out = global_mean_pool(self._gnn_hidden, batch.batch)  # [B, hidden]

        # ── Batched physics features ─────────────────────────────────────
        p_norm = (batch.physics - self.phys_mean) / self.phys_std  # [B, n_phys]

        # ── Batched compress + project ───────────────────────────────────
        combined = torch.cat([g_out, p_norm], dim=1)       # [B, hidden + n_phys]
        compressed = torch.relu(self.compressor(combined))  # [B, n_compressed]
        normed = self.pre_q_norm(compressed)                # [B, n_compressed]  FIX-1
        q_inputs = self.q_projection(normed).clamp(-np.pi, np.pi)  # [B, n_qubits]

        # ── Per-sample VQC (PennyLane cannot batch circuits) ─────────────
        B = q_inputs.shape[0]
        q_outputs = []
        for i in range(B):
            q_out_i = torch.stack(self._qnode(q_inputs[i], self.q_weights))
            q_outputs.append(q_out_i)
        q_out = torch.stack(q_outputs)  # [B, n_qubits]
        q_out = q_out.float()

        # ── Batched readout ──────────────────────────────────────────────
        q_out = self.post_q_norm(q_out)                    # FIX-2
        return self.readout(q_out) * self.output_scale + self.output_shift  # [B, 1]
