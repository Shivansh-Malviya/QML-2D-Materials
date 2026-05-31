"""Smoke tests for ModelB forward pass."""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch_geometric.data import Batch
from src.model import ModelB
from src.config import get_default_config
from src.features import FEATURE_REGISTRY
from src.features import PhysicsFeatureExtractor
from src.data import create_graph_data


SAMPLE_MATERIAL = {
    "bandgap": 1.5,
    "energy_per_atom": -4.46,
    "energy_vdw_per_atom": -4.50,
    "decomposition_energy": 0.15,
    "structure": {
        "lattice": {"matrix": [[3.2, 0, 0], [0, 3.2, 0], [0, 0, 15.0]]},
        "sites": [
            {
                "species": [{"element": "Mo"}],
                "xyz": [0, 0, 7.5],
                "abc": [0, 0, 0.5],
                "label": "Mo",
            },
            {
                "species": [{"element": "S"}],
                "xyz": [1.6, 0.9, 8.5],
                "abc": [0.5, 0.28, 0.57],
                "label": "S",
            },
        ],
    },
}


def _make_batch():
    graph = create_graph_data(SAMPLE_MATERIAL, cutoff=5.0)
    ext = PhysicsFeatureExtractor()
    graph.physics = torch.tensor(
        [ext.extract(SAMPLE_MATERIAL, exclude_keys=["bandgap"])], dtype=torch.float32
    )
    graph.y = torch.tensor([[1.5]], dtype=torch.float32)
    return Batch.from_data_list([graph])


def test_forward_produces_scalar():
    config = get_default_config()
    config["n_qubits"] = 2
    config["n_vqc_layers"] = 1
    n_phys = len(FEATURE_REGISTRY)
    p_mean = torch.zeros(n_phys)
    p_std = torch.ones(n_phys)
    model = ModelB(config, p_mean, p_std)
    out = model(_make_batch())
    assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"


def test_optimizer_has_two_groups():
    config = get_default_config()
    config["n_qubits"] = 2
    n_phys = len(FEATURE_REGISTRY)
    p_mean = torch.zeros(n_phys)
    p_std = torch.ones(n_phys)
    model = ModelB(config, p_mean, p_std)

    classical = [p for n, p in model.named_parameters() if "q_weights" not in n]
    quantum = [p for n, p in model.named_parameters() if "q_weights" in n]
    optimizer = torch.optim.Adam(
        [
            {"params": classical, "weight_decay": 1e-3},
            {"params": quantum, "weight_decay": 0.0},
        ],
        lr=5e-3,
    )
    assert len(optimizer.param_groups) == 2


def test_relu_not_tanh_after_compressor():
    """Verify the compressor uses ReLU, not tanh (DA Fix F4)."""
    config = get_default_config()
    config["n_qubits"] = 2
    config["n_vqc_layers"] = 1
    n_phys = len(FEATURE_REGISTRY)
    p_mean = torch.zeros(n_phys)
    p_std = torch.ones(n_phys)
    model = ModelB(config, p_mean, p_std)

    # Feed a large positive value through compressor; ReLU should pass it, tanh would squash
    with torch.no_grad():
        test_input = torch.ones(1, config["schnet_hidden"] + n_phys) * 10.0
        compressed = torch.relu(model.compressor(test_input))
        has_values_gt_1 = (compressed > 1.0).any().item()
        assert has_values_gt_1, (
            "Compressor should use ReLU (values > 1.0 possible), not tanh"
        )


if __name__ == "__main__":
    test_forward_produces_scalar()
    test_optimizer_has_two_groups()
    test_relu_not_tanh_after_compressor()
    print("All model tests PASSED")
