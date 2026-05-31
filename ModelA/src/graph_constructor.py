"""
PBC-Aware Graph Construction for 2D Materials
==============================================

Creates molecular graphs with Periodic Boundary Conditions
Critical for crystal structures (previous models missed this!)
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist

# Atomic number mapping
ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "W": 74,
    "Pb": 82,
    "Bi": 83,
}


def create_graph_with_pbc(
    material: Dict, cutoff: float = 5.0, max_neighbors: int = 12
) -> Data:
    """
    Create PyTorch Geometric graph with Periodic Boundary Conditions

    Args:
        material: Material dict from MoDataset
        cutoff: Bond distance cutoff (Ångströms)
        max_neighbors: Maximum number of periodic images to check

    Returns:
        PyG Data object with:
            - x: Node features [n_atoms, feat_dim]
            - edge_index: Edge connectivity [2, n_edges]
            - pos: 3D coordinates [n_atoms, 3]
            - batch: Batch index (for batching multiple graphs)
    """
    structure = material["structure"]
    sites = structure["sites"]
    lattice_dict = structure["lattice"]

    # Extract atomic info
    coords = np.array([site["xyz"] for site in sites])
    elements = [site["species"][0]["element"] for site in sites]

    # Lattice vectors for PBC
    lattice_matrix = np.array(lattice_dict["matrix"])

    n_atoms = len(coords)

    # Node features: [atomic_number, x, y, z]
    node_features = []
    for i, el in enumerate(elements):
        atomic_num = ATOMIC_NUMBERS.get(el, 0)
        feat = [atomic_num]  # Can add more features later
        node_features.append(feat)

    node_features = torch.tensor(node_features, dtype=torch.float)
    positions = torch.tensor(coords, dtype=torch.float)

    # Build edges with PBC
    edge_index = build_edges_pbc(coords, lattice_matrix, cutoff, max_neighbors)

    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        pos=positions,
        num_nodes=n_atoms,
    )

    return data


def build_edges_pbc(
    coords: np.ndarray, lattice: np.ndarray, cutoff: float, max_neighbors: int = 12
) -> np.ndarray:
    """
    Build edge list considering periodic boundary conditions

    Includes edges to atoms in neighboring unit cells (critical for crystals!)
    """
    n_atoms = len(coords)
    edges = []

    # Generate periodic image offsets
    # For 2D materials, typically need [-1, 0, 1] in a,b directions
    # c-direction (out-of-plane) usually doesn't need PBC for monolayers
    offsets = []
    for i in range(-max_neighbors, max_neighbors + 1):
        for j in range(-max_neighbors, max_neighbors + 1):
            for k in range(-1, 2):  # Limited c-direction for 2D
                if i == 0 and j == 0 and k == 0:
                    continue  # Skip self-cell
                offsets.append([i, j, k])

    offsets = np.array(offsets)

    # For each atom pair
    for i in range(n_atoms):
        pos_i = coords[i]

        for j in range(n_atoms):
            pos_j_base = coords[j]

            # Check original cell (i != j to avoid self-loops)
            if i != j:
                dist = np.linalg.norm(pos_i - pos_j_base)
                if dist < cutoff:
                    edges.append([i, j])

            # Check periodic images
            for offset in offsets:
                # Apply lattice translation
                translation = np.dot(offset, lattice)
                pos_j_periodic = pos_j_base + translation

                dist = np.linalg.norm(pos_i - pos_j_periodic)
                if dist < cutoff:
                    edges.append([i, j])

    # Remove duplicates and convert to array
    if len(edges) > 0:
        edges = np.array(edges).T  # Shape: [2, n_edges]

        # Remove duplicate edges
        edges_set = set(map(tuple, edges.T))
        edges = np.array(list(edges_set)).T
    else:
        edges = np.array([[], []], dtype=np.int64)

    return edges


def test_graph_construction():
    """Test graph construction on sample materials"""
    try:
        from .mo_dataset import MoDataset
    except ImportError:  # pragma: no cover - script-style fallback
        from mo_dataset import MoDataset

    dataset = MoDataset(cache=True)

    print("Testing graph construction...\n")

    # Test on first 3 materials
    for idx in range(min(3, len(dataset))):
        material = dataset[idx]

        print(f"Material {idx + 1}: {material['formula']}")
        print(f"  Atoms: {len(material['sites'])}")

        # Build graph
        graph = create_graph_with_pbc(material, cutoff=5.0, max_neighbors=1)

        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.edge_index.shape[1]}")
        print(f"  Node features shape: {graph.x.shape}")
        print(f"  Positions shape: {graph.pos.shape}")

        # Check connectivity
        avg_degree = graph.edge_index.shape[1] / graph.num_nodes
        print(f"  Average degree: {avg_degree:.2f}")
        print()

    print("✅ Graph construction successful!")

    return graph


if __name__ == "__main__":
    test_graph_construction()
