import os
import json
import torch
import numpy as np
import random
import warnings
from typing import List, Dict, Tuple
from torch_geometric.data import Data

try:
    from pymatgen.core import Element, Structure, Lattice
    _HAS_PYMATGEN = True
except ImportError:
    _HAS_PYMATGEN = False
    warnings.warn('pymatgen not installed. Using fallback atomic numbers and no PBC.')

from .features import PhysicsFeatureExtractor


# Fallback atomic number map (used only if pymatgen unavailable)
_FALLBACK_Z = {
    'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33,
    'Se': 34, 'Mo': 42, 'Nb': 41, 'Ta': 73, 'W': 74, 'Te': 52, 'Sn': 50,
    'In': 49, 'Sb': 51, 'Bi': 83,
}


class MaterialDataset:
    """Loads and manages the 2Dmatpedia dataset.

    Supports NDJSON and standard JSON formats with auto-detection.
    """

    # Columns irrelevant to simulations
    STRIP_KEYS = {
        '_id', '_tasksbuilder', 'calc_settings', 'created_at',
        'creation_task_label', 'bandstructure', 'discovery_process',
        'formula_anonymous', 'formula_reduced_abc', 'relative_id',
        'source_id',
    }

    def __init__(self, raw_path: str = None, cache_path: str = None):
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            self.materials = data['materials'].tolist()
            print(f'[DATA] Loaded {len(self.materials)} materials from cache.')
        elif raw_path:
            self.materials = self._load_raw(raw_path)
            print(f'[DATA] Loaded {len(self.materials)} materials from raw JSON.')
        else:
            raise FileNotFoundError('Provide either raw_path or cache_path.')

    def _load_raw(self, path: str) -> List[Dict]:
        """Load JSON with auto-detection of NDJSON vs standard JSON array."""
        with open(path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                raw = json.load(f)
            else:
                raw = []
                for line in f:
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))

        for entry in raw:
            for k in self.STRIP_KEYS:
                entry.pop(k, None)

        return raw

    def print_stats(self, verbose: bool = True):
        """Print dataset statistics. Set verbose=False to suppress."""
        if not verbose:
            return
        n = len(self.materials)
        if n == 0:
            print('[DATA] Empty dataset.')
            return

        # Column count
        all_keys = set()
        for m in self.materials:
            all_keys.update(m.keys())
        print(f'\n{"="*60}')
        print(f'  DATASET SUMMARY')
        print(f'{"="*60}')
        print(f'  Records:  {n}')
        print(f'  Columns:  {len(all_keys)}')

        # Per-field coverage for numeric targets
        target_fields = ['bandgap', 'energy_per_atom', 'energy_vdw_per_atom',
                         'exfoliation_energy_per_atom', 'decomposition_energy',
                         'total_magnetization']
        print(f'\n  {"Field":<40s} {"Valid":>6s} {"Coverage":>9s}')
        print(f'  {"-"*55}')
        for field in target_fields:
            count = 0
            vals = []
            for m in self.materials:
                v = m.get(field)
                if v is not None:
                    try:
                        fv = float(v)
                        if not np.isnan(fv):
                            count += 1
                            vals.append(fv)
                    except (TypeError, ValueError):
                        pass
            pct = count / n * 100
            if vals:
                mn, mx = min(vals), max(vals)
                print(f'  {field:<40s} {count:>6d} {pct:>7.1f}%  [{mn:.3f}, {mx:.3f}]')
            else:
                print(f'  {field:<40s} {count:>6d} {pct:>7.1f}%')

        # Unique elements
        elements = set()
        for m in self.materials:
            sites = m.get('structure', {}).get('sites', [])
            for s in sites:
                sp = s.get('species', [{}])
                elements.add(sp[0].get('element', '?'))
        print(f'\n  Unique elements: {len(elements)}')
        print(f'  Elements: {", ".join(sorted(elements))}')
        print(f'{"="*60}\n')

    def filter_valid(self, target_name: str) -> 'MaterialDataset':
        """Keep only materials with a valid (non-null, non-NaN) target."""
        valid = []
        for m in self.materials:
            val = m.get(target_name)
            if val is None:
                continue
            try:
                if not np.isnan(float(val)):
                    valid.append(m)
            except (TypeError, ValueError):
                continue
        print(f'[DATA] Filtered: {len(self.materials)} -> {len(valid)} with valid "{target_name}".')
        self.materials = valid
        return self

    def filter_by_elements(self, required: List[str]) -> 'MaterialDataset':
        """Keep only materials containing ALL required elements."""
        filtered = []
        for m in self.materials:
            sites = m.get('structure', {}).get('sites', [])
            mat_elements = set()
            for s in sites:
                sp = s.get('species', [{}])
                mat_elements.add(sp[0].get('element', ''))
            if all(el in mat_elements for el in required):
                filtered.append(m)
        print(f'[DATA] Element filter ({required}): {len(self.materials)} -> {len(filtered)}.')
        self.materials = filtered
        return self

    def get_split(self, test_ratio: float = 0.2, seed: int = 42) -> Tuple[List, List]:
        """Deterministic train/test split."""
        shuffled = self.materials.copy()
        random.Random(seed).shuffle(shuffled)
        split = int(len(shuffled) * (1 - test_ratio))
        return shuffled[:split], shuffled[split:]

    def get_kfold_splits(self, k: int = 5, seed: int = 42) -> List[Tuple[List, List]]:
        """Generate k-fold cross-validation splits."""
        shuffled = self.materials.copy()
        random.Random(seed).shuffle(shuffled)
        fold_size = len(shuffled) // k
        folds = []
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(shuffled)
            val_set = shuffled[start:end]
            train_set = shuffled[:start] + shuffled[end:]
            folds.append((train_set, val_set))
        return folds

    def save_cache(self, path: str):
        """Save processed dataset as NPZ."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, materials=np.array(self.materials, dtype=object))
        print(f'[DATA] Saved cache: {path} ({len(self.materials)} materials)')


def compute_normalization_stats(materials: List[Dict], target_name: str,
                                extractor: PhysicsFeatureExtractor = None
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std from training data."""
    if extractor is None:
        extractor = PhysicsFeatureExtractor()
    excludes = list(extractor.get_auto_excludes(target_name))
    features = [extractor.extract(m, exclude_keys=excludes) for m in materials]
    features = np.array(features)
    mean = torch.tensor(np.mean(features, axis=0), dtype=torch.float32)
    std = torch.tensor(np.std(features, axis=0), dtype=torch.float32)
    std[std == 0] = 1.0
    return mean, std


def _get_atomic_number(symbol: str) -> int:
    """Get atomic number with pymatgen fallback."""
    if _HAS_PYMATGEN:
        try:
            return Element(symbol).Z
        except Exception:
            pass
    z = _FALLBACK_Z.get(symbol)
    if z is None:
        warnings.warn(f'Unknown element "{symbol}", atomic number set to 0')
        return 0
    return z


def create_graph_data(material: Dict, cutoff: float = 5.0) -> Data:
    """Build PyG Data object for SchNet from material dict.

    Uses pymatgen to compute periodic boundary conditions (PBC) for the
    neighbor graph. Falls back to Cartesian distances if pymatgen is
    unavailable or lattice data is missing.
    """
    structure_dict = material.get('structure', {})
    sites = structure_dict.get('sites', [])

    z_list, pos_list = [], []
    for s in sites:
        el = s['species'][0]['element']
        z_list.append(_get_atomic_number(el))
        pos_list.append(s['xyz'])

    z = torch.tensor(z_list, dtype=torch.long)
    pos = torch.tensor(pos_list, dtype=torch.float)
    batch = torch.zeros(len(z), dtype=torch.long)

    edge_index, edge_weight = _build_pbc_edges(structure_dict, cutoff)
    return Data(z=z, pos=pos, batch=batch, edge_index=edge_index, edge_attr=edge_weight)


def _build_pbc_edges(structure_dict: Dict, cutoff: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PBC-aware neighbor list using pymatgen."""
    lattice_data = structure_dict.get('lattice', {})
    sites = structure_dict.get('sites', [])

    if not _HAS_PYMATGEN or 'matrix' not in lattice_data:
        return _build_distance_edges(structure_dict, cutoff)

    lattice = Lattice(lattice_data['matrix'])
    species = [s['species'][0]['element'] for s in sites]
    frac_coords = [s.get('abc', [0, 0, 0]) for s in sites]

    struct = Structure(lattice, species, frac_coords)

    src_indices, dst_indices, distances = [], [], []
    all_neighbors = struct.get_all_neighbors(r=cutoff)
    for i, neighbors in enumerate(all_neighbors):
        for nn in neighbors:
            src_indices.append(i)
            dst_indices.append(nn.index)
            distances.append(nn.nn_distance)

    if not src_indices:
        n = len(sites)
        src_indices = list(range(n))
        dst_indices = list(range(n))
        distances = [0.0] * n

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    edge_weight = torch.tensor(distances, dtype=torch.float)
    return edge_index, edge_weight


def _build_distance_edges(structure_dict: Dict, cutoff: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback: build edges from Cartesian distances (no PBC)."""
    sites = structure_dict.get('sites', [])
    coords = np.array([s.get('xyz', [0, 0, 0]) for s in sites])

    if len(coords) < 2:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float)

    from scipy.spatial.distance import cdist as _cdist
    dists = _cdist(coords, coords)
    np.fill_diagonal(dists, np.inf)
    src, dst = np.where(dists < cutoff)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(dists[src, dst], dtype=torch.float)
    return edge_index, edge_weight


def prepare_pyg_dataset(
    materials: list,
    target_name: str,
    extractor: PhysicsFeatureExtractor,
    cutoff: float = 5.0,
) -> list:
    """Pre-compute PyG Data objects with physics features + target attached.

    Returns a list of PyG Data objects, each containing:
      - z, pos, edge_index, edge_attr (graph structure)
      - physics: float tensor of physics features [n_features]
      - y: float tensor of target value [1]

    Skips materials with invalid target or graph construction errors.
    """
    excludes = list(extractor.get_auto_excludes(target_name))
    dataset = []
    skipped = 0
    for mat in materials:
        # Validate target
        tval = mat.get(target_name)
        if tval is None:
            skipped += 1
            continue
        try:
            tv = float(tval)
            if np.isnan(tv):
                skipped += 1
                continue
        except (TypeError, ValueError):
            skipped += 1
            continue

        # Build graph
        try:
            graph = create_graph_data(mat, cutoff=cutoff)
        except Exception as e:
            warnings.warn(f'[DATA] Skipping material: {e}')
            skipped += 1
            continue

        # Physics features — store as [1, n_features] so DataLoader stacks to [B, n_features]
        phys = extractor.extract(mat, exclude_keys=excludes)
        graph.physics = torch.tensor(phys, dtype=torch.float32).unsqueeze(0)
        graph.y = torch.tensor([[tv]], dtype=torch.float32)  # [1, 1]

        # Remove the per-sample batch vector — DataLoader will create batched ones
        del graph.batch

        dataset.append(graph)

    print(f'[DATA] Pre-computed {len(dataset)} PyG graphs ({skipped} skipped).')
    return dataset
