import numpy as np
import warnings
from typing import Dict, List, Set

# Priority chain for electronegativity lookups: pymatgen → mendeleev → fallback
try:
    from pymatgen.core import Element as PmgElement

    _HAS_PYMATGEN = True
except ImportError:
    _HAS_PYMATGEN = False

try:
    from mendeleev import element as mendeleev_element

    _HAS_MENDELEEV = True
except ImportError:
    _HAS_MENDELEEV = False

from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# CORRELATED_TARGETS — scientifically validated exclusion map
#
# When predicting target X, also exclude these fields from features:
#
# Rationale (per 2Dmatpedia definitions, ACS Nano 2019):
#   energy_per_atom     : DFT total energy per atom WITHOUT vdW correction
#   energy_vdw_per_atom : DFT total energy per atom WITH vdW correction
#     → These are the SAME quantity computed with/without vdW functional.
#       Pearson r ≈ 0.99+. Must mutually exclude.
#
#   exfoliation_energy_per_atom : E_exf = E_2D - E_bulk (per atom)
#     → Directly computed FROM energy_per_atom.
#       Changes in energy_per_atom linearly shift exfoliation energy.
#       Must exclude energy_per_atom/energy_vdw when predicting exfoliation, and vice versa.
#       Reverse leakage: exfoliation_energy encodes energy_per_atom, so when
#       predicting energy_per_atom or energy_vdw_per_atom, exclude exfoliation too.
#
#   decomposition_energy : Energy to decompose into competing stable phases
#     → Computed from energy_per_atom vs hull of competing phases.
#       Weakly correlated with energy_per_atom (r ≈ 0.3-0.5 across 2Dmatpedia).
#       INDEPENDENT enough to keep as a feature when predicting energy_per_atom.
#       But when predicting decomposition_energy, exclude energy_per_atom (it's an input).
#
#   total_magnetization / magnetism :
#     → magnetism is a categorical label derived from total_magnetization.
#       Must mutually exclude.
#
#   bandgap : No direct mathematical derivation from other fields.
#     → Independent property. No exclusions needed beyond itself.
# ---------------------------------------------------------------------------
CORRELATED_TARGETS = {
    "bandgap": set(),
    "energy_per_atom": {"energy_vdw_per_atom", "exfoliation_energy_per_atom"},
    "energy_vdw_per_atom": {"energy_per_atom", "exfoliation_energy_per_atom"},
    "exfoliation_energy_per_atom": {"energy_per_atom", "energy_vdw_per_atom"},
    "total_magnetization": {"magnetism"},
    "decomposition_energy": {"energy_per_atom"},
}


# ---------------------------------------------------------------------------
# Feature registry: all engineered features with 2Dmatpedia definitions
# ---------------------------------------------------------------------------
FEATURE_REGISTRY = [
    # Energetic (4 features from DFT)
    ("energy_per_atom", "DFT total energy/atom without vdW correction (eV)"),
    ("energy_vdw_per_atom", "DFT total energy/atom with vdW correction (eV)"),
    ("decomposition_energy", "Energy to decompose into competing stable phases (eV)"),
    (
        "exfoliation_energy_per_atom",
        "Exfoliation energy per atom from DFT (eV), 0.0 if missing",
    ),
    # Electronic & magnetic (2 features from DFT)
    ("bandgap", "Electronic bandgap from DFT (eV)"),
    ("total_magnetization", "Total magnetic moment from DFT (μ_B)"),
    # Symmetry & composition (2 features from metadata)
    ("nelements", "Number of unique chemical elements in the material"),
    ("sg_number", "Space group number (1–230), encodes crystal symmetry"),
    # Structural (6 features, computed from structure)
    ("coord_num", "Mean coordination number (neighbors within 3.0 Å)"),
    ("packing_density", "Number of atoms / unit cell volume (atoms/ų)"),
    ("bond_density", "Number of bonds / unit cell volume (r < 2.8 Å)"),
    ("layer_thickness", "z_max - z_min of atomic positions (Å)"),
    ("mean_electronegativity", "Mean Pauling electronegativity of all atoms"),
    ("interlayer_spacing", "Max gap in sorted z-coordinates (Å)"),
    # Statistical (4 features, coordinate moments)
    ("std_x", "Standard deviation of x-coordinates"),
    ("std_y", "Standard deviation of y-coordinates"),
    ("std_z", "Standard deviation of z-coordinates"),
    ("mean_z", "Mean z-coordinate"),
]

DEFAULT_FEATURE_SWITCHES = {name: True for name, _ in FEATURE_REGISTRY}


def _get_electronegativity(element_symbol: str) -> float:
    """Get Pauling electronegativity. Chain: pymatgen → mendeleev → hardcoded fallback."""
    if _HAS_PYMATGEN:
        try:
            en = PmgElement(element_symbol).X
            if en is not None:
                return float(en)
        except Exception:
            pass
    if _HAS_MENDELEEV:
        try:
            el = mendeleev_element(element_symbol)
            if el.en_pauling is not None:
                return float(el.en_pauling)
        except Exception:
            pass
    # Hardcoded fallback (common 2D material elements)
    _FALLBACK = {
        "Mo": 2.16,
        "S": 2.58,
        "Se": 2.55,
        "Te": 2.1,
        "W": 2.36,
        "O": 3.44,
        "B": 2.04,
        "N": 3.04,
        "C": 2.55,
        "Si": 1.90,
        "P": 2.19,
        "Ti": 1.54,
        "Nb": 1.60,
        "Ta": 1.50,
        "Sn": 1.96,
        "Ge": 2.01,
        "Fe": 1.83,
        "Co": 1.88,
        "Ni": 1.91,
        "Cu": 1.90,
        "Zn": 1.65,
        "Ga": 1.81,
        "As": 2.18,
        "In": 1.78,
        "Sb": 2.05,
        "Bi": 2.02,
        "H": 2.20,
        "Cl": 3.16,
        "F": 3.98,
        "Br": 2.96,
    }
    val = _FALLBACK.get(element_symbol)
    if val is None:
        warnings.warn(
            f'[FEATURES] Unknown electronegativity for "{element_symbol}", using 2.0'
        )
        return 2.0
    return val


class PhysicsFeatureExtractor:
    """Extracts physics-informed features from a material dictionary.

    Supports dynamic target exclusion, correlated-field auto-exclusion,
    and feature switching. See docs/data_provenance.md for data context.
    """

    def __init__(self, feature_switches: Dict[str, bool] = None):
        self.switches = feature_switches or DEFAULT_FEATURE_SWITCHES.copy()

    def _get_volume(self, lattice: Dict) -> float:
        a = np.array(lattice["matrix"][0])
        b = np.array(lattice["matrix"][1])
        c = np.array(lattice["matrix"][2])
        return abs(np.dot(a, np.cross(b, c)))

    def get_auto_excludes(self, target_name: str) -> Set[str]:
        """Return the full set of keys to exclude for a given target."""
        excludes = {target_name}
        excludes.update(CORRELATED_TARGETS.get(target_name, set()))
        return excludes

    def extract(self, material: Dict, exclude_keys: List[str] = None) -> List[float]:
        """Extract features, excluding targets and correlated fields."""
        if exclude_keys is None:
            exclude_keys = []
        exclude_set = set(exclude_keys)

        def safe(key, default=0.0):
            if key in exclude_set:
                return default
            val = material.get(key)
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        structure = material.get("structure", {})
        sites = structure.get("sites", [])
        lattice = structure.get("lattice", {})

        elements, coords = [], []
        for s in sites:
            sp = s.get("species", [{}])
            elements.append(sp[0].get("element", "X"))
            coords.append(s.get("xyz", [0, 0, 0]))
        coords = np.array(coords) if coords else np.zeros((0, 3))

        vol = self._get_volume(lattice) if "matrix" in lattice else 1.0

        if len(coords) > 1:
            dists = cdist(coords, coords)
            np.fill_diagonal(dists, np.inf)
            coord_num = np.mean(np.sum(dists < 3.0, axis=1))
            n_bonds = np.sum(dists < 2.8) / 2
        else:
            coord_num, n_bonds = 0.0, 0

        z_coords = coords[:, 2] if len(coords) > 0 else np.array([0.0])
        en_vals = [_get_electronegativity(el) for el in elements] if elements else [0.0]

        feature_map = {
            "energy_per_atom": lambda: safe("energy_per_atom"),
            "energy_vdw_per_atom": lambda: safe("energy_vdw_per_atom"),
            "decomposition_energy": lambda: safe("decomposition_energy"),
            "exfoliation_energy_per_atom": lambda: safe("exfoliation_energy_per_atom"),
            "bandgap": lambda: safe("bandgap"),
            "total_magnetization": lambda: safe("total_magnetization"),
            "nelements": lambda: safe("nelements"),
            "sg_number": lambda: safe("sg_number"),
            "coord_num": lambda: coord_num,
            "packing_density": lambda: len(sites) / (vol + 1e-6),
            "bond_density": lambda: n_bonds / (vol + 1e-6),
            "layer_thickness": lambda: float(z_coords.max() - z_coords.min()),
            "mean_electronegativity": lambda: float(np.mean(en_vals)),
            "interlayer_spacing": lambda: (
                float(np.max(np.diff(np.sort(z_coords)))) if len(z_coords) > 1 else 0.0
            ),
            "std_x": lambda: float(np.std(coords[:, 0])) if len(coords) > 0 else 0.0,
            "std_y": lambda: float(np.std(coords[:, 1])) if len(coords) > 0 else 0.0,
            "std_z": lambda: float(np.std(z_coords)),
            "mean_z": lambda: float(np.mean(z_coords)),
        }

        features = []
        for name, _ in FEATURE_REGISTRY:
            if self.switches.get(name, True):
                features.append(feature_map[name]())
            else:
                features.append(0.0)

        return features

    def get_active_feature_count(self) -> int:
        return sum(1 for v in self.switches.values() if v)

    def get_feature_names(self) -> List[str]:
        return [name for name, _ in FEATURE_REGISTRY if self.switches.get(name, True)]
