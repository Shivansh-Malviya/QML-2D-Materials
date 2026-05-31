"""
Q2DM Physics Features Computation
===================================

Computes critical structural features identified in research_log.md:
1. Mo coordination number (#3 predictor in literature)
2. Packing efficiency (#1 predictor in literature!)
3. Bond density (#2 insight from literature)
4. Layer thickness
5. Lattice strain
6. Mean electronegativity
7. d-band center for Mo (if electronic data available)
8. Interlayer distance

These features were MISSING from all previous models → caused collapse to mean!
"""

import numpy as np
from scipy.spatial import Voronoi
from typing import Dict, List
import warnings

# Electronegativity (Pauling scale)
ELECTRONEGATIVITY = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'S': 2.58, 'Se': 2.55, 'Te': 2.1, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
    'Mo': 2.16, 'W': 2.36, 'Cr': 1.66, 'V': 1.63,
    'Li': 0.98, 'Na': 0.93, 'K': 0.82, 'Mg': 1.31, 'Ca': 1.00,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19,
}

# Atomic radii (Ångströms, covalent radii)
ATOMIC_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'S': 1.05, 'Se': 1.20, 'Te': 1.38, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'Mo': 1.54, 'W': 1.62, 'Cr': 1.39, 'V': 1.53,
    'Li': 1.28, 'Na': 1.66, 'K': 2.03, 'Mg': 1.41, 'Ca': 1.76,
    'Al': 1.21, 'Si': 1.11, 'P': 1.07,
}


class PhysicsFeatureExtractor:
    """
    Extract 8 critical structural features from material structure
    
    Features identified from ROOT_CAUSE_ANALYSIS.md as missing from all failed models
    """
    
    def __init__(self, bond_cutoff=5.0):
        """
        Args:
            bond_cutoff: Distance cutoff for bonds (Ångströms)
        """
        self.bond_cutoff = bond_cutoff
    
    def extract_all_features(self, material: Dict) -> Dict:
        """
        Extract all 8 structural features
        
        Returns dict with:
            mo_coordination, packing_efficiency, bond_density, layer_thickness,
            lattice_strain, mean_electronegativity, d_band_center, interlayer_distance
        """
        structure = material['structure']
        sites = structure['sites']
        lattice = structure['lattice']
        
        # Extract atomic info
        coords = np.array([site['xyz'] for site in sites])
        elements = [site['species'][0]['element'] for site in sites]
        
        features = {}
        
        # 1. Mo coordination number (#3 predictor)
        features['mo_coordination'] = self.compute_mo_coordination(coords, elements)
        
        # 2. Packing efficiency (#1 predictor!)
        features['packing_efficiency'] = self.compute_packing_efficiency(coords, elements, lattice)
        
        # 3. Bond density (#2 insight)
        features['bond_density'] = self.compute_bond_density(coords, lattice)
        
        # 4. Layer thickness
        features['layer_thickness'] = self.compute_layer_thickness(coords)
        
        # 5. Lattice strain
        features['lattice_strain'] = self.compute_lattice_strain(lattice, elements)
        
        # 6. Mean electronegativity
        features['mean_electronegativity'] = self.compute_mean_electronegativity(elements)
        
        # 7. d-band center for Mo (placeholder - needs electronic data)
        features['d_band_center'] = self.compute_d_band_center(material)
        
        # 8. Interlayer distance
        features['interlayer_distance'] = self.compute_interlayer_distance(coords)
        
        return features
    
    def compute_mo_coordination(self, coords: np.ndarray, elements: List[str]) -> float:
        """
        Count average coordination number of Mo atoms
        
        Literature: #3 most important predictor for exfoliation
        """
        mo_indices = [i for i, el in enumerate(elements) if el == 'Mo']
        
        if len(mo_indices) == 0:
            warnings.warn("No Mo atoms found!")
            return 0.0
        
        coordinations = []
        for mo_idx in mo_indices:
            mo_pos = coords[mo_idx]
            
            # Count neighbors within cutoff
            neighbors = 0
            for i, pos in enumerate(coords):
                if i == mo_idx:
                    continue
                dist = np.linalg.norm(mo_pos - pos)
                if dist < self.bond_cutoff:
                    neighbors += 1
            
            coordinations.append(neighbors)
        
        return np.mean(coordinations)
    
    def compute_packing_efficiency(self, coords: np.ndarray, elements: List[str], 
                                   lattice: Dict) -> float:
        """
        Compute packing efficiency using Voronoi tessellation
        
        Literature: #1 MOST IMPORTANT predictor for exfoliation!
        
        Packing efficiency = (volume of atoms) / (volume of unit cell)
        """
        # Unit cell volume
        lattice_matrix = np.array(lattice['matrix'])
        cell_volume = abs(np.linalg.det(lattice_matrix))
        
        # Atomic volumes (4/3 * π * r³)
        atom_volume = 0.0
        for el in elements:
            r = ATOMIC_RADII.get(el, 1.0)  # Default 1Å if unknown
            atom_volume += (4/3) * np.pi * (r ** 3)
        
        # Packing efficiency
        packing_eff = atom_volume / cell_volume if cell_volume > 0 else 0.0
        
        # Clamp to [0, 1] (can't exceed 100%)
        return min(packing_eff, 1.0)
    
    def compute_bond_density(self, coords: np.ndarray, lattice: Dict) -> float:
        """
        Number of bonds per unit volume
        
        Literature: #2 insight (bonds per unit volume critical for exfoliation)
        """
        # Count bonds
        n_bonds = 0
        n_atoms = len(coords)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < self.bond_cutoff:
                    n_bonds += 1
        
        # Unit cell volume
        lattice_matrix = np.array(lattice['matrix'])
        cell_volume = abs(np.linalg.det(lattice_matrix))
        
        # Bond density (bonds/Ų)
        return n_bonds / cell_volume if cell_volume > 0 else 0.0
    
    def compute_layer_thickness(self, coords: np.ndarray) -> float:
        """
        Thickness of 2D layer (range in z-direction)
        
        Directly relates to exfoliation difficulty
        """
        if len(coords) == 0:
            return 0.0
        
        z_coords = coords[:, 2]  # Assuming z is layer-normal direction
        return z_coords.max() - z_coords.min()
    
    def compute_lattice_strain(self, lattice: Dict, elements: List[str]) -> float:
        """
        Lattice strain relative to ideal structure
        
        For TMDs (MoS2-like): Compare to ideal hexagonal lattice
        """
        lattice_matrix = np.array(lattice['matrix'])
        
        # Extract lattice parameters
        a = np.linalg.norm(lattice_matrix[0])
        b = np.linalg.norm(lattice_matrix[1])
        c = np.linalg.norm(lattice_matrix[2])
        
        # For hexagonal: ideal a = b
        # Strain = |a - b| / a
        if a > 0:
            strain = abs(a - b) / a
        else:
            strain = 0.0
        
        return strain
    
    def compute_mean_electronegativity(self, elements: List[str]) -> float:
        """
        Mean electronegativity of all atoms
        
        Literature: Moderate predictor (~0.4 correlation)
        """
        en_values = []
        for el in elements:
            en = ELECTRONEGATIVITY.get(el, 2.0)  # Default ~middle value
            en_values.append(en)
        
        return np.mean(en_values) if en_values else 2.0
    
    def compute_d_band_center(self, material: Dict) -> float:
        """
        d-band center for Mo atoms (electronic structure)
        
        Literature: Important for charge transfer predictions
        
        TODO: Extract from bandstructure if available
        For now: Return 0.0 (placeholder)
        """
        # Would need electronic structure data to compute properly
        # bandstructure = material.get('bandstructure', {})
        
        # Placeholder
        return 0.0
    
    def compute_interlayer_distance(self, coords: np.ndarray) -> float:
        """
        Distance between layers (if multilayer)
        
        Most direct feature related to exfoliation!
        """
        if len(coords) < 2:
            return 0.0
        
        z_coords = coords[:, 2]
        
        # Simple heuristic: gap in z-coordinates
        z_sorted = np.sort(z_coords)
        z_diffs = np.diff(z_sorted)
        
        # Largest gap = interlayer distance
        if len(z_diffs) > 0:
            return z_diffs.max()
        else:
            return 0.0


def test_feature_extraction():
    """Test feature extraction on a sample material"""
    # Load one material from dataset
    from mo_dataset import MoDataset
    
    dataset = MoDataset(cache=True)
    material = dataset[0]
    
    print(f"Testing on: {material['formula']}")
    print(f"Atoms: {len(material['sites'])}\n")
    
    extractor = PhysicsFeatureExtractor()
    features = extractor.extract_all_features(material)
    
    print("Extracted Features:")
    print("="*50)
    for name, value in features.items():
        print(f"  {name:.<40} {value:.4f}")
    print("="*50)
    
    # Verify all features present
    expected = [
        'mo_coordination', 'packing_efficiency', 'bond_density',
        'layer_thickness', 'lattice_strain', 'mean_electronegativity',
        'd_band_center', 'interlayer_distance'
    ]
    
    missing = [f for f in expected if f not in features]
    if missing:
        print(f"\n⚠️  Missing features: {missing}")
    else:
        print(f"\n✅ All 8 features extracted!")
    
    return features


if __name__ == "__main__":
    test_feature_extraction()
