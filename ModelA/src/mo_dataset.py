"""
Q2DM Mo-Family Dataset Loader
==============================

Filters 128 Mo-containing materials from 2DMatPedia db.json
Extracts all 28 features needed for hybrid model

CRITICAL: Verifies vdW energy loading (root cause of previous failures!)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

# Compact-repo local paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / 'data' / 'raw' / '2dmatpedia_full.json'
CACHE_PATH = BASE_DIR / 'data' / 'mo_dataset_cache.npz'


class MoDataset:
    """
    Mo-family 2D materials dataset
    
    Loads 128 Mo-containing materials with:
    - Structure (atomic coords, elements, lattice)
    - DFT energetics (vdW, formation, decomposition, total)
    - Electronic properties (bandgap, magnetization, etc.)
    - Target: exfoliation_energy_per_atom
    """
    
    def __init__(self, db_path=DB_PATH, cache=True):
        self.db_path = db_path
        self.materials = []
        self.use_cache = cache
        
        if cache and CACHE_PATH.exists():
            print(f"Loading from cache: {CACHE_PATH}")
            self._load_cache()
        else:
            print(f"Loading from database: {db_path}")
            self._load_from_db()
            if cache:
                self._save_cache()
        
        self._verify_critical_features()
    
    def _load_from_db(self):
        """Load Mo materials from JSON database"""
        with open(self.db_path, 'r') as f:
            all_data = [json.loads(line) for line in f]
        
        print(f"Total materials in database: {len(all_data)}")
        
        # Filter Mo-containing materials
        mo_count = 0
        for entry in all_data:
            if 'Mo' not in entry.get('elements', []):
                continue
            
            # Required fields check
            if not self._has_required_fields(entry):
                continue
            
            material = self._extract_material(entry)
            if material is not None:
                self.materials.append(material)
                mo_count += 1
        
        print(f"Mo-containing materials loaded: {mo_count}")
    
    def _has_required_fields(self, entry: Dict) -> bool:
        """Check if entry has all required fields"""
        required = [
            'structure',
            'exfoliation_energy_per_atom',  # Target
            'elements',
            'formula_pretty'
        ]
        return all(field in entry for field in required)
    
    def _extract_material(self, entry: Dict) -> Dict:
        """Extract relevant features from db entry"""
        try:
            material = {
                # Identifier
                'material_id': entry.get('material_id', 'unknown'),
                'formula': entry['formula_pretty'],
                
                # Structure (for graph construction)
                'structure': entry['structure'],
                'lattice': entry['structure']['lattice'],
                'sites': entry['structure']['sites'],
                
                # TARGET
                'exfoliation_energy': entry['exfoliation_energy_per_atom'],
                
                # DFT Energetics (Block 2 - 4 features)
                'vdw_energy': entry.get('energy_vdw_per_atom'),  # CRITICAL!
                'formation_energy': self._compute_formation_energy(entry),
                'decomposition_energy': entry.get('decomposition_energy'),
                'energy_per_atom': entry.get('energy_per_atom'),
                
                # Electronic Properties (Block 3 - 4 features)
                'bandgap': entry.get('bandgap'),
                'magnetization': entry.get('total_magnetization', 0.0),
                'bandstructure': entry.get('bandstructure', {}),
                
                # Metadata
                'nelements': entry.get('nelements'),
                'spacegroup': entry.get('sg_number')
            }
            
            # Validate target
            if material['exfoliation_energy'] is None:
                return None
            
            return material
            
        except Exception as e:
            warnings.warn(f"Failed to extract material: {e}")
            return None
    
    def _compute_formation_energy(self, entry: Dict) -> float:
        """Compute formation energy if not directly available"""
        # Formation energy might be in thermo data
        thermo = entry.get('thermo', {})
        if isinstance(thermo, dict):
            return thermo.get('formation_energy_per_atom')
        return entry.get('formation_energy_per_atom')
    
    def _verify_critical_features(self):
        """
        CRITICAL VERIFICATION: Ensure vdW energy loaded correctly
        
        Previous failures caused by missing this feature!
        """
        print("\n" + "="*60)
        print("CRITICAL FEATURE VERIFICATION")
        print("="*60)
        
        n_materials = len(self.materials)
        
        # Check vdW energy
        vdw_present = sum(1 for m in self.materials if m['vdw_energy'] is not None)
        vdw_missing = n_materials - vdw_present
        
        print(f"\n1. vdW Energy (ROOT CAUSE of previous failures!):")
        print(f"   Present: {vdw_present}/{n_materials} ({100*vdw_present/n_materials:.1f}%)")
        print(f"   Missing: {vdw_missing}")
        
        if vdw_missing > 0:
            print(f"     WARNING: {vdw_missing} materials missing vdW energy!")
            print(f"   This was the ROOT CAUSE of collapse to mean!")
            print(f"   Recommendation: Impute or filter these materials")
        else:
            print(f"    ALL materials have vdW energy - GOOD!")
        
        # Check other DFT features
        print(f"\n2. Formation Energy:")
        fe_present = sum(1 for m in self.materials if m['formation_energy'] is not None)
        print(f"   Present: {fe_present}/{n_materials} ({100*fe_present/n_materials:.1f}%)")
        
        print(f"\n3. Bandgap:")
        bg_present = sum(1 for m in self.materials if m['bandgap'] is not None)
        print(f"   Present: {bg_present}/{n_materials} ({100*bg_present/n_materials:.1f}%)")
        
        print(f"\n4. Exfoliation Energy (Target):")
        target_present = sum(1 for m in self.materials if m['exfoliation_energy'] is not None)
        print(f"   Present: {target_present}/{n_materials} (should be 100%)")
        
        # Statistics
        vdw_values = [m['vdw_energy'] for m in self.materials if m['vdw_energy'] is not None]
        exf_values = [m['exfoliation_energy'] for m in self.materials]
        
        print(f"\n5. Value Ranges:")
        print(f"   vdW Energy: {min(vdw_values):.3f} to {max(vdw_values):.3f} eV/atom")
        print(f"   Exfoliation: {min(exf_values):.3f} to {max(exf_values):.3f} eV/atom")
        
        # Correlation check
        if len(vdw_values) == len(exf_values):
            corr = np.corrcoef(vdw_values, exf_values)[0, 1]
            print(f"\n6. vdW vs Exfoliation Correlation: {corr:.3f}")
            if abs(corr) > 0.5:
                print(f"    Strong correlation - vdW is good predictor!")
            else:
                print(f"     Weak correlation - other features needed")
        
        print("\n" + "="*60 + "\n")
    
    def get_train_test_split(self, test_ratio=0.2, seed=42):
        """
        Split into train/test
        
        For 128 materials:
        - train: 102 (80%)
        - test: 26 (20%)
        """
        np.random.seed(seed)
        n = len(self.materials)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        split_idx = int(n * (1 - test_ratio))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        train_materials = [self.materials[i] for i in train_idx]
        test_materials = [self.materials[i] for i in test_idx]
        
        print(f"Split: {len(train_materials)} train, {len(test_materials)} test")
        
        return train_materials, test_materials
    
    def _save_cache(self):
        """Save processed data to cache"""
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to arrays for efficient storage
        data = {
            'materials': self.materials  # Store as list of dicts for now
        }
        np.savez_compressed(CACHE_PATH, **data)
        print(f"Saved cache to {CACHE_PATH}")
    
    def _load_cache(self):
        """Load from cache"""
        data = np.load(CACHE_PATH, allow_pickle=True)
        self.materials = data['materials'].tolist()
        print(f"Loaded {len(self.materials)} materials from cache")
    
    def __len__(self):
        return len(self.materials)
    
    def __getitem__(self, idx):
        return self.materials[idx]
    
    def summary(self):
        """Print dataset summary"""
        print(f"\n{'='*60}")
        print(f"Mo-Family Dataset Summary")
        print(f"{'='*60}")
        print(f"Total materials: {len(self.materials)}")
        
        # Atom count distribution
        atom_counts = [len(m['sites']) for m in self.materials]
        print(f"\nAtom counts:")
        print(f"  Min: {min(atom_counts)}")
        print(f"  Max: {max(atom_counts)}")
        print(f"  Mean: {np.mean(atom_counts):.1f}")
        print(f"  Median: {np.median(atom_counts):.0f}")
        
        # Formula distribution (top 10)
        from collections import Counter
        formulas = [m['formula'] for m in self.materials]
        print(f"\nTop 10 formulas:")
        for formula, count in Counter(formulas).most_common(10):
            print(f"  {formula}: {count}")
        
        print(f"{'='*60}\n")


def main():
    """Test dataset loading"""
    print("Loading Mo-family dataset...\n")
    
    dataset = MoDataset(cache=True)
    dataset.summary()
    
    # Test train/test split
    train, test = dataset.get_train_test_split()
    
    # Inspect first material
    print("\nExample material (first in dataset):")
    example = dataset[0]
    print(f"  Formula: {example['formula']}")
    print(f"  Atoms: {len(example['sites'])}")
    print(f"  Exfoliation energy: {example['exfoliation_energy']:.4f} eV/atom")
    print(f"  vdW energy: {example['vdw_energy']:.4f} eV/atom" if example['vdw_energy'] else "  vdW energy: MISSING!")
    print(f"  Bandgap: {example['bandgap']:.4f} eV" if example['bandgap'] else "  Bandgap: MISSING!")
    
    print("\n Dataset loading successful!")


if __name__ == "__main__":
    main()
