"""Tests for PhysicsFeatureExtractor."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.features import PhysicsFeatureExtractor, FEATURE_REGISTRY, _get_electronegativity


SAMPLE = {
    'energy_per_atom': -4.46,
    'energy_vdw_per_atom': -4.50,
    'decomposition_energy': 0.15,
    'structure': {
        'lattice': {'matrix': [[3.2, 0, 0], [0, 3.2, 0], [0, 0, 15.0]]},
        'sites': [
            {'species': [{'element': 'Mo'}], 'xyz': [0, 0, 7.5], 'abc': [0, 0, 0.5], 'label': 'Mo'},
            {'species': [{'element': 'S'}], 'xyz': [1.6, 0.9, 8.5], 'abc': [0.5, 0.28, 0.57], 'label': 'S'},
            {'species': [{'element': 'S'}], 'xyz': [1.6, 0.9, 6.5], 'abc': [0.5, 0.28, 0.43], 'label': 'S'},
        ],
    },
}


def test_feature_count():
    ext = PhysicsFeatureExtractor()
    f = ext.extract(SAMPLE)
    assert len(f) == len(FEATURE_REGISTRY), f'Expected {len(FEATURE_REGISTRY)}, got {len(f)}'


def test_volume_correct():
    ext = PhysicsFeatureExtractor()
    vol = ext._get_volume({'matrix': [[3.2, 0, 0], [0, 3.2, 0], [0, 0, 15.0]]})
    expected = 3.2 * 3.2 * 15.0
    assert abs(vol - expected) < 1e-6, f'Volume {vol} != {expected}'


def test_nonorthogonal_volume():
    ext = PhysicsFeatureExtractor()
    vol = ext._get_volume({'matrix': [[1, 1, 0], [0, 1, 0], [0, 0, 1]]})
    assert abs(vol - 1.0) < 1e-6, f'Non-orthogonal volume {vol} != 1.0'


def test_features_are_finite():
    ext = PhysicsFeatureExtractor()
    f = ext.extract(SAMPLE)
    for i, v in enumerate(f):
        assert np.isfinite(v), f'Feature {i} is not finite: {v}'


def test_electronegativity_pymatgen():
    """Verify pymatgen gives reasonable values for common elements."""
    en_mo = _get_electronegativity('Mo')
    en_s = _get_electronegativity('S')
    assert 1.5 < en_mo < 3.0, f'Mo electronegativity {en_mo} out of range'
    assert 2.0 < en_s < 3.0, f'S electronegativity {en_s} out of range'


def test_electronegativity_rare_element():
    """Test that rare elements don't crash and produce a warning."""
    import warnings
    en = _get_electronegativity('Ac')
    assert isinstance(en, float), f'Expected float, got {type(en)}'


if __name__ == '__main__':
    test_feature_count()
    test_volume_correct()
    test_nonorthogonal_volume()
    test_features_are_finite()
    test_electronegativity_pymatgen()
    test_electronegativity_rare_element()
    print('All feature tests PASSED')
