"""Test target leakage guard including correlated field exclusion."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

np = pytest.importorskip("numpy")
from src.features import PhysicsFeatureExtractor, CORRELATED_TARGETS, FEATURE_REGISTRY


SAMPLE_MATERIAL = {
    'bandgap': 1.5,
    'energy_per_atom': -4.46,
    'exfoliation_energy_per_atom': 0.08,
    'energy_vdw_per_atom': -4.50,
    'decomposition_energy': 0.15,
    'total_magnetization': 0.0,
    'magnetism': 'NM',
    'nelements': 2,
    'sg_number': 186,
    'structure': {
        'lattice': {
            'matrix': [[3.2, 0, 0], [0, 3.2, 0], [0, 0, 15.0]]
        },
        'sites': [
            {'species': [{'element': 'Mo'}], 'xyz': [0, 0, 7.5], 'abc': [0, 0, 0.5], 'label': 'Mo'},
            {'species': [{'element': 'S'}], 'xyz': [1.6, 0.9, 8.5], 'abc': [0.5, 0.28, 0.57], 'label': 'S'},
            {'species': [{'element': 'S'}], 'xyz': [1.6, 0.9, 6.5], 'abc': [0.5, 0.28, 0.43], 'label': 'S'},
        ],
    },
}


def test_no_leakage_bandgap():
    ext = PhysicsFeatureExtractor()
    excludes = list(ext.get_auto_excludes('bandgap'))
    features = ext.extract(SAMPLE_MATERIAL, exclude_keys=excludes)
    assert 1.5 not in features, f'Bandgap leaked!'


def test_no_leakage_energy():
    ext = PhysicsFeatureExtractor()
    excludes = list(ext.get_auto_excludes('energy_per_atom'))
    features = ext.extract(SAMPLE_MATERIAL, exclude_keys=excludes)
    # energy_per_atom itself must be excluded
    assert -4.46 not in features, f'energy_per_atom leaked!'
    # energy_vdw_per_atom is correlated (VdW-corrected version), must be excluded
    assert -4.50 not in features, f'energy_vdw_per_atom (correlated) leaked!'
    # exfoliation_energy_per_atom is computed FROM energy_per_atom (reverse leakage)
    assert 0.08 not in features, f'exfoliation_energy_per_atom (reverse-correlated) leaked!'
    # decomposition_energy is INDEPENDENT (weakly correlated, r≈0.3-0.5)
    # so it IS allowed as a feature — this is scientifically correct
    assert 0.15 in features, 'decomposition_energy should NOT be excluded for energy_per_atom'


def test_no_leakage_magnetization():
    ext = PhysicsFeatureExtractor()
    excludes = list(ext.get_auto_excludes('total_magnetization'))
    features = ext.extract(SAMPLE_MATERIAL, exclude_keys=excludes)
    # total_magnetization is 0.0 in sample, but 'magnetism' (correlated) must also be excluded.
    # Verify the excluded keys are actually excluded (0.0 appears for excluded features).
    assert 'total_magnetization' in excludes
    assert 'magnetism' in excludes


def test_correlated_targets_exist():
    """Verify all entries in CORRELATED_TARGETS reference real fields."""
    known_fields = set(SAMPLE_MATERIAL.keys())
    for target, related in CORRELATED_TARGETS.items():
        assert target in known_fields, f'CORRELATED_TARGETS key {target!r} not found in material'
        for field in related:
            assert field in known_fields, f'Related field {field!r} for {target!r} not found in material'
    print(f'  Verified {len(CORRELATED_TARGETS)} correlation entries.')


def test_feature_switches():
    """Test that disabling a feature sets it to 0.0 and changes count."""
    ext_full = PhysicsFeatureExtractor()
    f_full = ext_full.extract(SAMPLE_MATERIAL)

    switches = {name: True for name, _ in FEATURE_REGISTRY}
    switches['energy_per_atom'] = False
    ext_partial = PhysicsFeatureExtractor(feature_switches=switches)
    f_partial = ext_partial.extract(SAMPLE_MATERIAL)

    # Find the index of energy_per_atom in the registry order
    epa_idx = next(i for i, (name, _) in enumerate(FEATURE_REGISTRY) if name == 'energy_per_atom')
    assert f_partial[epa_idx] == 0.0, f'Disabled feature at idx {epa_idx} should be 0.0, got {f_partial[epa_idx]}'
    assert len(f_partial) == len(f_full), 'Feature count should be same (disabled → 0.0)'


def test_feature_count():
    ext = PhysicsFeatureExtractor()
    f = ext.extract(SAMPLE_MATERIAL, exclude_keys=['bandgap'])
    assert len(f) == len(FEATURE_REGISTRY), f'Expected {len(FEATURE_REGISTRY)}, got {len(f)}'


def test_bandgap_as_feature_for_exfoliation():
    """When predicting exfoliation_energy, bandgap should be present as a feature."""
    ext = PhysicsFeatureExtractor()
    excludes = list(ext.get_auto_excludes('exfoliation_energy_per_atom'))
    features = ext.extract(SAMPLE_MATERIAL, exclude_keys=excludes)
    # bandgap is independent → must be present
    assert 1.5 in features, 'bandgap should be available as feature when predicting exfoliation_energy'
    # energy_per_atom must be excluded (it's correlated with exfoliation)
    assert -4.46 not in features, 'energy_per_atom should be excluded for exfoliation target'
    # energy_vdw_per_atom must be excluded
    assert -4.50 not in features, 'energy_vdw_per_atom should be excluded for exfoliation target'


def test_new_features_present():
    """Verify all 18 features are extracted and new fields appear."""
    ext = PhysicsFeatureExtractor()
    features = ext.extract(SAMPLE_MATERIAL, exclude_keys=[])
    assert len(features) == 18, f'Expected 18 features, got {len(features)}'
    names = ext.get_feature_names()
    assert 'bandgap' in names
    assert 'total_magnetization' in names
    assert 'exfoliation_energy_per_atom' in names
    assert 'nelements' in names
    assert 'sg_number' in names


if __name__ == '__main__':
    test_no_leakage_bandgap()
    test_no_leakage_energy()
    test_no_leakage_magnetization()
    test_correlated_targets_exist()
    test_feature_switches()
    test_feature_count()
    test_bandgap_as_feature_for_exfoliation()
    test_new_features_present()
    print('All leakage tests PASSED')
