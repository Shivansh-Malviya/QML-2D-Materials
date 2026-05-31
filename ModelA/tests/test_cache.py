import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import pytest

np = pytest.importorskip("numpy")

def test_modela_cache():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    cache_path = os.path.join(base_dir, 'data', 'mo_dataset_cache.npz')

    if not os.path.exists(cache_path):
        pytest.skip("ModelA dataset-derived cache is absent; cache-backed contract skipped.")
    
    # Check expected keys
    data = np.load(cache_path, allow_pickle=True)
    assert 'materials' in data.files, "Cache must contain 'materials' key"
    
    mats = data['materials'].tolist()
    
    # Record count check - empirically found to be 86 valid materials from db.json
    assert len(mats) == 86, f"Expected 86 materials, got {len(mats)}"
    
    # Check target unit sanity and structure
    exf_energies = []
    for item in mats:
        assert isinstance(item, dict)
        assert 'exfoliation_energy' in item
        assert 'formula' in item
        assert 'structure' in item
        assert 'vdw_energy' in item
        
        exf_energies.append(item['exfoliation_energy'])
        
    exf_energies = np.array(exf_energies)
    assert np.all(exf_energies >= -5.0) and np.all(exf_energies <= 5.0), "Exfoliation energies outside reasonable sanity range (-5 to 5 eV/atom)"
    
    # Graph-construction smoke test if feasible
    import sys
    src_dir = os.path.join(base_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        
    try:
        from graph_constructor import create_graph_with_pbc
        from physics_features import PhysicsFeatureExtractor
    except ImportError:
        pytest.skip("Could not import ModelA src modules")
        
    extractor = PhysicsFeatureExtractor()
    feat = extractor.extract_all_features(mats[0])
    assert len(feat) == 8, f"Expected 8 physics features, got {len(feat)}"
    
    # Just checking first few for smoke test
    for i in range(min(3, len(mats))):
        graph = create_graph_with_pbc(mats[i])
        assert hasattr(graph, 'x') and hasattr(graph, 'edge_index'), "Graph object invalid"
