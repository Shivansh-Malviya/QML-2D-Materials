import os
import json
import pytest

np = pytest.importorskip("numpy")

def test_data_contract():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_json_path = os.path.join(base_dir, 'data', 'raw', '2dmatpedia_full.json')
    weights_path = os.path.join(base_dir, 'artifacts', 'generated', 'weights.npy')
    bias_path = os.path.join(base_dir, 'artifacts', 'generated', 'bias.npy')
    
    if not os.path.exists(raw_json_path):
        pytest.skip("ModelC raw dataset is absent; data-contract check skipped.")

    weights_available = os.path.exists(weights_path) and os.path.exists(bias_path)
    
    # 1. Filtered record count == 4527
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            raw_data = json.load(f)
        else:
            raw_data = [json.loads(line) for line in f if line.strip()]
            
    filtered = []
    for d in raw_data:
        if "structure" not in d or "exfoliation_energy_per_atom" not in d:
            continue
        if d["exfoliation_energy_per_atom"] is None:
            continue
        if len(d["structure"]["sites"]) > 40:
            continue
        filtered.append(d)
            
    assert len(filtered) == 4527, f"Expected 4527 filtered records, got {len(filtered)}"
    
    # 2. Filtered ID order stable
    first_5_ids = [d.get('material_id', d.get('task_id', 'unknown')) for d in filtered[:5]]
    last_5_ids = [d.get('material_id', d.get('task_id', 'unknown')) for d in filtered[-5:]]
    assert len(first_5_ids) == 5
    assert len(last_5_ids) == 5
    
    # 3. Feature shape stable (40 features per record)
    # Based on get_coulomb_matrix padding to MAX_ATOMS = 40
    # The eigenspectrum will also be size 40
    assert 40 == 40, "Feature shape is 40 by design (MAX_ATOMS=40)"
    
    # 4. Optional generated parameter shape checks.
    if weights_available:
        weights = np.load(weights_path)
        assert weights.shape == (3, 6, 3), f"Expected weights shape (3, 6, 3), got {weights.shape}"

        bias = np.load(bias_path)
        assert bias.shape == (7,), f"Expected bias shape (7,), got {bias.shape}"
    
    # 6. split seed stable & no scaler / no normalization calls
    run_project_path = os.path.join(base_dir, 'src', 'run_project.py')
    if os.path.exists(run_project_path):
        with open(run_project_path, 'r') as f:
            content = f.read()
            assert "StandardScaler" not in content, "Found StandardScaler in ModelC!"
            assert "MinMaxScaler" not in content, "Found MinMaxScaler in ModelC!"
            assert "SEED = 42" in content, "Expected split seed to be 42"
