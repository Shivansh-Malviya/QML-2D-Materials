# Data Contract (Model C)

## Core Dataset Filter
Model C operates on a filtered subset of the full 2DMatPedia dataset.

The raw dataset `2dmatpedia_full.json` contains 6,351 materials. However, Model C's `run_project.py` applies specific filtering criteria:

1.  **Atom Count Limit:** Materials with more than `MAX_ATOMS = 40` are excluded. This bounds the dimensionality of the Coulomb matrices.
2.  **Target Existence:** Materials lacking the required target property (e.g., `exfoliation_energy` or `bandgap` depending on the active configuration) are excluded.

## Expected Record Count
After applying these filters to the standard full dataset for the `exfoliation_energy` target, exactly **4,527 records** are retained.

This count is strictly enforced by `ModelC/tests/test_data_contract.py`. Any modification to the data parsing logic or target selection must ensure this contract is either maintained or explicitly renegotiated.
