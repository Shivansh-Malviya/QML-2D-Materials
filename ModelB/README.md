# ModelB

ModelB is the primary retained Graph-QNN line for Q2DM.

## Role

ModelB contains the strongest split-aware preprocessing, target-correlation exclusions, and leakage tests in the project.

## Retained Material

- Modular source in `src/`.
- Tests in `tests/`.
- Data-directory placeholders.
- One representative result capsule under `artifacts/` containing config, training history, and plots.
- Experimental VQC code in `experimental/`.
- Future redesign notes in `v4_planned/`.

## Data Boundary

Full raw and processed datasets are not versioned. The expected dataset-derived paths are:

```text
ModelB/data/raw/2dmatpedia_full.json
ModelB/data/processed/dataset.npz
```

## Evidence Notes

`tests/test_leakage.py` records the strongest retained guardrail: feature construction should not include target-correlated fields such as decomposition-energy proxies for bandgap prediction.
