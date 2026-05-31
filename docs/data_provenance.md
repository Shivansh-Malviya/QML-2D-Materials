# Data Provenance

The project uses third-party 2D materials records and derived local training caches. Those source datasets are independent research/database assets and are not covered by the MIT license for this code.

## Dataset-Derived Inputs

The code references these dataset-derived paths:

- `ModelB/data/raw/2dmatpedia_full.json`
- `ModelB/data/processed/dataset.npz`
- `ModelC/data/raw/2dmatpedia_full.json`
- `ModelA/data/mo_dataset_cache.npz`

The repository keeps placeholders for the expected structure but does not include the dataset files themselves.

## Targets

Targets explored across the project include:

- `bandgap`
- `energy_per_atom`
- `exfoliation_energy_per_atom`

## Handling Principles

- Dataset-derived files remain outside version control.
- Source database and paper citations remain separate from the repository's MIT license.
- Normalization statistics are scoped to training folds in the ModelB line to reduce target leakage risk.
