# Model A

ModelA is an intermediate graph-QNN line using a simpler GCN-style encoder and angle encoding.

## Role

ModelA documents a meaningful stage between the older Coulomb-matrix baseline and the more disciplined SchNet-based ModelB line.

## Retained Material

- Modular source files under `src/`.
- Helper files needed by the retained snapshot.
- Representative result artifacts under `artifacts/`.
- A cache-shape test for the historical Mo-family processed cache.

## Data Boundary

The historical ModelA cache was an 86-record Mo-family cache at `data/mo_dataset_cache.npz`. That cache is dataset-derived and not versioned.

See `docs/data_provenance.md` and the root `THIRD_PARTY_DATA.md`.

## Evidence Notes

`tests/test_cache.py` records the expected cache shape and graph-construction assumptions when the dataset-derived cache is present.
