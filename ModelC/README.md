# ModelC

ModelC is the reference baseline using Coulomb-matrix eigenspectra and amplitude encoding.

## Role

ModelC shows the older global-descriptor baseline that helped anchor the project's early comparisons. The project later moved toward graph-based pipelines, but ModelC remains useful as a reference point.

## Scope

The repository keeps the source, docs, and representative plots. Raw dataset files and generated parameter arrays are not versioned.

## Data Boundary

Expected dataset-derived path:

```text
ModelC/data/raw/2dmatpedia_full.json
```

## Evidence Notes

`tests/test_data_contract.py` records the filtered 4,527-record contract for the standard full dataset and active target configuration.
