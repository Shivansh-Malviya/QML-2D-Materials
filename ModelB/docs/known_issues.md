# ModelB Limitations

These retained implementation risks keep the ModelB claims appropriately scoped.

## Critical

### MB-RUNTIME-01: OpenMP/MKL Library Collision

Some Windows/Conda/Pip environments can trigger an OpenMP runtime collision:

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

The durable fix is a single-source dependency stack for numerical packages rather than mixing incompatible OpenMP runtimes.

### Device Mismatch Risk In `model.py`

PennyLane QNode output can be CPU-backed while downstream normalization layers may be CUDA-backed. GPU-backed runs need explicit tensor-device checks during debugging.

### Best-Checkpoint Metric Reporting

The training loop can return last-epoch validation metrics after restoring best weights. Final reporting should use metrics recomputed from the restored best weights.

## High

### Gradient Norm Logging

Gradient norm logging reflects the final batch rather than a full-epoch aggregate. These values are diagnostic hints, not formal barren-plateau evidence.

### Correlated-Feature Exclusions

Feature exclusions remain target- and dataset-dependent. The retained tests cover the current sample contract but do not prove every possible future target mapping.

## Medium

### MAPE For Zero Targets

MAPE is unstable for metallic materials with zero or near-zero bandgap. MAE, MSE, and R2 are more defensible for those cases.

### Resume Edge Cases

Checkpoint resume behavior can skip the training loop when the resumed epoch is already beyond the requested epoch budget.
