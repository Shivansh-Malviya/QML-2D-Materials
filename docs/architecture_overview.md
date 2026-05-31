# Architecture Evolution Overview

The Q2DM project evolved through three distinct architectural phases. This compact repository retains all three to demonstrate the progression from simple baseline, to experimental graph representation, to a mature, feature-rich hybrid quantum model.

## Evolutionary Sequence

| Phase | Model | Representation | Description | Detailed Architecture |
|---|---|---|---|---|
| **Phase 1** | Model C | Global Descriptor | Baseline using Coulomb-matrix eigenspectra and amplitude encoding. | See `ModelC/docs/architecture_overview.md` |
| **Phase 2** | Model A | Graph | Intermediate line using a simpler GCN-style encoder and angle encoding. | See `ModelA/docs/architecture_overview.md` |
| **Phase 3** | Model B | Hybrid Graph-QNN | The primary mainline using SchNet, physics features, and a VQC hybrid. | See `ModelB/docs/architecture_overview.md` |

*(Note: Detailed architectural discussions, quantum encodings, and design rationale are located in the model-specific `docs/architecture_overview.md` files.)*
