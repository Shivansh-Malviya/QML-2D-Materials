# Model B: Architecture Overview

## Representation
Model B models materials as **graphs augmented with physics features**. It incorporates explicit physical properties (e.g., electronegativity, atomic radius) alongside the graph topology.

## Encoder
It uses **SchNet** (a continuous-filter convolutional neural network). SchNet is specifically designed for modeling quantum interactions in molecules and materials, making it highly effective for capturing complex interatomic potentials and respecting periodic boundary conditions (PBC).

## Quantum Encoding
Model B employs a **VQC (Variational Quantum Circuit) hybrid** approach. The classical SchNet encoder produces a latent representation, which is then fed into a parameterized quantum circuit. The VQC acts as a highly expressive, non-linear transformation layer.

## Role in the Pipeline
Model B is the **primary retained mainline**.
It is the line where the project became most structurally disciplined: modular source files, explicit tests, PBC-aware graph construction, richer feature controls, and a more inspectable train/eval pipeline.

## Architectural Lessons Learned
The decisive question during Model B's evolution was not only "quantum or classical?" but also:
- Are the cross-validation folds leak-safe?
- Are the features scoped correctly (e.g., excluding target-correlated features like `decomposition_energy`)?
- Is the graph representation physically faithful enough (PBC)?
