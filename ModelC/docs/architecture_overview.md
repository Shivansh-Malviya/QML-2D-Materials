# Model C: Architecture Overview

## Representation
Model C represents materials using the **Coulomb-matrix eigenspectrum**, a **global descriptor**. The Coulomb matrix captures the electrostatic interactions between all pairs of nuclei in the system. The sorted eigenspectrum of this matrix provides a rotationally and translationally invariant global representation of the material.

## Encoder
Model C does **not use a classical neural network encoder**. The eigenspectrum is computed deterministically and fed directly into the quantum layer.

## Quantum Encoding
Model C uses **amplitude encoding**. The classical vector (the eigenspectrum) is normalized and embedded into the amplitudes of a quantum state. This is highly space-efficient, encoding an $N$-dimensional vector into $\log_2(N)$ qubits.

## Role in the Pipeline
Model C is retained as the **reference baseline**. It represents the Phase 1 attempt at a quantum-classical hybrid, providing a benchmark for the graph-based approaches (Models A and B) that followed.
