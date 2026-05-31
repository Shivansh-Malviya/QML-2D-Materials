# Parameter Rationale

This document records the ModelB parameters most relevant to understanding the retained code.

## ModelB

- `n_qubits` is a practical expressivity-vs-trainability choice, not a solved optimum.
- SchNet hidden dimensions and interaction depth matter because they govern the classical encoder before the quantum stage.
- Compression is necessary, but overly aggressive compression was a recurring failure mode in earlier lines.
