# Project History

This document summarizes the project's most important technical turns.

## Timeline

| Phase | What happened | Why it mattered |
|---|---|---|
| Early baselines | Coulomb-matrix and compressed-QML experiments | Established a reference point but exposed collapse and representation limits |
| Unified baseline / Model C | Amplitude-encoding baseline achieved partial non-collapse | Proved a smaller reference model could behave sensibly |
| Early graph attempts | Graph-based models improved structure representation but still suffered from compression and implementation issues | Motivated a more disciplined modular rebuild |
| `Q2DM_v2` modularization | The pipeline was split into `config`, `data`, `features`, `model`, `train`, and tests | Made targeted debugging and validation possible |
| Validation pass | Leakage, normalization, feature exclusions, parameter rationale, and documentation claims were re-checked | Shifted the project from narrative confidence to evidence-grounded correction |
| Model-line organization | The work was organized around ModelA, ModelB, and ModelC | Makes the technical progression easier to inspect |

## Corrections That Changed The Story

- The newest packaged folder was not automatically the best source of truth.
- Some earlier claims about parameter counts, feature behavior, and model status required explicit re-verification.
- Version-scoped discrepancy and fix notes were useful during development, but the final project story is clearer when organized by model line.

## Why `production_pipeline.py` Was Not Promoted

A classical production-style script existed in earlier work. It was not promoted as a fourth active code track because the repository is deliberately organized around the A/B/C model taxonomy. Its value survives as historical and methodological context rather than a separate maintained model line.
