# Model Evolution Log

## Condensed Evolution

- **Model C era** - unified baseline using Coulomb-matrix eigenspectra and amplitude encoding.
- **Model A era** - graph-based intermediate line using simpler GCN-style encoding and local Mo-focused work.
- **Model B era** - SchNet + physics-feature hybrid with the strongest modular structure and leakage-aware training design.
- **v4 planned** - future redesign informed by the limitations observed in the retained ModelB line.

## Key Retained Decisions

- Graph representations matter for periodic materials.
- Quantum layers need disciplined classical scaffolding to be meaningful.
- Documentation must distinguish implemented code from planned architecture.
- The project story is clearest when the model lineage is preserved without every intermediate copy of the source tree.
