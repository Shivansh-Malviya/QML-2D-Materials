# QML-2D-Materials

QML-2D-Materials is my Q2DM project: a sequence of hybrid graph and quantum machine-learning experiments for structure-property modeling in two-dimensional materials.

The repository is organized as a project showcase. It presents the model lineage, retained source code, validation checks, plots, and technical notes that document the work.

## Project Focus

- Hybrid quantum/classical modeling for 2D materials.
- Graph-based crystal representations and compact physics features.
- Leakage-aware training design, especially fold-scoped normalization.
- Comparison between older global-descriptor baselines and later graph-QNN pipelines.

## Model Lines

- **ModelA** - intermediate GCN-style graph model with angle-encoded quantum features.
- **ModelB** - primary Graph-QNN line using SchNet-style graph encoding, physics features, and leakage checks.
- **ModelC** - Coulomb-matrix/amplitude-encoding baseline retained as an early reference point.
- **ModelB/v4_planned** - design notes for a future ModelB redesign; not an implemented model.

## Evidence Artifacts

- Source code for the retained model lines.
- Tests for feature behavior, leakage checks, model contracts, and data-shape assumptions.
- Representative result artifacts in `ModelA/artifacts/`, `ModelB/artifacts/`, and `ModelC/artifacts/`.
- Technical notes under `docs/` describing architecture, evolution, data provenance, and limitations.

## Data Boundary

Full third-party materials datasets, processed dataset caches, paper PDFs, and trained checkpoint binaries are not part of the repository. Dataset-derived paths are documented for provenance only:

- `ModelB/data/raw/2dmatpedia_full.json`
- `ModelB/data/processed/dataset.npz`
- `ModelC/data/raw/2dmatpedia_full.json`
- `ModelA/data/mo_dataset_cache.npz`

See [docs/data_provenance.md](docs/data_provenance.md) and [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## Key Files

- `ModelB/run.py` - primary ModelB training entrypoint.
- `ModelB/tests/test_leakage.py` - target-leakage guard using synthetic sample material.
- `ModelA/src/train_a.py` - ModelA training script.
- `ModelC/src/run_project.py` - ModelC baseline script.
- `docs/proof_of_work_index.md` - index of retained result artifacts.

## License

Code and original documentation are distributed under the MIT License in [LICENSE](LICENSE). Third-party datasets, scientific papers, model weights, and generated artifacts remain subject to their own provenance, citation, and redistribution boundaries.
