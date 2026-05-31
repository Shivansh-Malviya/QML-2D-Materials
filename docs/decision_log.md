# Technical Decisions

This log records modeling and documentation decisions that affect scientific interpretation.

| Decision ID | Decision | Rationale | Status |
|-------------|----------|-----------|--------|
| DEC-001 | Organize the project around ModelA, ModelB, and ModelC. | Keeps the model lineage readable and separates baseline, intermediate, and mainline work. | Active |
| DEC-002 | Exclude full raw and processed third-party datasets from Git. | Dataset/database terms and attribution obligations are separate from this repo's MIT license. | Active |
| DEC-003 | Preserve selected result artifacts. | They document representative outputs without requiring every experiment directory. | Active |
| DEC-004 | Keep ModelB as the main retained line. | It contains the strongest split-aware preprocessing and leakage guardrails. | Active |
| DEC-005 | Keep VQC code experimental unless execution evidence supports production claims. | The experimental branch is scientifically useful but not the main tested pipeline. | Active |
| DEC-006 | Document ModelA as cache-dependent. | The retained ModelA line depends on a dataset-derived Mo-family cache. | Active |
| DEC-007 | Treat ModelC as a reference baseline. | It is retained for comparison, not as the active mainline. | Active |

## Residual Limitations

- Dataset-backed training depends on third-party materials data outside this repository.
- Selected artifacts represent historical runs; they are evidence snapshots, not a complete experiment log.
- Environment issues such as OpenMP runtime conflicts may appear on some Windows/Conda/Pip setups.
