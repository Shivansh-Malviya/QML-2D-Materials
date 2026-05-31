# Research Log

## Overview

This document summarizes the research arc behind Q2DM.

## Model Evolution

- Early baselines used global descriptors and helped expose representation limits.
- ModelA introduced graph-style processing and angle-encoded quantum components.
- ModelB became the main retained line because it better separates data loading, feature extraction, model definition, and training logic.
- ModelC remains as a useful historical baseline.

## Key Lessons

- Newer packaging is not automatically better than older, better-tested behavior.
- Target leakage and normalization placement are central scientific risks in this project.
- VQC components should be treated as experimental unless backed by stronger execution and regression evidence.
- Documentation should distinguish implemented code, planned work, and historical evidence artifacts.

## Current Scope

The repository keeps source, tests, documentation, selected plots, and training histories. Full third-party datasets and paper PDFs remain outside version control.

## Remaining Limitations

- Dataset-backed runs depend on third-party materials data outside this repository.
- ModelA's retained snapshot depends on a local Mo-family cache.
- Some long-form historical training runs are represented by selected artifacts rather than complete run directories.
