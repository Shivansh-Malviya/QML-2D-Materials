# Technical Deep Dive

## Overview

Q2DM explores graph-based and hybrid quantum-classical models for two-dimensional materials property prediction. The retained source contrasts graph-based approaches against older global-descriptor baselines while keeping experimental quantum code clearly scoped.

## ModelA: GCN With Angle Encoding

ModelA represents an intermediate implementation using a Graph Convolutional Network with angle encoding. The source and representative artifacts are retained; the historical 86-record cache is treated as dataset-derived provenance.

## ModelB: SchNet Graph-QNN

ModelB is the primary retained pipeline. It uses graph construction, compressed physics features, and leakage-aware training logic. The tests emphasize target-correlation exclusions and split-aware preprocessing assumptions.

## ModelC: Coulomb Baseline

ModelC uses Coulomb-matrix eigenspectra and amplitude encoding. It is retained as a reference baseline with source and representative outputs.

## Experimental Quantum Components

The VQC code under `ModelB/experimental/` remains experimental. It is included for research continuity, not as the default model path.

## Verification Boundary

Retained evidence includes:

- source review,
- syntax checks,
- data-free leakage tests,
- artifact inspection.

The repository does not claim full retraining from only the files in version control because dataset-derived inputs remain external.
