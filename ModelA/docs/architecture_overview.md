# Model A: Architecture Overview

## Representation
Model A models molecules and crystals as **graphs**. Nodes represent atoms, and edges represent bonds or spatial proximity.

## Encoder
It uses a **simple GCN-style message passing** neural network. This allows the model to learn local structural features before passing them to the quantum layer.

## Quantum Encoding
Model A employs **angle encoding** to embed classical graph features into the quantum state. Each classical feature is mapped to the rotation angle of a quantum gate (e.g., $R_y$ or $R_z$).

## Role in the Pipeline
Model A serves as an intermediate evolutionary step. It established the baseline for hybrid Graph-QNN architectures in this project but was superseded by Model B, which introduced richer physics features and a more sophisticated VQC hybrid approach.
