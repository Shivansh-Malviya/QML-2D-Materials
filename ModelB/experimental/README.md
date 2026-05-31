# ModelB Experimental VQC Drafts

**Status: experimental. Not used for final result claims.**

These files represent an exploratory direction for the VQC layer that was not validated end-to-end.

## Known Issues

- `train_vqc.py` computes normalization statistics on the full material dataset before the train/validation/test split. This is a data-leakage risk.
- `config_vqc.py` uses backend and hyperparameter defaults that differ from the retained ModelB result line.

## Relationship To Main ModelB

- `ModelB/src/model.py` is the main retained implementation.
- `model_vqc.py` keeps the same broad scaffold but introduces a VQC-specific iteration across the module.
- `ModelB/artifacts/` contains the representative retained training artifacts.

## Purpose

The folder documents an unfinished research direction for later redesign work.
