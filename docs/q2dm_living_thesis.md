# Q2DM Technical Thesis

**Document type:** technical thesis for the Q2DM project.  
**Purpose:** summarize the intellectual trajectory of the retained model lines.

## 1. Scientific Question

Q2DM asks whether a hybrid graph-based quantum-classical pipeline can learn useful structure-property relationships for 2D materials from crystal structure and compact physics features. The focus has been on targets such as `bandgap` and `exfoliation_energy_per_atom`, where anisotropy, crystal topology, and DFT-derived descriptors all matter.

## 2. Why The Project Kept Changing

The project did not evolve linearly. Earlier attempts mixed real modeling issues with silent implementation flaws. Over time, several key lessons emerged:

- Coulomb-matrix baselines were useful as references, but they compressed away too much structural detail.
- Graph models were necessary to represent periodic crystal structure more faithfully.
- Target leakage and normalization placement mattered enough to invalidate otherwise plausible results.
- Quantum layers were not the only challenge; feature design, compression strategy, and experimental hygiene were equally important.

## 3. Retained Models

### Model A

A GCN-based hybrid model using angle encoding. It is retained as an intermediate graph-QNN line of development rather than the final mainline. Its historical Mo-family cache is dataset-derived and not versioned.

### Model B

The primary retained line. This uses the `Q2DM_v2` modular source set because it preserves correctness-critical behavior such as fold-scoped normalization and stronger target-correlation exclusions.

### Model C

The older unified baseline based on Coulomb-matrix eigenspectra and amplitude encoding. It remains valuable as a compact historical reference, not as the active mainline.

### Model B v4 Planned

A docs-only planned redesign. It is intentionally kept separate from runnable code and should not be mistaken for an implemented system.

## 4. Verified Lessons

- **Do not equate newest packaging with best code.** `Q2DM_Final_Submission` packaged the project cleanly but regressed in correctness-sensitive places relative to `Q2DM_v2`.
- **Documentation must distinguish implemented from planned.** Many source docs mixed future ambitions with current state. The retained docs make those boundaries explicit.
- **Evidence is best kept as selected artifacts.** A few artifact groups communicate the work more effectively than every run folder.
- **Legacy code should earn promotion.** Older material is either promoted into a justified active model line or summarized in canonical docs.

## 5. Current Project Status

The repository keeps:

- one canonical runnable ModelB code path,
- one intermediate ModelA path with local helper files and result artifacts,
- one reference ModelC baseline,
- a curated bibliography,
- common technical docs.

It omits:

- full historical run trees,
- duplicate notebooks and cloud-specific copies,
- deployment-specific helper docs,
- full raw and processed third-party dataset files.

## 6. Limitations

- The repository is not a complete experiment archive.
- Some source docs still contain claims that were useful historically but are not treated as final evidence here.
- `ModelB/v4_planned` is a proposed redesign, not a shipped implementation.
- `ModelC` is retained mainly as a historical baseline and may need further modernization for full reruns.

## 7. What This Repository Demonstrates

- The project story and retained code.
- Academic reference and application portfolio material.
- How the modeling strategy changed over time.
- Representative evidence without repository bloat.
