# Q2DM v4 Blueprint

## What v4 Is Trying To Solve

The retained ModelB code is useful, but it still carries architectural debt and historical compromises. The v4 redesign direction addresses those issues systematically rather than by patching around them.

## Design Goals

1. Make fold-scoped statistics structurally hard to misuse.
2. Make the graph, feature, and quantum interfaces more explicit and testable.
3. Separate implemented evidence from planned improvements.
4. Make the quantum contribution scientifically comparable to a meaningful classical counterpart.

## Core Proposed Shifts

- A more explicit fold manager and normalization ownership model.
- A more explicit and stable graph-encoder interface.
- A redesigned quantum core rather than an inherited circuit layer.
- Stronger comparison against classical baselines.

## Boundary

This document does not claim that v4 is implemented, that every recommendation is experimentally validated, or that quantum advantage has been demonstrated.
