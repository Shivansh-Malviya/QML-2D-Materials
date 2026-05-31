# Dependency Snapshot

The retained code was developed around Python 3.10 and a scientific Python stack spanning PyTorch, PyTorch Geometric, PennyLane, pymatgen, NumPy, SciPy, pandas, matplotlib, and pytest.

Environment artifacts:

- `environment.yml`
- `requirements.txt`
- `ModelB/environment.yml`
- `ModelB/requirements.txt`

Known environment constraint:

- Mixed Conda/Pip numerical stacks can trigger an OpenMP runtime collision on Windows when multiple OpenMP runtimes load together.

This snapshot records the dependency surface that shaped the experiments. It is not presented as a deployment target.
