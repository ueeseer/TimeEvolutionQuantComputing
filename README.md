# TimeEvolutionQuantComputing

Quantum spin-chain time-evolution simulation using exact diagonalisation and Trotterisation.

## Project Structure

```
├── GroupProject/
│   ├── __init__.py
│   ├── run_time_evolution_project.py   # CLI entry point
│   └── time_evolution/
│       ├── __init__.py
│       ├── spin_chain.py               # Core: Pauli operators, XXZ Hamiltonian, exact evolution
│       └── project_pipeline.py         # Pipeline: Trotter decomposition, noise model, plotting
├── tests/
│   └── test_spin_chain.py              # Unit tests for spin_chain module
├── pyproject.toml                      # Project metadata and dependencies
├── Project_TimeEvolution.pdf           # Project specification document
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -e ".[dev]"
```

If you want NumPy-vs-Qiskit comparison outputs, also install Qiskit:

```bash
pip install qiskit
```

For the report notebook, open Jupyter from the project root after installation so the same
Python environment can import the package cleanly:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

The notebook at `GroupProject/final_report.ipynb` now auto-detects the project root by looking
for `pyproject.toml`, so it works whether Jupyter was launched from the repository root or from
inside `GroupProject/`.

## Running the Project

Run the full simulation pipeline (exact evolution, Trotter decomposition, noisy trajectories, and error-scaling analysis):

```bash
python -m GroupProject.run_time_evolution_project
```

For a faster smoke run during development:

```bash
python -m GroupProject.run_time_evolution_project --quick
```

Results (figures, `.npz` data files, and JSON metrics) are written to `GroupProject/results/` by default.  Use `--output-dir` to change the output directory.
By default the pipeline also writes Qiskit comparison outputs for each case:

- `<case>_spacetime_qiskit.png`
- `<case>_numpy_vs_qiskit.png`

To disable Qiskit comparison while keeping the rest of the pipeline:

```bash
python -m GroupProject.run_time_evolution_project --no-qiskit-compare
```

## Running Tests

```bash
pytest
```

## License

Apache-2.0 – see [LICENSE](LICENSE) for details.
