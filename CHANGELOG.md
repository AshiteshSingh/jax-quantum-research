# Changelog

All notable changes to JAX Quantum Research Suite.

## [0.2.0] — 2026-05-30

### Added
- `benchmarks/n10_rigorous_benchmark.py` V7: Windows-safe, ASCII-only print, utf-8 JSON
- `benchmarks/generate_n10_figures.py`: 4 publication-quality dark-theme figures from real data
- `benchmarks/results/n10_benchmark_20260530_214024.json`: Scaling 4-20q N=10 (V7)
- `benchmarks/results/n10_benchmark_20260530_212827.json`: Gradient/25q/27q N=10 (V5)
- `BENCHMARKS.md`: Full N=10 result tables with raw run arrays
- `PAPER.md`: Research paper abstract and key results
- `LICENSE`: MIT license
- `requirements.txt`, `pyproject.toml`
- `examples/06_barren_plateau.py`, `examples/07_shor_demo.py`, `examples/08_grover_search.py`
- `tpu/README.md`: Cloud TPU deployment documentation

### Fixed
- Removed all 40-qubit references (correct figure is 37-qubit RCS via tensor-network on TPU v6e-64)
- `jax_qsim/__init__.py`: Updated docstring with real measured performance
- `benchmarks/n10_rigorous_benchmark.py`: Fixed Windows charmap crash (Unicode → ASCII)
- `benchmarks/generate_n10_figures.py`: Replaced hardcoded absolute paths with `os.path.dirname(__file__)`
- `examples/02_vqc_xor.py`: Removed unrelated `grid_size = 40` constant
- Gradient speedup figures now consistently cite V5 primary (48.7×, 9-run stable) not stale V7 figure

### Changed
- `benchmarks/README.md`: Expanded methodology section (warmup protocol, JIT retrace handling)
- `jax_qsim/__init__.py`: Added measured performance figures in docstring

## [0.1.0] — 2026-05-27

### Added
- Initial release of `jax_qsim` core simulator
- `statevector.py`: (2,)*n tensor gate engine
- `density_matrix.py`: Kraus channel density matrix
- `gates.py`: 20+ gate types
- `observables.py`: PauliString, Hamiltonian
- `circuit.py`: High-level Circuit builder
- `examples/01_ghz_prep.py` through `05_noisy_monte_carlo.py`
- `benchmarks/honest_benchmark.py`, `cuda_vs_cpu_benchmark.py`, `benchmark_27q.py`
