# Contributing to JAX Quantum Research Suite

## Code Style

- Python 3.10+, type hints encouraged
- All JAX code must be `jax.jit`-compatible (no Python-side side-effects inside jit)
- Use `jnp` for all numerical operations (never `np` inside JIT-traced functions)
- Docstrings: Google style

## Adding Benchmarks

1. Use `N_RUNS = 10`, `N_WARMUP = 2` as standard
2. Report both 10-run mean and 9-run stable mean (run 1 may be JIT retrace for grad functions)
3. Save raw JSON with `encoding='utf-8'` — required for Windows compatibility
4. Use only ASCII characters in `print()` statements (no Unicode box-drawing)
5. Name result files: `n10_benchmark_YYYYMMDD_HHMMSS.json`

## Qubit Count References

- Maximum statevector (GPU RTX 2050): **29 qubits**
- Maximum statevector (TPU v5e-16 sharded): **33 qubits**
- Maximum RCS experiment (TPU v6e-64, tensor-network): **37 qubits**
- **Do NOT use 40 qubits** — this is not physically measured. The correct value is 37.

## Pull Request Guidelines

1. All benchmark claims must reference a committed JSON result file
2. Include hardware specification (CPU model / GPU model) for any timing claim
3. Label preliminary data (N<10) with `†` in tables
4. Do not introduce Unicode characters in benchmark output strings (Windows compatibility)

## Running Tests

```bash
python -c "from jax_qsim.circuit import Circuit; import jax.numpy as jnp; c = Circuit(2); c.h(0); c.cnot(0,1); s = c.run(jnp.array([]), 'statevector'); print('OK:', s.shape)"
```

## Reproducing Paper Benchmarks

```bash
# N=10 scaling + gradient (CPU, ~4 min)
python benchmarks/n10_rigorous_benchmark.py

# GPU scaling (requires CUDA GPU)
python benchmarks/cuda_vs_cpu_benchmark.py

# Generate all paper figures
python benchmarks/generate_n10_figures.py
```
