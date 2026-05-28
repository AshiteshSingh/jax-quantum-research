# High-Performance Differentiable JAX Quantum Simulation Suite

This directory contains `jax_qsim`, a research-level quantum circuit simulation framework built entirely in **pure JAX**, compiled via `jax.jit` into high-performance XLA GPU/CPU instructions.

## Directory Structure

- **`jax_qsim/`**: Simulator core (gates, statevector, density matrix, circuit runner, observables).
- **`examples/`**: Elite physical validations (GHZ state, VQC classification, Molecular VQE chemistry, Graph QAOA, and parallelized Monte Carlo noise studies).
- **`benchmarks/`**: Cross-framework performance benchmarks.
- **`results/`**: Generated high-resolution PNG charts showing convergence and 32x speedups against Google's `Cirq`.
