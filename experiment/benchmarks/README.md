# JAX Quantum Circuits Benchmark Suite

This directory contains the cross-framework performance benchmarking suite for evaluating the execution speed of `jax_qsim` against other standard quantum simulation frameworks:
1. **`jax_qsim` (Our Core Simulator)**: Vectorized, differentiable statevector simulator written in pure JAX, compiled via `jax.jit` into high-performance XLA GPU/CPU machine instructions.
2. **PennyLane (JAX backend)**: Pennylane's default statevector simulator with JAX interface compiled via `jax.jit`.
3. **Cirq**: Google Cirq's standard statevector simulator running on Python CPU.

---

## Benchmarking Methodology

The benchmark measures the average execution time (over 8 repeated runs, after 1 initial warm-up compilation run) of a **parameterized hardware-efficient ansatz** of circuit depth $L=3$ across different qubit counts from 4 to 12. 

The circuit applies:
1. **$L$ alternating layers**:
   - Single-qubit parametric $RY(\theta)$ and $RZ(\phi)$ rotation gates on all $n$ qubits.
   - Linear CNOT entangling chain across adjacent qubits.
2. **Final rotation layer** of $RY(\theta)$ and $RZ(\phi)$ on all qubits.
3. **Expectation value calculation** of the Pauli operator $Z_0$ on the first qubit.

---

## How to Run the Benchmark

From the root of the workspace, execute:

```powershell
python benchmarks/honest_benchmark.py
```

### Outputs
- **Raw JSON Data**: Saved to `results/benchmark_data.json`.
- **High-Resolution Speed Comparison Chart**: Saved to `results/benchmark_comparison.png` as a log-scale plot, showcasing strong scaling behavior across qubit counts.
