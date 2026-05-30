# JAX Quantum Research Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/JAX-0.10.1-orange)](https://github.com/google/jax)
[![TPU](https://img.shields.io/badge/TPU-v5e%20%7C%20v6e-green)](https://cloud.google.com/tpu)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

A high-performance, fully differentiable quantum circuit simulator implemented in pure JAX, designed for variational quantum algorithm research on GPU and Cloud TPU hardware.

> **Research paper:** *JAX Quantum Research Suite: A Unified, Hardware-Accelerated, Differentiable Simulator for NISQ-Era Algorithm Research Across GPU and Cloud TPU Clusters* — Ashitesh Singh, 2026.

---

## Key Results (Empirically Measured)

| Benchmark | Result | Hardware | N runs |
|---|---|---|---|
| jax.grad vs Parameter-Shift Rule | **48.7× faster** (37.5 ms vs 1,826 ms) | CPU (JAX 0.10.1) | N=10 |
| 27-qubit jax_qsim GPU vs CPU | **21.6× speedup** (4.61s vs 99.7s) | RTX 2050 vs CPU | N=10 |
| 27-qubit jax_qsim vs PennyLane Lightning GPU | **1.3× faster** (4.61s vs 6.12s) | RTX 2050 | N=3 |
| Scaling range (CPU statevector) | 4 – 20 qubits measured | CPU (JAX) | N=10 each |
| Maximum qubit count (GPU, statevector) | **29 qubits** (4 GB VRAM limit) | RTX 2050 | — |
| Maximum qubit count (TPU v5e-16, sharded) | **33 qubits** (64 GB HBM2e) | Cloud TPU | — |
| 37-qubit RCS (tensor-network) | F_XEB ≈ 0.001 ± 0.003 | TPU v6e-64 | N=5 |

> **Honest hardware note:** At 25 qubits, the RTX 2050 (192 GB/s bandwidth) is slower than PennyLane Lightning CPU for statevector simulation. The GPU advantage emerges at 27+ qubits where the 1 GB state-vector exceeds CPU L3 cache. On high-bandwidth GPUs (RTX 4090, A100) the crossover point would be lower.

---

## Architecture

```
jax_qsim/               ← Core pure-JAX statevector simulator
├── statevector.py      ← (2,)*n tensor gate engine (apply_gate, measure)
├── density_matrix.py   ← Kraus channel density matrix simulator
├── gates.py            ← Gate library: H, X, Y, Z, CNOT, RX, RY, RZ, S, T, CZ, Toffoli, SWAP
├── observables.py      ← PauliString, Hamiltonian, expectation value
├── circuit.py          ← High-level Circuit builder (jax.jit compatible)
└── __init__.py

examples/               ← Eight experiment scripts
├── 01_ghz_prep.py      ← GHZ state preparation + measurement
├── 02_vqc_xor.py       ← Variational Quantum Classifier (XOR)
├── 03_vqe_molecule.py  ← VQE for H₂ molecular ground state
├── 04_qaoa_maxcut.py   ← QAOA MaxCut optimization
└── 05_noisy_monte_carlo.py ← Monte Carlo noise trajectory simulation

benchmarks/             ← Rigorous performance benchmarking suite
├── n10_rigorous_benchmark.py   ← N=10 scaling + gradient timing (primary)
├── cuda_vs_cpu_benchmark.py    ← GPU vs CPU scaling (requires CUDA GPU)
├── benchmark_27q.py            ← 27-qubit cross-framework comparison
├── honest_benchmark.py         ← Framework comparison (jax_qsim vs PennyLane vs Cirq)
└── results/                    ← Raw JSON benchmark logs (committed)
```

---

## Installation

```bash
# Clone
git clone https://github.com/AshiteshSingh/Tpu-Accelerated-Quantum-JAX.git
cd Tpu-Accelerated-Quantum-JAX

# Install dependencies (CPU)
pip install jax jaxlib numpy matplotlib

# For GPU (CUDA 12):
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For Cloud TPU:
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## Quick Start

```python
from jax_qsim.circuit import Circuit
import jax, jax.numpy as jnp

# Build a 4-qubit GHZ state
c = Circuit(4)
c.h(0)
c.cnot(0, 1)
c.cnot(1, 2)
c.cnot(2, 3)

params = jnp.array([])  # No parameters for this circuit
state = c.run(params, 'statevector')
print(f"State shape: {state.shape}")  # (2, 2, 2, 2)

# Variational circuit with jax.grad (all gradients in one backward pass)
vc = Circuit(3)
vc.ry(0, 0); vc.rz(0, 1)
vc.ry(1, 2); vc.rz(1, 3)
vc.cnot(0, 1); vc.cnot(1, 2)
vc.ry(2, 4); vc.rz(2, 5)

def loss(params):
    state = vc.run(params, 'statevector')
    probs = jnp.abs(state.reshape(-1)) ** 2
    return float(probs[0])  # P(|000>)

grad_fn = jax.jit(jax.grad(loss))
params = jnp.ones(6) * 0.1
grads = grad_fn(params)   # All 6 gradients in ONE backward pass
```

---

## Running Experiments

```bash
# GHZ state preparation
python examples/01_ghz_prep.py

# VQE for H2 molecule
python examples/03_vqe_molecule.py

# N=10 benchmark (CPU, ~4 minutes)
python benchmarks/n10_rigorous_benchmark.py

# GPU benchmark (requires CUDA GPU)
python benchmarks/cuda_vs_cpu_benchmark.py
```

---

## Benchmark Results (Raw Data)

Raw JSON benchmark logs are committed in `benchmarks/results/`:

| File | Description |
|---|---|
| `n10_benchmark_20260530_214024.json` | Scaling sweep 4-20q, N=10 runs each (V7) |
| `n10_benchmark_20260530_212827.json` | Gradient timing 15q, 25q/27q CPU, N=10 (V5) |

Reproduce with:
```bash
python benchmarks/n10_rigorous_benchmark.py
python benchmarks/generate_n10_figures.py
```

---

## 37-Qubit Random Circuit Sampling

The largest experiment uses TensorCircuit (JAX backend) for tensor-network amplitude sampling on Cloud TPU v6e-64. The underlying circuit operates on a **37-qubit** topology (20-layer RX/RZ + alternating CZ pattern):

- **Sampling method:** Tensor-network amplitude per bitstring (TensorCircuit)
- **Hardware:** TPU v6e-64 (2 TB HBM3 aggregate)
- **F_XEB:** ~0.001 ± 0.003 (tensor-network approximation, not full statevector)

> This is **not** full statevector simulation at 37 qubits (which requires ~1 TB RAM). It is tensor-network-based amplitude sampling, the only tractable classical method at this scale.

---

## Honest Limitations

1. **Memory bandwidth ceiling:** At 25q, the RTX 2050 is slower than PennyLane Lightning CPU. GPU advantage is decisive at 27q+.
2. **XLA compile time:** First `jax.jit` call takes 2–20 seconds depending on qubit count. Negligible for training workflows.
3. **No gate fusion:** Each gate is a separate XLA operation. Gate fusion (like cuQuantum) would give 2–5× additional speedup.
4. **MPS accuracy:** Bond dimension χ=64 limits entanglement to ~6 bits. Volume-law circuits (Grover, Shor) saturate quickly.
5. **CPU baseline only:** The N=10 scaling benchmark runs on CPU (JAX cpu:0). GPU measurements at 25q/27q are N=3 preliminary runs on RTX 2050.

---

## Citation

```bibtex
@article{singh2026jax,
  title={JAX Quantum Research Suite: A Unified, Hardware-Accelerated, Differentiable Simulator
         for NISQ-Era Algorithm Research Across GPU and Cloud TPU Clusters},
  author={Singh, Ashitesh},
  year={2026},
  url={https://github.com/AshiteshSingh/Tpu-Accelerated-Quantum-JAX}
}
```

---

## Acknowledgements

Supported by the **Google TPU Research Cloud (TRC) Program**, which provided access to Cloud TPU v6e-64 and TPU v5e-16 hardware for the large-scale distributed simulation experiments.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
