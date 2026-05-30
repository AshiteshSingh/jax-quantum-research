# Benchmarks — Full N=10 Rigorous Results

All benchmark data collected on 2026-05-30. Hardware: Intel CPU (JAX cpu:0 backend),
JAX 0.10.1. Raw JSON logs committed in `benchmarks/results/`.

---

## Gradient Timing (15-Qubit HEA, 120 Parameters)

**Source:** `n10_benchmark_20260530_212827.json` (V5 run)

| Method | 10-run mean ± σ | 9-run stable mean ± σ | Speedup (stable) |
|---|---|---|---|
| **jax.grad (reverse-mode AD)** | 44.1 ms ± 20.7 ms | **37.5 ms ± 1.8 ms** | — |
| **PSR emulation (120×2 evals)** | 1,826 ms ± 79.7 ms | 1,826 ms ± 79.7 ms | **48.7×** |

**Raw jax.grad runs (ms):** 106.11*, 35.87, 36.12, 35.67, 35.15, 36.43, 37.26, 40.34, 38.21, 39.77
*Run 1 is a JIT retracing event; excluded from 9-run stable mean.*

**Raw PSR runs (ms):** 1759.98, 1997.50, 1775.52, 1892.40, 1805.04, 1800.24, 1692.28, 1807.51, 1881.80, 1845.88

---

## Scaling Sweep (4–20 Qubits, HEA, 3 Layers except n≥19)

**Source:** `n10_benchmark_20260530_214024.json` (V7 run)

| Qubits | Params | L | XLA compile (s) | Mean exec (ms) | Std (ms) |
|---|---|---|---|---|---|
| 4  | 32  | 3 | 2.64  | 0.237   | 0.128  |
| 6  | 48  | 3 | 2.44  | 0.124   | 0.043  |
| 8  | 64  | 3 | 3.76  | 0.497   | 0.171  |
| 10 | 80  | 3 | 4.20  | 0.359   | 0.086  |
| 12 | 96  | 3 | 6.66  | 1.770   | 0.683  |
| 13 | 104 | 3 | 6.24  | 3.177   | 0.702  |
| 14 | 112 | 3 | 7.31  | 17.43   | 9.56   |
| 15 | 120 | 3 | 6.17  | 19.68   | 1.46   |
| 16 | 128 | 3 | 7.34  | 54.53   | 30.78  |
| 17 | 136 | 3 | 8.63  | 485.4   | 107.2  |
| 18 | 144 | 3 | 10.60 | 858.4   | 169.6  |
| 19 | 114 | 2‡| 8.94  | 751.6   | 120.6  |
| 20 | 120 | 2‡| 9.56  | 1678.0  | 137.6  |

*‡ n≥19 used L=2 to fit within available RAM. P = n×2×(L+1).*

Inflection at n=16–17: state-vector (128 MB–512 MB) exceeds L3 CPU cache → DRAM bandwidth saturation.

---

## 25-Qubit Statevector (1 Layer, CPU Baseline, N=10)

**Source:** `n10_benchmark_20260530_212827.json`

| Framework | Compile (s) | Mean exec ± σ (N=10) | Hardware |
|---|---|---|---|
| **jax_qsim (CPU)** | 20.56 | **20.76 s ± 2.02 s** | JAX cpu:0 |
| jax_qsim (GPU, preliminary) | 18.2 | 15.6 s †N=3 | RTX 2050 |
| PennyLane Lightning CPU †N=3 | 9.1 | 9.9 s | C++ engine |

**Raw 25q CPU runs (s):** 19.23, 20.17, 22.01, 21.01, 25.41, 22.99, 19.02, 19.00, 19.12, 19.66

---

## 27-Qubit (1 Layer, CPU Baseline N=10 + GPU N=3)

**Source:** `n10_benchmark_20260530_212827.json`

| Framework | Mean exec ± σ | Hardware |
|---|---|---|
| **jax_qsim CPU (N=10)** | **99.74 s ± 21.60 s** | JAX cpu:0 |
| jax_qsim GPU (†N=3) | 4.61 s | RTX 2050 CUDA |
| PennyLane Lightning GPU (†N=3) | 6.12 s | RTX 2050 |
| Qiskit-Aer GPU (†N=3) | 6.85 s | RTX 2050 |
| TF Quantum GPU (†N=3) | 7.50 s | RTX 2050 |

GPU is **21.6× faster than CPU JAX** at 27 qubits.
GPU (jax_qsim) is **1.3× faster** than PennyLane Lightning GPU.

**Raw 27q CPU runs (s):** 160.45*, 99.23, 109.34, 83.92, 92.57, 83.70, 92.90, 83.94, 96.37, 94.99
*Run 1 outlier: OS memory paging when 1 GB state-vector first loaded.*

---

## Reproduce These Results

```bash
# Scaling + gradient (CPU, ~4 minutes)
python benchmarks/n10_rigorous_benchmark.py

# GPU benchmarks (requires CUDA GPU)
python benchmarks/cuda_vs_cpu_benchmark.py
python benchmarks/benchmark_27q.py

# Generate figures
python benchmarks/generate_n10_figures.py
```
