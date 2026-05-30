# Benchmark Results

This directory contains raw JSON benchmark logs from N=10 rigorous measurement runs.

## Files

### `n10_benchmark_20260530_214024.json` (V7)
- **Run date:** 2026-05-30 21:40:24
- **Script:** `benchmarks/n10_rigorous_benchmark.py` V7
- **Backend:** JAX 0.10.1, cpu:0
- **Contents:**
  - `bench_C_scaling`: Scaling sweep 4–20 qubits, N=10 timed runs per point
  - `bench_A_proxy_20q`: 20-qubit proxy circuit execution timing

### `n10_benchmark_20260530_212827.json` (V5)
- **Run date:** 2026-05-30 21:28:27
- **Script:** `benchmarks/n10_rigorous_benchmark.py` V5
- **Backend:** JAX 0.10.1, cpu:0
- **Contents:**
  - `bench_D_gradient`: Gradient timing 15q, 120 params (jax.grad + PSR emulation), N=10
  - `bench_A_25q`: 25-qubit statevector execution, N=10 timed runs
  - `bench_B_27q`: 27-qubit statevector execution, N=10 timed runs

## JSON Format

```json
{
  "meta": {
    "timestamp": "YYYYMMDD_HHMMSS",
    "backend": "cpu",
    "n_runs": 10,
    "n_warmup": 2,
    "jax_version": "0.10.1",
    "devices": ["cpu:0"]
  },
  "bench_C_scaling": {
    "4": {
      "n_qubits": 4,
      "n_params": 32,
      "compile_s": 2.639,
      "runs_s": [...],
      "mean_s": 0.000237,
      "std_s": 0.000128
    }
  }
}
```

## Reproduce

```bash
python benchmarks/n10_rigorous_benchmark.py
```

New result files are timestamped and will not overwrite existing ones.
