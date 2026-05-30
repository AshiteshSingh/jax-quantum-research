# Examples

Eight experiment scripts demonstrating `jax_qsim` capabilities.
All scripts run on CPU. GPU/TPU experiments require appropriate hardware.

## Scripts

| Script | Topic | Key Result |
|---|---|---|
| `01_ghz_prep.py` | GHZ state preparation | Equal superposition, |00…0⟩ + |11…1⟩ |
| `02_vqc_xor.py` | Variational Quantum Classifier (XOR) | Binary classification via VQC |
| `03_vqe_molecule.py` | VQE for H₂ ground state | Ground energy ≈ −1.137 Ha (chemical accuracy) |
| `04_qaoa_maxcut.py` | QAOA MaxCut | Approximate combinatorial optimization |
| `05_noisy_monte_carlo.py` | Monte Carlo noise simulation | Depolarizing channel fidelity decay |
| `06_barren_plateau.py` | Barren plateau analysis | Gradient variance ∝ 2^{-n} scaling |
| `07_shor_demo.py` | Shor's order-finding circuit | Structure demonstration (small-scale) |
| `08_grover_search.py` | Grover's algorithm | Amplitude amplification, O(√N) search |

## Run All

```bash
# From repo root
python examples/01_ghz_prep.py
python examples/02_vqc_xor.py
python examples/03_vqe_molecule.py
python examples/04_qaoa_maxcut.py
python examples/05_noisy_monte_carlo.py
```

## Expected Runtimes (CPU, JAX JIT warm-up included)

| Script | First run | Subsequent |
|---|---|---|
| `01_ghz_prep.py` | ~3 s | <1 s |
| `02_vqc_xor.py` | ~5 s | ~2 s |
| `03_vqe_molecule.py` | ~10 s | ~5 s |
| `04_qaoa_maxcut.py` | ~8 s | ~3 s |
| `05_noisy_monte_carlo.py` | ~6 s | ~2 s |
