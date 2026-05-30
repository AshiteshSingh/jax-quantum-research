# JAX Quantum Research Suite — Research Paper

**Title:** JAX Quantum Research Suite: A Unified, Hardware-Accelerated, Differentiable Simulator
for NISQ-Era Algorithm Research Across GPU and Cloud TPU Clusters

**Author:** Ashitesh Singh  
**Institution:** Independent Quantum Computing Researcher  
**Code:** https://github.com/AshiteshSingh/Tpu-Accelerated-Quantum-JAX  
**Supported by:** Google TPU Research Cloud (TRC) Program  

---

## Abstract

We present the JAX Quantum Research Suite, a high-performance, differentiable quantum circuit
simulator spanning two co-existing hardware acceleration layers: a GPU division targeting NVIDIA
RTX-class consumer GPUs via CUDA, and a Cloud TPU division distributed across Google Cloud TPU
v5e-16 and v6e-64 VM clusters.

Key empirically measured results (N=10 rigorous runs, 2026-05-30):

- **jax.grad is 48.7× faster than Parameter-Shift Rule** at 120 parameters (37.5 ms vs 1,826 ms,
  15-qubit HEA, CPU baseline, 9 stable post-JIT runs)
- **27-qubit GPU speedup:** 21.6× faster than CPU-only JAX (4.61 s vs 99.74 s on RTX 2050)
- **Cross-framework at 27q:** jax_qsim achieves 1.3× speedup over PennyLane Lightning GPU
- **Scaling:** Exponential O(2^n) confirmed 4–20 qubits; L3 cache inflection at 16–17 qubits
- **Maximum qubit count:** 29q (RTX 2050 VRAM), 33q (TPU v5e-16, sharded), 37q RCS
  (TPU v6e-64, tensor-network)

The 37-qubit random circuit sampling benchmark uses TensorCircuit (JAX backend) for
tensor-network amplitude sampling — not full statevector simulation. F_XEB ≈ 0.001 ± 0.003
(preliminary, N=5 runs) characterizes the tensor-network approximation fidelity.

We also present a differentiable MPS engine with novel numerical stability contributions:
SVD epsilon floors, Wirtinger gradient singularity characterization, and momentum SGD
for V-bounce damping — enabling stable MPS-VQE convergence at 512–1024 qubits.

---

## Honest Limitations

1. At **25 qubits**, the RTX 2050 (192 GB/s) is **slower** than PennyLane Lightning CPU (multi-threaded C++).
   The GPU advantage becomes decisive at 27+ qubits. Higher-bandwidth GPUs (RTX 4090, A100) would
   shift this crossover lower.
2. XLA first-call compilation takes 2–20 seconds. Negligible for training workflows (>10 gradient steps).
3. No gate fusion pass — each gate is a separate XLA memory operation. cuQuantum-style fusion would
   give an additional 2–5× speedup.
4. The N=10 scaling and gradient data are **CPU baseline** measurements (JAX cpu:0). The 25q/27q GPU
   figures are N=3 preliminary runs on RTX 2050.
5. The 37-qubit claim is **tensor-network amplitude sampling**, not statevector simulation.
   Full 37-qubit statevector requires ~1 TB RAM.

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

## Key References

- Bradbury et al. (2018). JAX: composable transformations of Python+NumPy programs.
- Mitarai et al. (2018). Quantum circuit learning. *Physical Review A*, 98(3), 032309.
- Zhang et al. (2023). TensorCircuit: a Quantum Software Framework for the NISQ Era. *Quantum*, 7, 912.
- Bergholm et al. (2022). PennyLane: Automatic differentiation of hybrid quantum-classical computations.
- McClean et al. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*.
