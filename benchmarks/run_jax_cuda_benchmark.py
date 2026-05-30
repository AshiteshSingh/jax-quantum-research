"""
Real JAX CUDA Benchmark Runner
Measures performance of jax_qsim on NVIDIA CUDA GPU inside WSL2.
"""

import sys
import os
import time
import json
import numpy as np

import jax
import jax.numpy as jnp

# Ensure jax_qsim is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_qsim.circuit import Circuit
from jax_qsim.statevector import zero_state

QUBIT_RANGE = list(range(4, 23))
NUM_REPEATS = 5
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Confirm GPU
try:
    gpu_device = jax.devices("gpu")[0]
    print(f"[SUCCESS] Target GPU: {gpu_device}")
except Exception as e:
    print(f"[FATAL] No GPU device found in JAX! Devices: {jax.devices()}")
    sys.exit(1)

def build_jax_qsim(n):
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for q in range(n - 1):
        c.cnot(q, q + 1)
    for q in range(n):
        c.ry(q, q)
        
    def run_fn(state, p):
        final_state = c.run(p, 'statevector', initial_state=state)
        state_flat = final_state.reshape(-1)
        probs = jnp.abs(state_flat) ** 2
        half = 1 << (n - 1)
        marginal_0 = jnp.sum(probs[:half])
        marginal_1 = jnp.sum(probs[half:])
        return jnp.real(marginal_0 - marginal_1)
        
    return jax.jit(run_fn, device=gpu_device)

def main():
    print("=" * 80)
    print(" EXECUTING NATIVE CUDA GPU SIMULATION BENCHMARK ".center(80, "="))
    print("=" * 80)
    
    gpu_results = {}
    
    for n in QUBIT_RANGE:
        print(f"Running {n:2d} Qubits on CUDA...")
        params = jax.random.uniform(jax.random.PRNGKey(42), shape=(n,))
        
        jax_fn = build_jax_qsim(n)
        state = zero_state(n)
        
        # Warmup (Measure JIT Compilation time)
        t_start = time.time()
        _ = jax_fn(state, params).block_until_ready()
        t_comp = time.time() - t_start
        
        # Execution Runs
        times = []
        for _ in range(NUM_REPEATS):
            t0 = time.time()
            _ = jax_fn(state, params).block_until_ready()
            times.append(time.time() - t0)
        mean_time = np.mean(times)
        
        gpu_results[n] = {
            'compilation': t_comp,
            'execution': mean_time
        }
        
        print(f"  Warmup JIT: {t_comp*1000:7.2f} ms")
        print(f"  Execution : {mean_time*1000:7.2f} ms")
        print("-" * 40)
        
    # Save output
    json_path = os.path.join(results_dir, "gpu_real_data.json")
    with open(json_path, 'w') as f:
        json.dump(gpu_results, f, indent=2)
        
    print("=" * 80)
    print(f"[SUCCESS] Native GPU benchmark completed! Results saved to: {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
