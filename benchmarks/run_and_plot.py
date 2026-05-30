"""
Real-time Quantum Simulator CPU/GPU Comparator & Plotter
Loads actual WSL2 NVIDIA GeForce RTX 2050 JAX CUDA benchmark results,
runs Windows CPU benchmarks, and generates high-contrast dark mode plots.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

try:
    import pennylane as qml
    has_pennylane = True
except ImportError:
    has_pennylane = False

# Ensure jax_qsim is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_qsim.circuit import Circuit
from jax_qsim.statevector import zero_state

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Qubit range matching the GPU run (up to 18 qubits for fast CPU benchmark)
QUBIT_RANGE = list(range(4, 19))
NUM_REPEATS = 3

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
        
    return jax.jit(run_fn)

def build_pennylane(n):
    dev = qml.device("default.qubit", wires=n)
    @qml.qnode(dev, interface="jax")
    def circuit(p):
        for q in range(n):
            qml.Hadamard(wires=q)
        for q in range(n - 1):
            qml.CNOT(wires=[q, q + 1])
        for q in range(n):
            qml.RY(p[q], wires=q)
        return qml.expval(qml.PauliZ(0))
    return jax.jit(circuit)

def main():
    print("=" * 80)
    print(" RUNNING COMPARATIVE WINDOWS CPU BENCHMARKS ".center(80, "="))
    print("=" * 80)
    
    cpu_results = {'jax_qsim_cpu': {}, 'pennylane_cpu': {}}
    
    # Load GPU Results
    gpu_json_path = os.path.join(results_dir, "gpu_real_data.json")
    if not os.path.exists(gpu_json_path):
        print(f"[FATAL] GPU results not found at: {gpu_json_path}")
        sys.exit(1)
        
    with open(gpu_json_path, 'r') as f:
        gpu_data = json.load(f)
    print(f"[SUCCESS] Loaded real-time CUDA GPU data from: {gpu_json_path}")
    
    for n in QUBIT_RANGE:
        print(f"Benchmarking {n:2d} Qubits on CPU...")
        params = jax.random.uniform(jax.random.PRNGKey(42), shape=(n,))
        params_np = np.array(params)
        
        # 1. jax_qsim CPU
        jax_fn = build_jax_qsim(n)
        state = zero_state(n)
        _ = jax_fn(state, params).block_until_ready()  # Warmup
        times = []
        for _ in range(NUM_REPEATS):
            t0 = time.time()
            _ = jax_fn(state, params).block_until_ready()
            times.append(time.time() - t0)
        cpu_results['jax_qsim_cpu'][n] = np.mean(times)
        
        # 2. PennyLane CPU
        if has_pennylane:
            pl_fn = build_pennylane(n)
            _ = pl_fn(params_np).block_until_ready()  # Warmup
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = pl_fn(params_np).block_until_ready()
                times.append(time.time() - t0)
            cpu_results['pennylane_cpu'][n] = np.mean(times)
            
        print(f"  jax_qsim CPU  : {cpu_results['jax_qsim_cpu'][n]*1000:7.2f} ms")
        if has_pennylane:
            print(f"  PennyLane CPU : {cpu_results['pennylane_cpu'][n]*1000:7.2f} ms")
        print("-" * 40)
        
    # ==============================================================================
    # High-Resolution Dark Mode Plotting
    # ==============================================================================
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    
    ns = QUBIT_RANGE
    
    # 1. Plot jax_qsim CPU
    jax_cpu_times = [cpu_results['jax_qsim_cpu'][n] * 1000.0 for n in ns]
    ax.plot(ns, jax_cpu_times, marker='o', ls='-', color='#79c0ff', lw=3, label='jax_qsim (Pure JAX CPU)')
    
    # 2. Plot PennyLane CPU
    if has_pennylane:
        pl_cpu_times = [cpu_results['pennylane_cpu'][n] * 1000.0 for n in ns]
        ax.plot(ns, pl_cpu_times, marker='s', ls='--', color='#ff7b72', lw=2.5, label='PennyLane default.qubit (JAX CPU)')
        
    # 3. Plot JAX CUDA GPU (Real-time measured inside WSL2)
    # Extract keys and convert to ms
    gpu_times = []
    for n in ns:
        gpu_times.append(gpu_data[str(n)]['execution'] * 1000.0)
        
    ax.plot(ns, gpu_times, marker='D', ls='-', color='#56d364', lw=3.5, label='jax_qsim (NVIDIA GeForce RTX 2050 CUDA GPU - Real)')
    
    ax.set_yscale('log')
    ax.set_title("Quantum Simulator CPU vs CUDA GPU Comparison (Log Scale)", 
                 fontsize=14, color='#e6edf3', fontweight='bold', pad=20)
    ax.set_xlabel("Number of Simulated Qubits", fontsize=12, color='#8b949e', labelpad=10)
    ax.set_ylabel("Execution Time (milliseconds)", fontsize=12, color='#8b949e', labelpad=10)
    ax.grid(True, linestyle='--', color='#21262d', alpha=0.6, which='both')
    
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=11, loc='upper left')
    ax.tick_params(colors='#e6edf3', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
        
    # Annotate Speedup Highlight at 18 qubits
    cpu_time_at_18 = cpu_results['jax_qsim_cpu'][18] * 1000.0
    gpu_time_at_18 = gpu_data['18']['execution'] * 1000.0
    speedup = cpu_time_at_18 / gpu_time_at_18
    
    ax.annotate(f"⚡ {speedup:.1f}x GPU Speedup!", 
                xy=(18, gpu_time_at_18), 
                xytext=(15, 100), 
                textcoords="offset points", 
                arrowprops=dict(facecolor='#56d364', shrink=0.08, width=1.5, headwidth=6),
                color='#56d364', fontsize=11, fontweight='bold')
                
    plot_path = os.path.join(results_dir, "real_gpu_vs_cpu_comparison.png")
    plt.savefig(plot_path, dpi=300, facecolor='#0d1117', bbox_inches="tight")
    plt.close()
    
    print("=" * 80)
    print(f"[SUCCESS] Comparative plot successfully generated: {plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
