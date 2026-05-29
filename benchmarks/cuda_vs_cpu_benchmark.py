"""
CUDA vs CPU Benchmark: jax_qsim (Pure JAX on GPU) vs PennyLane (JAX backend on CPU)
Compares execution time across different qubit counts (4 to 16).
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
from jax_qsim.observables import PauliString, Hamiltonian

# Setup parameters
LAYERS = 3
QUBIT_RANGE = list(range(4, 15))  # Benchmark from 4 to 14 qubits
NUM_REPEATS = 5
os.makedirs("results", exist_ok=True)

# Detect devices
try:
    gpu_device = jax.devices("gpu")[0]
    print(f"[INFO] GPU detected: {gpu_device}")
except Exception:
    gpu_device = None
    print("[WARNING] No JAX GPU device detected. Falling back to CPU (run under WSL2/Linux for GPU support).")

cpu_device = jax.devices("cpu")[0]

# ==============================================================================
# 1. Build and Compile jax_qsim on GPU/CUDA
# ==============================================================================
def build_jax_qsim(n):
    c = Circuit(n)
    p_idx = 0
    for _ in range(LAYERS):
        for q in range(n):
            c.ry(q, p_idx); p_idx += 1
            c.rz(q, p_idx); p_idx += 1
        for q in range(n - 1):
            c.cnot(q, q + 1)
            
    for q in range(n):
        c.ry(q, p_idx); p_idx += 1
        c.rz(q, p_idx); p_idx += 1
        
    H = Hamiltonian([1.0], [PauliString({0: 'Z'})])
    
    def run_fn(params):
        state = c.run(params, 'statevector')
        probs = jnp.abs(state) ** 2
        axes = tuple(range(1, n))
        marginal = jnp.sum(probs, axis=axes)
        return jnp.real(marginal[0] - marginal[1])
        
    # Force GPU execution if available
    target_device = gpu_device if gpu_device else cpu_device
    return c.num_params, jax.jit(run_fn, device=target_device)

# ==============================================================================
# 2. Build and Compile PennyLane on CPU
# ==============================================================================
def build_pennylane(n):
    # PennyLane device mapped to JAX CPU
    dev = qml.device("default.qubit", wires=n)
    
    @qml.qnode(dev, interface="jax")
    def circuit(params):
        p_idx = 0
        for _ in range(LAYERS):
            for q in range(n):
                qml.RY(params[p_idx], wires=q); p_idx += 1
                qml.RZ(params[p_idx], wires=q); p_idx += 1
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
                
        for q in range(n):
            qml.RY(params[p_idx], wires=q); p_idx += 1
            qml.RZ(params[p_idx], wires=q); p_idx += 1
            
        return qml.expval(qml.PauliZ(0))
        
    num_params = n * 2 * (LAYERS + 1)
    # Force CPU execution
    return num_params, jax.jit(circuit, device=cpu_device)

# ==============================================================================
# Execute Benchmark Suite
# ==============================================================================
def run_benchmark():
    print("=" * 80)
    print(" CUDA VS CPU CROSS-FRAMEWORK PERFORMANCE BENCHMARK ".center(80, "="))
    print("=" * 80)
    print(f"JAX (jax_qsim) Device  : {gpu_device if gpu_device else 'CPU (Fallback)'}")
    print(f"PennyLane Device        : CPU")
    print(f"Qubit range             : {QUBIT_RANGE}")
    print(f"Circuit Depth           : {LAYERS} layers")
    print(f"Repeat runs             : {NUM_REPEATS}")
    print("-" * 80)
    
    results = {'jax_qsim_gpu': {}, 'pennylane_cpu': {}}
    
    for n in QUBIT_RANGE:
        print(f"Benchmarking {n:^2d} Qubits...")
        
        num_params = n * 2 * (LAYERS + 1)
        params = jax.random.uniform(jax.random.PRNGKey(42), shape=(num_params,))
        
        # Move params to target device
        params_gpu = jax.device_put(params, gpu_device if gpu_device else cpu_device)
        params_cpu = jax.device_put(params, cpu_device)
        
        # --- A. jax_qsim GPU/CUDA ---
        _, jax_fn = build_jax_qsim(n)
        _ = jax_fn(params_gpu).block_until_ready()  # Warmup compilation
        times = []
        for _ in range(NUM_REPEATS):
            t0 = time.time()
            _ = jax_fn(params_gpu).block_until_ready()
            times.append(time.time() - t0)
        results['jax_qsim_gpu'][n] = {'mean': np.mean(times), 'std': np.std(times)}
        print(f"  jax_qsim (GPU/Target): {np.mean(times)*1000:7.2f} ms")
        
        # --- B. PennyLane CPU ---
        if has_pennylane:
            _, pl_fn = build_pennylane(n)
            _ = pl_fn(params_cpu).block_until_ready()  # Warmup compilation
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = pl_fn(params_cpu).block_until_ready()
                times.append(time.time() - t0)
            results['pennylane_cpu'][n] = {'mean': np.mean(times), 'std': np.std(times)}
            print(f"  PennyLane (CPU)      : {np.mean(times)*1000:7.2f} ms")
        else:
            results['pennylane_cpu'][n] = {'mean': None, 'std': None}
            print("  PennyLane (CPU)      : Skipped (Not Installed)")
        
        print("-" * 40)
        
    # Save raw benchmark JSON data
    json_path = os.path.join("results", "cuda_cpu_benchmark_data.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # ==============================================================================
    # Plotting & Visualization
    # ==============================================================================
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    
    # Colors & styling
    gpu_label = 'jax_qsim (Pure JAX on GPU/CUDA)' if gpu_device else 'jax_qsim (Pure JAX on CPU)'
    frameworks = [
        ('jax_qsim_gpu', '#56d364', gpu_label),
        ('pennylane_cpu', '#79c0ff', 'PennyLane (JAX backend on CPU)')
    ]
    
    for key, color, label in frameworks:
        ns = [n for n in QUBIT_RANGE if results[key][n]['mean'] is not None]
        means = [results[key][n]['mean'] * 1000.0 for n in ns]  # convert to ms
        stds = [results[key][n]['std'] * 1000.0 for n in ns]
        
        ax.errorbar(ns, means, yerr=stds, marker='o', ls='-', color=color, 
                    lw=2.5, elinewidth=1.5, capsize=5, label=label)
                    
    ax.set_yscale('log')
    ax.set_title("CUDA vs CPU Cross-Framework Speed Comparison (Log Scale)", 
                 fontsize=14, color='#e6edf3', fontweight='bold', pad=15)
    ax.set_xlabel("Number of Simulated Qubits", fontsize=12, color='#8b949e', labelpad=10)
    ax.set_ylabel("Execution Time (milliseconds)", fontsize=12, color='#8b949e', labelpad=10)
    ax.grid(True, linestyle='--', color='#21262d', alpha=0.6, which='both')
    
    ax.tick_params(colors='#e6edf3', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
        
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=11)
    
    plot_path = os.path.join("results", "cuda_cpu_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    
    print("=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Raw data saved to: {json_path}")
    print(f"Comparison plot saved to: {plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
