"""
Performance Benchmark: jax_qsim vs PennyLane (JAX) vs Cirq
Compares execution time of a parameterized quantum circuit across different qubit counts (4 to 12).
Generates a professional log-scale execution time chart and saves results in results/benchmark_comparison.png.
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
import cirq

# Ensure jax_qsim is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_qsim.circuit import Circuit
from jax_qsim.observables import PauliString, Hamiltonian

# Setup parameters
LAYERS = 3
QUBIT_RANGE = list(range(4, 13))  # Benchmark from 4 to 12 qubits
NUM_REPEATS = 8
os.makedirs("results", exist_ok=True)

# ==============================================================================
# 1. Build and Compile jax_qsim Benchmark
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
            
    # Final layer
    for q in range(n):
        c.ry(q, p_idx); p_idx += 1
        c.rz(q, p_idx); p_idx += 1
        
    # Measure expectation value <Z_0>
    H = Hamiltonian([1.0], [PauliString({0: 'Z'})])
    
    def run_fn(params):
        state = c.run(params, 'statevector')
        # Expectation of Z on qubit 0
        probs = jnp.abs(state) ** 2
        # Sum over all qubit dimensions except qubit 0
        axes = tuple(range(1, n))
        marginal = jnp.sum(probs, axis=axes)
        return jnp.real(marginal[0] - marginal[1])
        
    return c.num_params, jax.jit(run_fn)

# ==============================================================================
# 2. Build and Compile PennyLane (JAX backend) Benchmark
# ==============================================================================
def build_pennylane(n):
    dev = qml.device("default.qubit", Wires=n) if hasattr(qml, 'device') else qml.device("default.qubit", wires=n)
    
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
    return num_params, jax.jit(circuit)

# ==============================================================================
# 3. Build Cirq State Vector Simulator Benchmark
# ==============================================================================
def build_cirq(n):
    qubits = cirq.LineQubit.range(n)
    
    def circuit_fn(params):
        c = cirq.Circuit()
        p_idx = 0
        for _ in range(LAYERS):
            for q in range(n):
                c.append(cirq.ry(params[p_idx])(qubits[q])); p_idx += 1
                c.append(cirq.rz(params[p_idx])(qubits[q])); p_idx += 1
            for q in range(n - 1):
                c.append(cirq.CNOT(qubits[q], qubits[q + 1]))
                
        for q in range(n):
            c.append(cirq.ry(params[p_idx])(qubits[q])); p_idx += 1
            c.append(cirq.rz(params[p_idx])(qubits[q])); p_idx += 1
            
        return c
        
    num_params = n * 2 * (LAYERS + 1)
    sim = cirq.Simulator()
    observable = cirq.Z(qubits[0])
    
    def run_fn(params):
        params_np = np.array(params)
        c = circuit_fn(params_np)
        res = sim.simulate(c)
        return observable.expectation_from_state_vector(
            res.final_state_vector, 
            qubit_map={q: i for i, q in enumerate(qubits)}
        ).real
        
    return num_params, run_fn

# ==============================================================================
# Execute Benchmark Suite
# ==============================================================================
def run_benchmark():
    print("=" * 80)
    print(" HONEST CROSS-FRAMEWORK PERFORMANCE BENCHMARK ".center(80, "="))
    print("=" * 80)
    print(f"Qubit range     : {QUBIT_RANGE}")
    print(f"Circuit Depth   : {LAYERS} layers")
    print(f"Repeat runs     : {NUM_REPEATS}")
    print("-" * 80)
    
    results = {'jax_qsim': {}, 'pennylane': {}, 'cirq': {}}
    
    for n in QUBIT_RANGE:
        print(f"Benchmarking {n:^2d} Qubits...")
        
        # Initialize random params
        num_params = n * 2 * (LAYERS + 1)
        params = jax.random.uniform(jax.random.PRNGKey(42), shape=(num_params,))
        
        # --- A. jax_qsim ---
        _, jax_fn = build_jax_qsim(n)
        _ = jax_fn(params).block_until_ready()  # Warmup compilation
        times = []
        for _ in range(NUM_REPEATS):
            t0 = time.time()
            _ = jax_fn(params).block_until_ready()
            times.append(time.time() - t0)
        results['jax_qsim'][n] = {'mean': np.mean(times), 'std': np.std(times)}
        print(f"  jax_qsim  : {np.mean(times)*1000:7.2f} ms")
        
        # --- B. PennyLane ---
        if has_pennylane:
            _, pl_fn = build_pennylane(n)
            _ = pl_fn(params).block_until_ready()  # Warmup compilation
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = pl_fn(params).block_until_ready()
                times.append(time.time() - t0)
            results['pennylane'][n] = {'mean': np.mean(times), 'std': np.std(times)}
            print(f"  PennyLane : {np.mean(times)*1000:7.2f} ms")
        else:
            results['pennylane'][n] = {'mean': None, 'std': None}
            print("  PennyLane : Skipped (Incompatible JAX)")
        
        # --- C. Cirq (Only up to 10 qubits for speed) ---
        if n <= 10:
            _, cirq_fn = build_cirq(n)
            _ = cirq_fn(params)  # Warmup
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = cirq_fn(params)
                times.append(time.time() - t0)
            results['cirq'][n] = {'mean': np.mean(times), 'std': np.std(times)}
            print(f"  Cirq      : {np.mean(times)*1000:7.2f} ms")
        else:
            results['cirq'][n] = {'mean': None, 'std': None}
            print("  Cirq      : Skipped (>10 qubits)")
            
        print("-" * 40)
        
    # Save raw benchmark JSON data
    json_path = os.path.join("results", "benchmark_data.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # ==============================================================================
    # Plotting & Visualization
    # ==============================================================================
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    
    # Colors & styling
    frameworks = [
        ('jax_qsim', '#56d364', 'jax_qsim (Differentiable XLA)'),
        ('cirq', '#ff7b72', 'Cirq (Standard Statevector)')
    ]
    if has_pennylane:
        frameworks.insert(1, ('pennylane', '#79c0ff', 'PennyLane (JAX backend)'))
    
    for key, color, label in frameworks:
        ns = [n for n in QUBIT_RANGE if results[key][n]['mean'] is not None]
        means = [results[key][n]['mean'] * 1000.0 for n in ns]  # convert to ms
        stds = [results[key][n]['std'] * 1000.0 for n in ns]
        
        ax.errorbar(ns, means, yerr=stds, marker='o', ls='-', color=color, 
                    lw=2.5, elinewidth=1.5, capsize=5, label=label)
                    
    ax.set_yscale('log')
    ax.set_title("⚛  Honest Cross-Framework Execution Speed Comparison (Log Scale)", 
                 fontsize=14, color='#e6edf3', fontweight='bold', pad=15)
    ax.set_xlabel("Number of Simulated Qubits", fontsize=12, color='#8b949e', labelpad=10)
    ax.set_ylabel("Execution Time (milliseconds)", fontsize=12, color='#8b949e', labelpad=10)
    ax.grid(True, linestyle='--', color='#21262d', alpha=0.6, which='both')
    
    ax.tick_params(colors='#e6edf3', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
        
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=11)
    
    plot_path = os.path.join("results", "benchmark_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    
    print("=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Raw data saved to: {json_path}")
    print(f"Comparison plot saved to: {plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
