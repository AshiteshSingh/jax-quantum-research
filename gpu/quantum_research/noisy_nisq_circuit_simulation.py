import os
import sys
import time
import json
from functools import partial
from datetime import datetime

# Set up the python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jax_qsim import zero_state, apply_gate, ops
from jax_qsim.noise import apply_channel, depolarizing_channel
from jax_qsim.core import state_vector_flat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- GPU NISQ Circuit Simulator ---

def run_nisq_trajectory(key, noise_p, params, num_qubits, depth):
    """Executes a single stochastic quantum trajectory of a NISQ circuit.
    
    Args:
        key: PRNGKey.
        noise_p: Depolarizing noise probability.
        params: Parameters for the RX/RY rotations.
        num_qubits: Number of qubits (static).
        depth: Number of layers (static).
    """
    # Initialize the state |00...0>
    state = zero_state(num_qubits)
    
    curr_key = key
    param_idx = 0
    
    for d in range(depth):
        # 1. Single Qubit Parameterized Rotations + Depolarizing Noise
        for i in range(num_qubits):
            # Apply Rx
            rx_mat = ops.rx(params[param_idx])
            state = apply_gate(state, rx_mat, [i])
            param_idx += 1
            
            # Apply Ry
            ry_mat = ops.ry(params[param_idx])
            state = apply_gate(state, ry_mat, [i])
            param_idx += 1
            
            # Apply Depolarizing noise to simulate environmental interaction
            curr_key, subkey = jax.random.split(curr_key)
            kraus = depolarizing_channel(noise_p)
            state, _ = apply_channel(state, kraus, [i], subkey)
            
        # 2. Entangling 2-Qubit CNOT Gates + Depolarizing Noise
        for i in range(0, num_qubits - 1, 2):
            state = apply_gate(state, ops.CNOT, [i, i+1])
            
            # Apply noise on both qubits after CNOT gate
            curr_key, subkey1, subkey2 = jax.random.split(curr_key, 3)
            kraus = depolarizing_channel(noise_p)
            state, _ = apply_channel(state, kraus, [i], subkey1)
            state, _ = apply_channel(state, kraus, [i+1], subkey2)
            
    return state

# --- JIT-compiled vectorized functions ---

@partial(jax.jit, static_argnums=(3, 4))
def run_vectorized_trajs(keys, noise_p, params, num_qubits, depth):
    """JIT-compiled parallel trajectories for a single noise rate."""
    return jax.vmap(run_nisq_trajectory, in_axes=(0, None, None, None, None))(keys, noise_p, params, num_qubits, depth)

@partial(jax.jit, static_argnums=(3, 4))
def run_vectorized_nisq(keys, noise_rates, params, num_qubits, depth):
    """JIT-compiled parallel noise levels and parallel trajectories."""
    return jax.vmap(run_vectorized_trajs, in_axes=(None, 0, None, None, None))(keys, noise_rates, params, num_qubits, depth)


def run_nisq_heavy_simulation():
    print("=" * 80)
    print("        GPU-Heavy NISQ Quantum Simulation & Scaling Benchmark")
    print("=" * 80)
    
    # Check GPU execution status
    devices = jax.devices()
    print(f"JAX Default Device: {devices[0]}")
    print(f"Available Devices:  {devices}")
    
    # 1. Simulation parameters (Heavy scale to make RTX 2050 work!)
    num_qubits = 14
    depth = 5
    num_trajectories = 100
    noise_rates = jnp.linspace(0.0, 0.05, 8) # 8 noise steps
    
    # Total parameterized gates = num_qubits * depth * 2 (Rx + Ry)
    num_params = num_qubits * depth * 2
    
    print(f"\nConfiguring Heavy Simulation:")
    print(f"  - Qubits: {num_qubits} (State size: {2**num_qubits} amplitudes)")
    print(f"  - Depth: {depth} layers")
    print(f"  - Trajectories per noise step: {num_trajectories}")
    print(f"  - Total Parallel Trajectories: {len(noise_rates) * num_trajectories}")
    
    # Initialize random parameters and keys
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    params = jax.random.uniform(subkey1, shape=(num_params,), minval=0.0, maxval=2 * jnp.pi)
    
    # Generate batch keys for all trajectories
    batch_keys = jax.random.split(subkey2, num_trajectories)
    
    # --- PHASE 1: JIT Compilation & Real Simulation ---
    print("\nCompiling and running the simulation on the GPU...")
    start_time = time.time()
    
    # Run the heavy vectorized simulation on the GPU!
    noisy_states = run_vectorized_nisq(batch_keys, noise_rates, params, num_qubits, depth)
    noisy_states.block_until_ready() # Block until JAX finishes async GPU execution
    
    execution_time = time.time() - start_time
    print(f"Completed GPU execution in: {execution_time:.3f} seconds (Compilation included)!")
    
    # --- PHASE 2: Compute Fidelity & Theoretical Bounds ---
    print("\nAnalyzing state fidelity statistics...")
    
    # Ideal noiseless final state (noise_p = 0.0, trajectory index = 0)
    ideal_state = noisy_states[0, 0]
    ideal_flat = state_vector_flat(ideal_state)
    
    fidelities = []
    
    # Compute fidelity for all trajectories across noise levels
    for noise_idx, p in enumerate(noise_rates):
        level_fidelities = []
        for t in range(num_trajectories):
            noisy_flat = state_vector_flat(noisy_states[noise_idx, t])
            # Fidelity F = |<ideal|noisy>|^2
            overlap = jnp.vdot(ideal_flat, noisy_flat)
            fidelity = jnp.abs(overlap) ** 2
            level_fidelities.append(float(fidelity))
        fidelities.append(level_fidelities)
        
    fidelities = jnp.array(fidelities) # Shape: (8, 100)
    mean_fidelities = jnp.mean(fidelities, axis=1)
    
    # Count noisy gates
    num_noisy_gates = depth * (num_qubits + (num_qubits // 2) * 2)
    theoretical_fidelity = (1.0 - noise_rates) ** num_noisy_gates
    
    # --- PHASE 3: Qubit Scaling Benchmarking (rtx 2050 stress test) ---
    print("\nRunning Qubit Scaling Benchmark (rtx 2050 stress test)...")
    scaling_qubits = [10, 11, 12, 13, 14, 15]
    benchmark_times = []
    
    for n_q in scaling_qubits:
        t_keys = jax.random.split(subkey2, 50) # 50 trajectories
        t_params = jax.random.uniform(subkey1, shape=(n_q * depth * 2,), minval=0.0, maxval=2 * jnp.pi)
        
        # JIT warm-up
        _ = run_vectorized_trajs(t_keys, 0.01, t_params, n_q, depth)
        _.block_until_ready()
        
        # Benchmark timing
        t0 = time.time()
        res = run_vectorized_trajs(t_keys, 0.01, t_params, n_q, depth)
        res.block_until_ready()
        t_elapsed = time.time() - t0
        benchmark_times.append(t_elapsed * 1000.0) # convert to ms
        print(f"  - Qubits: {n_q:2d} | 50 Parallel Trajectories: {t_elapsed * 1000.0:8.2f} ms")
        
    # --- PHASE 4: Visualizing the Results (Stunning Plots) ---
    print("\nGenerating visual performance plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1e1e2e')
    plt.subplots_adjust(wspace=0.25)
    
    # Graph 1: Fidelity Decay
    ax1.set_facecolor('#24273a')
    
    # Plot individual trajectories as light scatter points to show envelope
    for i, p in enumerate(noise_rates):
        ax1.scatter([p] * num_trajectories, fidelities[i], color='#89b4fa', alpha=0.15, s=20, edgecolors='none', zorder=2)
        
    # Plot mean fidelity and theoretical curve
    ax1.plot(noise_rates, mean_fidelities, label='Mean Trajectory Fidelity', color='#a6e3a1', marker='o', linewidth=3, zorder=5)
    ax1.plot(noise_rates, theoretical_fidelity, label='Theoretical Bound $(1-p)^{N_{gates}}$', color='#f38ba8', linestyle='--', linewidth=2.5, zorder=4)
    
    ax1.set_title(f'Quantum Fidelity Decay vs. Noise Rate ({num_qubits} Qubits)', fontsize=13, fontweight='bold', color='#cdd6f4', pad=12)
    ax1.set_xlabel('Depolarizing Noise Rate ($p$)', fontsize=11, color='#cdd6f4')
    ax1.set_ylabel('State Fidelity ($F$)', fontsize=11, color='#cdd6f4')
    ax1.grid(True, linestyle='--', color='#585b70', alpha=0.4)
    ax1.tick_params(colors='#cdd6f4', labelsize=10)
    ax1.legend(facecolor='#1e1e2e', edgecolor='#cba6f7', labelcolor='#cdd6f4')
    
    # Graph 2: GPU Scaling Benchmark
    ax2.set_facecolor('#24273a')
    ax2.bar(scaling_qubits, benchmark_times, color='#cba6f7', edgecolor='#b4befe', width=0.5, alpha=0.85, zorder=3)
    ax2.plot(scaling_qubits, benchmark_times, color='#f9e2af', marker='D', markersize=6, linewidth=2, zorder=4)
    
    ax2.set_title('GPU Scaling Benchmark (rtx 2050 Execution Time)', fontsize=13, fontweight='bold', color='#cdd6f4', pad=12)
    ax2.set_xlabel('Number of Qubits ($N$)', fontsize=11, color='#cdd6f4')
    ax2.set_ylabel('Execution Time for 50 Trajectories (ms)', fontsize=11, color='#cdd6f4')
    ax2.grid(True, linestyle='--', color='#585b70', alpha=0.4)
    ax2.tick_params(colors='#cdd6f4', labelsize=10)
    
    # Main Figure Title
    fig.suptitle('RTX GPU Acceleration: Heavy NISQ Simulation & Scaling Benchmark', 
                 fontsize=15, fontweight='bold', color='#cdd6f4', y=0.98)
    
    # Save the output plots
    plot_dir = os.path.join(ROOT, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'nisq_heavy_benchmark_{TS}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save JSON results
    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f'nisq_heavy_benchmark_{TS}.json')
    json.dump({
        "experiment": "nisq_benchmark",
        "num_qubits": num_qubits,
        "depth": depth,
        "mean_fidelities": mean_fidelities.tolist(),
        "scaling_qubits": scaling_qubits,
        "scaling_times_ms": benchmark_times
    }, open(json_path, "w"), indent=2)
    
    print("\n" + "=" * 80)
    print(f"Success! Heavy simulation and stress test completed successfully.")
    print(f"Plot saved to '{plot_path}'")
    print(f"Results JSON saved to '{json_path}'")
    print("=" * 80)

if __name__ == '__main__':
    run_nisq_heavy_simulation()
