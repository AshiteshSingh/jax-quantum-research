"""
Experiment 1: GHZ State Preparation
Prepares a 3-qubit GHZ state (|000> + |111>)/sqrt(2) using a hardware-efficient
variational ansatz in pure JAX, optimized via a JIT-compiled Adam optimizer.
"""

import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys

# Ensure jax_qsim is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_qsim.circuit import Circuit
from jax_qsim import gates
import jax_qsim.statevector as sv

# ==============================================================================
# Setup Experiment Parameters
# ==============================================================================
NUM_QUBITS = 3
LAYERS = 3
NUM_PARAMS = LAYERS * NUM_QUBITS * 3 # 3 rotation gates (RX, RY, RZ) per qubit per layer

# Setup paths
os.makedirs("results", exist_ok=True)

# Define the target state (|000> + |111>)/sqrt(2)
target_sv = jnp.zeros(2**NUM_QUBITS, dtype=jnp.complex64)
target_sv = target_sv.at[0].set(1.0 / jnp.sqrt(2.0))
target_sv = target_sv.at[7].set(1.0 / jnp.sqrt(2.0))

# ==============================================================================
# Define the Differentiable Variational Circuit
# ==============================================================================
def build_ghz_ansatz(params):
    c = Circuit(NUM_QUBITS)
    p_idx = 0
    for _ in range(LAYERS):
        # Rotation gates
        for q in range(NUM_QUBITS):
            c.rx(q, p_idx); p_idx += 1
            c.ry(q, p_idx); p_idx += 1
            c.rz(q, p_idx); p_idx += 1
        # Entangling layer (Linear CNOT chain)
        for q in range(NUM_QUBITS - 1):
            c.cnot(q, q + 1)
    return c.run(params)

def loss_fn(params):
    # Compute output wavefunction
    state = build_ghz_ansatz(params)
    state_flat = state.reshape(-1)
    # Compute fidelity: F = |<target|psi>|^2
    fidelity = jnp.abs(jnp.vdot(target_sv, state_flat)) ** 2
    # Loss: 1 - Fidelity
    return 1.0 - fidelity

# ==============================================================================
# JIT-compiled Optimization Step with Adam
# ==============================================================================
def adam_update(p, g, m, v, t, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
    t = t + 1
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * (g ** 2)
    mh = m / (1.0 - b1 ** t)
    vh = v / (1.0 - b2 ** t)
    return p - lr * mh / (jnp.sqrt(vh) + eps), m, v, t

@jax.jit
def step(params, m, v, t):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, m, v, t = adam_update(params, grads, m, v, t)
    return params, m, v, t, loss

# ==============================================================================
# Execute Optimization Loop
# ==============================================================================
def run_experiment():
    print("=" * 80)
    print(" EXPERIMENT 1: GHZ State Preparation ".center(80, "="))
    print("=" * 80)
    
    # Initialize parameters randomly
    key = jax.random.PRNGKey(42)
    params = jax.random.normal(key, (NUM_PARAMS,)) * 0.1
    m = jnp.zeros(NUM_PARAMS)
    v = jnp.zeros(NUM_PARAMS)
    t = 0
    
    epochs = 150
    loss_history = []
    
    print(f"{'Epoch':^10} | {'Loss (1 - F)':^18} | {'State Fidelity (F)':^22}")
    print("-" * 60)
    
    t0 = time.time()
    for ep in range(1, epochs + 1):
        params, m, v, t, current_loss = step(params, m, v, t)
        loss_history.append(float(current_loss))
        
        if ep == 1 or ep % 15 == 0:
            fidelity = 1.0 - current_loss
            print(f"{ep:^10d} | {current_loss:^18.8f} | {fidelity:^22.8%}")
            
    total_time = time.time() - t0
    final_fidelity = 1.0 - loss_history[-1]
    
    print("-" * 60)
    print(f"Final State Fidelity: {final_fidelity:.6%}")
    print(f"Total Compilation & Execution Time: {total_time:.3f} seconds")
    print("=" * 80)
    
    # ==============================================================================
    # Plotting & Visualization
    # ==============================================================================
    # Establish modern premium design palette
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    
    epochs_range = list(range(1, epochs + 1))
    ax.plot(epochs_range, loss_history, color='#f78166', lw=2.5, label='Loss (1 − F)')
    ax.plot(epochs_range, [1.0 - l for l in loss_history], color='#58a6ff', lw=2.5, label='State Fidelity')
    
    ax.set_title("⚛  Variational GHZ State Preparation — Fidelity Convergence", fontsize=14, color='#e6edf3', fontweight='bold', pad=15)
    ax.set_xlabel("Epoch / Training Step", fontsize=12, color='#8b949e', labelpad=10)
    ax.set_ylabel("Fidelity / Loss Value", fontsize=12, color='#8b949e', labelpad=10)
    ax.grid(True, linestyle='--', color='#21262d', alpha=0.7)
    
    ax.tick_params(colors='#e6edf3', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
        
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=11)
    
    # Save the premium plot
    plot_path = os.path.join("results", "01_state_prep.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    
    print(f"Plot saved successfully to: {plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
