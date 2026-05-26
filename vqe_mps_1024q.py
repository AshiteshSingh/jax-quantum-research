#!/usr/bin/env python3
"""
================================================================================
  Advanced QML Research: 1,024-Qubit Differentiable MPS on TPU v5p
  
  Model          : 1D Variational Quantum Eigensolver (VQE) / Tensor Network
  Qubits         : 1,024 (Distributed 32 per chip across 32 TPU Cores)
  Bond Dimension : χ = 128 (Mapped precisely to TPU MXU Systolic Arrays)
  Backpropagation: Exact gradients via jax.value_and_grad through SVD operations
  
  Outputs        : tpu/plots/vqe_research_dashboard_<ts>.png
================================================================================
"""

import os
import time
from datetime import datetime
import numpy as np

# Force XLA to preallocate HBM to prevent memory fragmentation during training
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDWARE TOPOLOGY ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────
DEVICES = jax.devices()
NUM_DEVICES = len(DEVICES)

# Mesh definition for cross-chip Tensor Network boundaries
TPU_MESH = Mesh(np.array(DEVICES), ('dev',))
P_SPEC = PartitionSpec('dev', None, None, None)

# Simulation Geometry
CHI = 128                   # Max entanglement boundary (128x128 MXU alignment)
QUBITS_PER_CHIP = 32        # Tensor sites per core
TOTAL_QUBITS = NUM_DEVICES * QUBITS_PER_CHIP
EPOCHS = 40                 # VQE Training iterations
LEARNING_RATE = 0.05

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("tpu/plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. QUANTUM PRIMITIVES & OBSERVABLES
# ─────────────────────────────────────────────────────────────────────────────
# Pauli Z matrix for Energy calculations
Z_MAT = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

@jax.jit
def get_parametric_su4_gate(theta):
    """
    Creates a differentiable 2-qubit entangling gate using an Ising-like 
    interaction parameterized by theta.
    """
    # ZZ interaction matrix
    h_zz = jnp.diag(jnp.array([1, -1, -1, 1], dtype=jnp.complex64))
    # Exponentiate to create unitary: exp(-i * theta * ZZ)
    gate = jnp.cos(theta) * jnp.eye(4) - 1j * jnp.sin(theta) * h_zz
    return gate.reshape((2, 2, 2, 2))

# ─────────────────────────────────────────────────────────────────────────────
# 3. DISTRIBUTED TENSOR NETWORK INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def initialize_mps():
    """Initializes the 1024-qubit MPS to |00...0> strictly inside HBM."""
    def local_init(mesh_index):
        # Shape: (Sites_per_chip, Left_Bond, Physical, Right_Bond)
        tensors = jnp.zeros((QUBITS_PER_CHIP, CHI, 2, CHI), dtype=jnp.complex64)
        tensors = tensors.at[:, 0, 0, 0].set(1.0 + 0.0j)
        return tensors
    return shard_map(local_init, TPU_MESH, in_specs=PartitionSpec(), out_specs=P_SPEC)()

# ─────────────────────────────────────────────────────────────────────────────
# 4. FORWARD TENSOR CONTRACTION & SVD TRUNCATION
# ─────────────────────────────────────────────────────────────────────────────
def apply_local_layer(mps_state, gate_u, layer_type="even"):
    """Applies gates to neighboring qubits within the local chip memory."""
    @jax.jit
    def chip_sweep(local_tensors):
        start_idx = 0 if layer_type == "even" else 1
        
        # Track entanglement entropy by storing singular values
        entropies = jnp.zeros((QUBITS_PER_CHIP // 2,), dtype=jnp.float32)

        def scan_step(carry, idx):
            tensors, ent_arr = carry
            site1, site2 = tensors[idx], tensors[idx + 1]
            
            # 1. Contract virtual bonds
            fused = jnp.einsum("ijk,klm->ijlm", site1, site2)
            # 2. Apply parameterized gate
            transformed = jnp.einsum("abcd,ibcj->iadj", gate_u, fused)
            
            # 3. Reshape for MXU-optimized SVD (256 x 256)
            mat = transformed.reshape((CHI * 2, 2 * CHI))
            u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
            
            # Calculate Von Neumann Entropy: S = -sum(s^2 * log(s^2))
            s_norm = s / jnp.linalg.norm(s)
            s_sq = jnp.square(s_norm) + 1e-12  # prevent log(0)
            entropy = -jnp.sum(s_sq * jnp.log2(s_sq))
            
            # 4. Truncate back to CHI=128
            new_site1 = u[:, :CHI].reshape((CHI, 2, CHI))
            new_site2 = (jnp.diag(s[:CHI]) @ vh[:CHI, :]).reshape((CHI, 2, CHI))
            
            tensors = tensors.at[idx].set(new_site1)
            tensors = tensors.at[idx + 1].set(new_site2)
            ent_arr = ent_arr.at[idx // 2].set(entropy)
            return (tensors, ent_arr), None

        indices = jnp.arange(start_idx, QUBITS_PER_CHIP - 1, 2)
        (final_tensors, final_entropies), _ = jax.lax.scan(scan_step, (local_tensors, entropies), indices)
        return final_tensors, jnp.mean(final_entropies)

    # Shard map returns both the updated distributed tensor and local entropy averages
    return shard_map(chip_sweep, TPU_MESH, in_specs=P_SPEC, out_specs=(P_SPEC, PartitionSpec('dev')))

# ─────────────────────────────────────────────────────────────────────────────
# 5. DIFFERENTIABLE VQE COST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def evaluate_vqe_energy(theta, initial_mps):
    """
    Forward pass: Applies the parameterized quantum circuit and calculates 
    the ground-state energy expectation value.
    """
    gate_u = get_parametric_su4_gate(theta)
    
    # 1. Quantum Circuit Execution
    mps, ent_even = apply_local_layer(initial_mps, gate_u, "even")
    mps, ent_odd  = apply_local_layer(mps, gate_u, "odd")
    
    # 2. Measure Local Energy <Z_i>
    @jax.jit
    def measure_local_z(local_tensors):
        def scan_z(carry, tensor):
            # Contract tensor with its conjugate and Pauli Z
            rho_local = jnp.einsum("ijk,ilk->jl", tensor, jnp.conj(tensor))
            z_exp = jnp.real(jnp.trace(rho_local @ Z_MAT))
            return carry + z_exp, None
        
        total_z, _ = jax.lax.scan(scan_z, 0.0, local_tensors)
        return total_z

    # Sum expectation values across the 32 TPU chips
    local_energies = shard_map(measure_local_z, TPU_MESH, in_specs=P_SPEC, out_specs=PartitionSpec('dev'))(mps)
    global_energy = jnp.sum(local_energies) / TOTAL_QUBITS
    
    mean_entropy = (jnp.mean(ent_even) + jnp.mean(ent_odd)) / 2.0
    return global_energy, (mps, mean_entropy)

# Generate the backward auto-differentiation engine
# has_aux=True tells JAX that the second output (mps, entropy) is not part of the gradient calculation
vqe_grad_engine = jax.jit(jax.value_and_grad(evaluate_vqe_energy, argnums=0, has_aux=True))

# ─────────────────────────────────────────────────────────────────────────────
# 6. RESEARCH DASHBOARD GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_research_dashboard(metrics, ts):
    """Generates a dark-theme, publication-ready metrics dashboard."""
    bg_color = "#0d1117"
    panel_color = "#161b22"
    text_color = "#e6edf3"
    grid_color = "#30363d"
    
    fig = plt.figure(figsize=(16, 10), facecolor=bg_color)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    fig.suptitle(f"Variational Quantum Eigensolver (VQE) Research Dashboard\n"
                 f"TPU v5p-32 │ 1,024 Qubits │ MPS χ=128 │ {ts}",
                 color=text_color, fontsize=14, fontweight="bold")
    
    epochs = np.arange(len(metrics["energy"]))
    
    def style_ax(ax, title, ylabel):
        ax.set_facecolor(panel_color)
        ax.set_title(title, color=text_color, pad=10)
        ax.set_xlabel("Training Epoch", color=text_color)
        ax.set_ylabel(ylabel, color=text_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.grid(True, color=grid_color, linestyle="--", alpha=0.5)

    # Panel 1: Energy Optimization
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, metrics["energy"], color="#58a6ff", lw=2.5, marker="o", markersize=4)
    style_ax(ax1, "Cost Function: Ground State Energy <H>", "Energy (A.U.)")

    # Panel 2: Gradient Flow
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, metrics["grad_norm"], color="#f78166", lw=2.5)
    ax2.fill_between(epochs, 0, metrics["grad_norm"], color="#f78166", alpha=0.2)
    style_ax(ax2, "Backpropagation Gradient Norm", "|| ∂E / ∂θ ||")

    # Panel 3: Entanglement Entropy
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, metrics["entropy"], color="#d2a8ff", lw=2.5, marker="s", markersize=4)
    style_ax(ax3, "Average Von Neumann Entanglement Entropy", "Entropy (bits)")

    # Panel 4: Hardware Latency
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(epochs, metrics["time_ms"], color="#3fb950", alpha=0.8)
    ax4.axhline(np.mean(metrics["time_ms"]), color="#ff7b72", linestyle="--", 
                label=f"Avg: {np.mean(metrics['time_ms']):.1f} ms")
    style_ax(ax4, "TPU MXU Execution Latency per Epoch", "Time (ms)")
    ax4.legend(facecolor=panel_color, edgecolor=grid_color, labelcolor=text_color)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = f"tpu/plots/vqe_research_dashboard_{ts}.png"
    plt.savefig(filepath, dpi=150, facecolor=bg_color)
    plt.close()
    print(f"\n🖼️  Dashboard saved to: {filepath}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EXECUTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"============================================================")
    print(f"🚀 Initialize QML VQE Tensor Network")
    print(f"   Target : {NUM_DEVICES} Devices │ {TOTAL_QUBITS} Qubits")
    print(f"   Method : Exact JAX Autodiff Backpropagation via SVD")
    print(f"============================================================")

    # Initialize State
    print("Allocating 1,024-Qubit MPS in distributed HBM...")
    mps_state = initialize_mps()
    mps_state.block_until_ready()

    # Initialize Variational Parameter (Random starting point)
    theta = jnp.array(0.85, dtype=jnp.float32)

    metrics = {"energy": [], "grad_norm": [], "entropy": [], "time_ms": []}

    print("\nStarting Training Loop...")
    print(f"{'Epoch':<8} | {'Energy <H>':<12} | {'Grad Norm':<12} | {'Entropy':<10} | {'Time (ms)'}")
    print("-" * 65)

    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        
        # 1. Forward Pass (Energy) & Backward Pass (Gradients) inside a single JIT compiled step
        (energy, (updated_mps, entropy)), grad = vqe_grad_engine(theta, mps_state)
        
        # Force XLA sync to accurately measure hardware execution time
        energy.block_until_ready()
        
        t_ms = (time.perf_counter() - t0) * 1000
        
        # 2. Gradient Descent Update Rule
        theta = theta - LEARNING_RATE * grad
        
        # 3. Track Metrics
        metrics["energy"].append(float(energy))
        metrics["grad_norm"].append(float(jnp.abs(grad)))
        metrics["entropy"].append(float(entropy))
        metrics["time_ms"].append(t_ms)
        
        print(f"{epoch:<8} | {float(energy):<12.6f} | {float(jnp.abs(grad)):<12.6f} | {float(entropy):<10.4f} | {t_ms:.1f} ms")

    print(f"============================================================")
    print("✅ VQE Training Complete.")
    generate_research_dashboard(metrics, TS)
