#!/usr/bin/env python3
"""
================================================================================
  Advanced QML Research: 1,024-Qubit Differentiable MPS on TPU v5e-16
  
  Model          : 1D Variational Quantum Eigensolver (VQE) / Tensor Network
  Qubits         : 1,024 (Distributed 64 per chip across 16 TPU Cores)
  Bond Dimension : χ = 128 (Mapped precisely to TPU MXU Systolic Arrays)
  Outputs        : Dashboard (.png) and Research Report (.txt)
================================================================================
"""

import os
import time
from datetime import datetime
import numpy as np

# Force XLA to preallocate HBM to prevent memory fragmentation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
# CRITICAL FOR MULTI-HOST CLUSTERS: Initialize global worker coordination service
jax.distributed.initialize()
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDWARE TOPOLOGY ORCHESTRATION (v5e-16 Optimized)
# ─────────────────────────────────────────────────────────────────────────────
NUM_DEVICES = jax.device_count() # 16 Cores

# Simulation Geometry optimized for 16 cores
CHI = 128                   # Max entanglement boundary (MXU alignment)
QUBITS_PER_CHIP = 64        # 1024 qubits / 16 cores = 64 sites per core
TOTAL_QUBITS = NUM_DEVICES * QUBITS_PER_CHIP
EPOCHS = 40                 
LEARNING_RATE = 0.05

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("tpu/plots", exist_ok=True)
os.makedirs("tpu/logs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. QUANTUM PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
Z_MAT = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

def get_parametric_su4_gate(theta):
    h_zz = jnp.diag(jnp.array([1, -1, -1, 1], dtype=jnp.complex64))
    gate = jnp.cos(theta) * jnp.eye(4) - 1j * jnp.sin(theta) * h_zz
    return gate.reshape((2, 2, 2, 2))

# ─────────────────────────────────────────────────────────────────────────────
# 3. DISTRIBUTED TENSOR NETWORK (PMAP ISOLATED)
# ─────────────────────────────────────────────────────────────────────────────
# Standard @jax.pmap works fine here because it has no arguments
@jax.pmap
def initialize_local_mps(device_index):
    # Initializes a local 64-qubit chunk natively inside each TPU's HBM
    tensors = jnp.zeros((QUBITS_PER_CHIP, CHI, 2, CHI), dtype=jnp.complex64)
    tensors = tensors.at[:, 0, 0, 0].set(1.0 + 0.0j)
    return tensors

def apply_local_layer(local_tensors, gate_u, layer_type="even"):
    start_idx = 0 if layer_type == "even" else 1
    entropies_list = []
    limit = QUBITS_PER_CHIP - 1
    
    # Python unrolling forces XLA to inline SVD, bypassing lax.scan and Auto-SPMD bugs
    for idx in range(start_idx, limit, 2):
        site1 = local_tensors[idx]
        site2 = local_tensors[idx + 1]
        
        fused = jnp.einsum("ijk,klm->ijlm", site1, site2)
        transformed = jnp.einsum("abcd,ibcj->iadj", gate_u, fused)
        
        mat = transformed.reshape((CHI * 2, 2 * CHI))
        u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
        
        s_norm = s / jnp.linalg.norm(s)
        s_sq = jnp.square(s_norm) + 1e-12 
        entropy = -jnp.sum(s_sq * jnp.log2(s_sq))
        
        new_site1 = u[:, :CHI].reshape((CHI, 2, CHI))
        new_site2 = (jnp.diag(s[:CHI]) @ vh[:CHI, :]).reshape((CHI, 2, CHI))
        
        local_tensors = local_tensors.at[idx].set(new_site1)
        local_tensors = local_tensors.at[idx + 1].set(new_site2)
        entropies_list.append(entropy)

    return local_tensors, jnp.mean(jnp.stack(entropies_list))

# ─────────────────────────────────────────────────────────────────────────────
# 4. DIFFERENTIABLE AUTODIFF ENGINE (PMAP REDUCTION)
# ─────────────────────────────────────────────────────────────────────────────
# Define the base function normally without the decorator
def _vqe_grad_engine_impl(theta, local_mps):
    
    def evaluate_local_energy(t, mps):
        gate_u = get_parametric_su4_gate(t)
        mps, ent_even = apply_local_layer(mps, gate_u, "even")
        mps, ent_odd  = apply_local_layer(mps, gate_u, "odd")
        
        total_z = 0.0
        for idx in range(QUBITS_PER_CHIP):
            tensor = mps[idx]
            rho_local = jnp.einsum("ijk,ilk->jl", tensor, jnp.conj(tensor))
            z_exp = jnp.real(jnp.trace(rho_local @ Z_MAT))
            total_z = total_z + z_exp
            
        mean_entropy = (ent_even + ent_odd) / 2.0
        return total_z, (mps, mean_entropy)

    # Differentiate the local chunk of the circuit
    grad_fn = jax.value_and_grad(evaluate_local_energy, argnums=0, has_aux=True)
    (local_z, (new_local_mps, local_ent)), local_grad = grad_fn(theta, local_mps)
    
    # Hardware-native Cross-TPU AllReduce reductions (bypasses Auto-SPMD partitioner)
    global_z = jax.lax.psum(local_z, axis_name='dev') / TOTAL_QUBITS
    global_grad = jax.lax.psum(local_grad, axis_name='dev') / TOTAL_QUBITS
    global_ent = jax.lax.pmean(local_ent, axis_name='dev')
    
    return global_z, global_grad, new_local_mps, global_ent

# Map over the 16 devices (axis 0 of local_mps), while broadcasting theta (None)
# We pass the function explicitly as the first argument to avoid Python decorator errors
vqe_grad_engine = jax.pmap(_vqe_grad_engine_impl, axis_name='dev', in_axes=(None, 0))

# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA EXPORT (PLOTS & TXT)
# ─────────────────────────────────────────────────────────────────────────────
def export_research_artifacts(metrics, ts):
    txt_filepath = f"tpu/logs/vqe_report_{ts}.txt"
    with open(txt_filepath, "w") as f:
        f.write("============================================================\n")
        f.write(f" VQE 1,024-Qubit MPS Training Log (TPU v5e-16)\n")
        f.write(f" Timestamp: {ts}\n")
        f.write("============================================================\n")
        f.write(f"{'Epoch':<8} | {'Energy':<12} | {'Grad Norm':<12} | {'Entropy':<10} | {'Time(ms)'}\n")
        f.write("-" * 65 + "\n")
        for i in range(len(metrics["energy"])):
            f.write(f"{i:<8} | {metrics['energy'][i]:<12.6f} | {metrics['grad_norm'][i]:<12.6f} | {metrics['entropy'][i]:<10.4f} | {metrics['time_ms'][i]:.1f}\n")
        f.write("-" * 65 + "\n")
        f.write(f"Final Energy Convergence : {metrics['energy'][-1]:.6f}\n")
        f.write(f"Average Execution Latency: {np.mean(metrics['time_ms']):.2f} ms\n")
    
    print(f"\n📄 TXT Data Log saved to: {txt_filepath}")

    bg_color, panel_color, text_color, grid_color = "#0d1117", "#161b22", "#e6edf3", "#30363d"
    fig = plt.figure(figsize=(16, 10), facecolor=bg_color)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    fig.suptitle(f"Variational Quantum Eigensolver (VQE) Research Dashboard\n"
                 f"TPU v5e-16 │ 1,024 Qubits │ MPS χ=128 │ {ts}",
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

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, metrics["energy"], color="#58a6ff", lw=2.5, marker="o", markersize=4)
    style_ax(ax1, "Cost Function: Ground State Energy <H>", "Energy (A.U.)")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, metrics["grad_norm"], color="#f78166", lw=2.5)
    ax2.fill_between(epochs, 0, metrics["grad_norm"], color="#f78166", alpha=0.2)
    style_ax(ax2, "Backpropagation Gradient Norm", "|| ∂E / ∂θ ||")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, metrics["entropy"], color="#d2a8ff", lw=2.5, marker="s", markersize=4)
    style_ax(ax3, "Average Von Neumann Entanglement Entropy", "Entropy (bits)")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(epochs, metrics["time_ms"], color="#3fb950", alpha=0.8)
    style_ax(ax4, "TPU MXU Execution Latency per Epoch", "Time (ms)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    png_filepath = f"tpu/plots/vqe_dashboard_{ts}.png"
    plt.savefig(png_filepath, dpi=150, facecolor=bg_color)
    plt.close()
    print(f"🖼️  Dashboard saved to  : {png_filepath}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN EXECUTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"============================================================")
    print(f"🚀 VQE 1,024-Qubit MPS Initialize (pmap Architecture)")
    print(f"   Target : {NUM_DEVICES} Cores (v5e-16) │ {TOTAL_QUBITS} Qubits")
    print(f"============================================================")

    # Initialize 16 isolated states across the chips
    mps_state = initialize_local_mps(jnp.arange(NUM_DEVICES))
    mps_state.block_until_ready()
    
    # Initialize theta as a complex scalar to bypass the JAX ComplexWarning
    theta = jnp.array(0.85, dtype=jnp.complex64)
    metrics = {"energy": [], "grad_norm": [], "entropy": [], "time_ms": []}

    print("\nStarting Training Loop...")
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        
        # pmap executes in parallel and returns arrays of identical sums across the 16 cores
        global_z, global_grad, updated_mps, global_ent = vqe_grad_engine(theta, mps_state)
        global_z.block_until_ready()
        
        t_ms = (time.perf_counter() - t0) * 1000
        
        # All 16 devices hold the exact same global values, so we just slice index [0]
        energy = float(jnp.real(global_z[0]))
        grad_val = float(jnp.real(global_grad[0]))
        entropy = float(jnp.real(global_ent[0]))
        
        theta = theta - LEARNING_RATE * grad_val
        mps_state = updated_mps 
        
        metrics["energy"].append(energy)
        metrics["grad_norm"].append(abs(grad_val))
        metrics["entropy"].append(entropy)
        metrics["time_ms"].append(t_ms)
        
        print(f"Epoch {epoch:<3} | E: {energy:<9.5f} | Grad: {abs(grad_val):<8.5f} | Time: {t_ms:.1f} ms")

    export_research_artifacts(metrics, TS)
