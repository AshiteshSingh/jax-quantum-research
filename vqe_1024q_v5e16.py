#!/usr/bin/env python3
"""
Advanced QML: Differentiable 1000-Qubit MPS
Configured for 100-Epoch Stable Convergence
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 1. ENVIRONMENT SETUP
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
jax.distributed.initialize()

# 2. CONFIGURATION
TOTAL_QUBITS = 1000
CHI = 64
EPOCHS = 100
BASE_LR = 0.05
EPS = 1e-7
NUM_GLOBAL_DEVICES = jax.device_count()
QUBITS_PER_CHIP = TOTAL_QUBITS // NUM_GLOBAL_DEVICES

# 3. QUANTUM PRIMITIVES
Z_MAT = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

def get_parametric_su4_gate(theta):
    h_zz = jnp.diag(jnp.array([1, -1, -1, 1], dtype=jnp.complex64))
    gate = jnp.cos(theta) * jnp.eye(4, dtype=jnp.complex64) - 1j * jnp.sin(theta) * h_zz
    return gate.reshape((2, 2, 2, 2))

@jax.pmap
def initialize_local_mps(global_dev_idx):
    key = jax.random.PRNGKey(global_dev_idx)
    noise = jax.random.normal(key, (QUBITS_PER_CHIP, CHI, 2, CHI)) * 1e-2
    tensors = noise.astype(jnp.complex64)
    tensors = tensors.at[:, 0, 0, 0].set(1.0 + 0.0j)
    return tensors / (jnp.linalg.norm(tensors) + EPS)

def apply_local_layer(local_tensors, gate_u, layer_type="even"):
    start_idx = 0 if layer_type == "even" else 1
    limit = QUBITS_PER_CHIP - 1
    for idx in range(start_idx, limit, 2):
        site1, site2 = local_tensors[idx], local_tensors[idx + 1]
        fused = jnp.einsum("ijk,klm->ijlm", site1, site2)
        transformed = jnp.einsum("abcd,ibcj->iadj", gate_u, fused)
        mat = transformed.reshape((CHI * 2, 2 * CHI))
        u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
        s_trunc = s[:CHI] / (jnp.linalg.norm(s[:CHI]) + EPS)
        new_site1 = u[:, :CHI].reshape((CHI, 2, CHI))
        new_site2 = (jnp.diag(s_trunc) @ vh[:CHI, :]).reshape((CHI, 2, CHI))
        local_tensors = local_tensors.at[idx].set(new_site1 / (jnp.linalg.norm(new_site1) + EPS))
        local_tensors = local_tensors.at[idx + 1].set(new_site2 / (jnp.linalg.norm(new_site2) + EPS))
    return local_tensors

def _vqe_grad_engine_impl(theta, local_mps):
    def evaluate_local_energy(t):
        gate_u = get_parametric_su4_gate(t)
        mps = apply_local_layer(local_mps, gate_u, "even")
        mps = apply_local_layer(mps, gate_u, "odd")
        rho = jnp.einsum("ijk,ilk->jl", mps[0], jnp.conj(mps[0]))
        return jnp.real(jnp.trace(rho @ Z_MAT)), mps
    
    (local_z, new_local_mps), local_grad = jax.value_and_grad(evaluate_local_energy, has_aux=True)(theta)
    # Stability: Gradient Clipping
    local_grad = jnp.clip(jnp.real(local_grad), -0.5, 0.5).astype(jnp.complex64)
    return (jax.lax.psum(local_z, 'dev') / NUM_GLOBAL_DEVICES, 
            jax.lax.psum(local_grad, 'dev') / NUM_GLOBAL_DEVICES, 
            new_local_mps)

vqe_grad_engine = jax.pmap(_vqe_grad_engine_impl, axis_name='dev', in_axes=(None, 0))

# 4. TRAINING LOOP
def run_training():
    mps_state = initialize_local_mps(jnp.arange(NUM_GLOBAL_DEVICES))
    theta = jnp.array(0.85, dtype=jnp.complex64)
    energies = []
    
    for epoch in range(EPOCHS):
        lr = BASE_LR * (0.95 ** (epoch // 10))
        global_z, global_grad, mps_state = vqe_grad_engine(theta, mps_state)
        
        grad_val = float(jnp.real(global_grad[0]))
        theta = theta - lr * grad_val
        energy = float(jnp.real(global_z[0]))
        energies.append(energy)
        
        if jax.process_index() == 0 and epoch % 5 == 0:
            print(f"Epoch {epoch:<3} | E: {energy:.6f} | LR: {lr:.5f}")

    if jax.process_index() == 0:
        plt.plot(energies)
        plt.savefig("vqe_1000q_stable.png")
        print("✅ Graph saved.")

if __name__ == "__main__":
    run_training()
