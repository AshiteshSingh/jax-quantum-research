 import os
import time
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# 1. TPU INITIALIZATION & SHARDING CONFIGURATION
# -------------------------------------------------------------------------
print("Initializing TPU Environment...")
devices = jax.devices()
num_devices = len(devices)
assert num_devices == 4, f"Expected 4 TPU chips for v6e-4, found {num_devices}."

# Establish a 1D Mesh across the 4 physical TPU v6e chips
mesh = Mesh(mesh_utils.create_device_mesh((4,)), axis_names=('chips',))
state_sharding = NamedSharding(mesh, P('chips'))

NUM_QUBITS = 33
STATE_SIZE = 1 << NUM_QUBITS  # 2^33 elements

print(f"Allocating 33-qubit state vector ({STATE_SIZE * 8 / 1e9:.2f} GB) across {num_devices} chips...")
# Initialize to the ground state |00...0>
# Only the very first element is 1.0
init_state = jnp.zeros((STATE_SIZE,), dtype=jnp.complex64)
init_state = init_state.at[0].set(1.0 + 0.0j)

# Distribute the array across the TPU mesh
state = jax.device_put(init_state, state_sharding)
jax.block_until_ready(state)
print("State vector successfully sharded across TPU HBM pools.")

# -------------------------------------------------------------------------
# 2. HIGH-PERFORMANCE GATE OPERATIONS (JIT-COMPILED GSPMD)
# -------------------------------------------------------------------------
@jax.jit
def apply_1q_gate(state_vec, gate_matrix, target):
    """
    Applies an arbitrary 1-qubit gate using hardware-accelerated tensor contractions.
    XLA GSPMD automatically injects 800 GB/s ICI AllToAll collectives for targets < 2.
    """
    left_dim = 1 << target
    right_dim = 1 << (NUM_QUBITS - target - 1)
    
    # Reshape to isolate target dimension
    tensor = state_vec.reshape((left_dim, 2, right_dim))
    # Contract gate matrix along target axis
    tensor = jnp.einsum('ij,ajb->aib', gate_matrix, tensor)
    return tensor.reshape((-1,))

@jax.jit
def _apply_x_local(sub_state, target_idx, total_bits):
    left = 1 << target_idx
    right = 1 << (total_bits - target_idx - 1)
    tensor = sub_state.reshape((left, 2, right))
    X_gate = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)
    tensor = jnp.einsum('ij,ajb->aib', X_gate, tensor)
    return tensor.reshape((-1,))

@jax.jit
def apply_cnot(state_vec, control, target):
    """
    Applies a CNOT gate via targeted subspace masking to eliminate redundant operations.
    """
    left_c = 1 << control
    right_c = 1 << (NUM_QUBITS - control - 1)
    tensor = state_vec.reshape((left_c, 2, right_c))
    
    # Split into control=0 (Identity) and control=1 (Apply X) paths
    state_c0 = tensor[:, 0, :]
    state_c1 = tensor[:, 1, :]
    
    # Calculate target index offset relative to the extracted control bit
    relative_target = target - 1 if target > control else target
    state_c1_flipped = _apply_x_local(state_c1, relative_target, NUM_QUBITS - 1)
    
    # Reassemble tracking components
    state_c0 = state_c0.reshape((left_c, right_c))
    state_c1_flipped = state_c1_flipped.reshape((left_c, right_c))
    
    combined = jnp.stack([state_c0, state_c1_flipped], axis=1)
    return combined.reshape((-1,))

# -------------------------------------------------------------------------
# 3. BENCHMARKING RUN & PERFORMANCE MONITORING
# -------------------------------------------------------------------------
print("\nStarting Benchmark Circuit...")

# Define Standard Gate Operators
Hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)

qubit_latencies = []
gate_depth_times = []

# Warm-up to trigger XLA compilation (excluded from final metrics)
print("Compiling XLA graph (Warm-up run)...")
tmp = apply_1q_gate(state, Hadamard, 0)
tmp = apply_cnot(tmp, 0, 5)
jax.block_until_ready(tmp)
print("Compilation complete. Executing benchmark...")

start_total = time.perf_counter()

# Run a sequence of 1-Qubit gates across all qubits to measure interconnect variance
for q in range(NUM_QUBITS):
    t0 = time.perf_counter()
    state = apply_1q_gate(state, Hadamard, q)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    
    latency = (t1 - t0) * 1000  # Convert to ms
    qubit_latencies.append(latency)
    gate_depth_times.append(time.perf_counter() - start_total)
    print(f"Gate Depth {q+1:02d}: 1Q-Gate on Qubit {q:02d} | Execution Time: {latency:.2f} ms")

# Run entangling CNOT layers
cnot_pairs = [(i, (i + 7) % NUM_QUBITS) for i in range(10)]
for idx, (ctrl, tgt) in enumerate(cnot_pairs):
    t0 = time.perf_counter()
    state = apply_cnot(state, ctrl, tgt)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    
    latency = (t1 - t0) * 1000
    gate_depth_times.append(time.perf_counter() - start_total)
    print(f"Gate Depth {NUM_QUBITS + idx + 1:02d}: CNOT ctrl={ctrl:02d} tgt={tgt:02d} | Execution Time: {latency:.2f} ms")

end_total = time.perf_counter()
print(f"\nSimulation complete. Total circuit execution time: {end_total - start_total:.4f} seconds.")

# -------------------------------------------------------------------------
# 4. GRAPH GENERATION METRICS
# -------------------------------------------------------------------------
print("\nGenerating performance diagnostic plots...")
os.makedirs("metrics", exist_ok=True)

# Plot 1: Gate Latency Profiles Across Qubits
plt.figure(figsize=(10, 5))
plt.bar(range(NUM_QUBITS), qubit_latencies, color='royalblue', edgecolor='black', alpha=0.85)
plt.axvline(x=1.5, color='crimson', linestyle='--', linewidth=2, label='Distributed Interconnect Boundary')
plt.title("TPU v6e-4 Latency Profile by Qubit Index", fontsize=14, fontweight='bold')
plt.xlabel("Target Qubit Index", fontsize=12)
plt.ylabel("Execution Latency (ms)", fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("metrics/qubit_latency_profile.png", dpi=300)
plt.close()

# Plot 2: Total Execution Scalability Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(gate_depth_times) + 1), gate_depth_times, marker='o', color='forestgreen', linewidth=2)
plt.title("Total Simulation Run Time vs. Gate Depth", fontsize=14, fontweight='bold')
plt.xlabel("Gate Execution Step Depth", fontsize=12)
plt.ylabel("Cumulative Elapsed Time (s)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("metrics/runtime_scaling.png", dpi=300)
plt.close()

print("Performance graphs successfully generated and saved to the 'metrics/' folder:")
print(" - metrics/qubit_latency_profile.png (Highlights local memory vs. ICI communication performance)")
print(" - metrics/runtime_scaling.png (Tracks total run duration step-by-step)")
