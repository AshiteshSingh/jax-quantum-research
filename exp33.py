import os
import time
import functools
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

NUM_QUBITS = 32
STATE_SIZE = 1 << NUM_QUBITS  # 2^32 elements

print(f"Allocating 32-qubit state vector ({STATE_SIZE * 8 / 1e9:.2f} GB) across {num_devices} chips...")

# -------------------------------------------------------------------------
# 2. NATIVE SHARDED INITIALIZATION
# -------------------------------------------------------------------------
@jax.jit
def init_ground_state():
    state_vec = jnp.zeros((STATE_SIZE,), dtype=jnp.complex64)
    return state_vec.at[0].set(1.0 + 0.0j)

# Force XLA to compile the allocation directly into the sharded device mesh
init_ground_state_sharded = jax.jit(init_ground_state, out_shardings=state_sharding)
state = init_ground_state_sharded()
jax.block_until_ready(state)

print("State vector successfully sharded across TPU HBM pools.")

# -------------------------------------------------------------------------
# 3. MEMORY-DONATED ZERO-COPY GATE OPERATORS (FAST COMPILATION)
# -------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=2, donate_argnums=0)
def apply_1q_gate(state_vec, gate_matrix, target):
    """Applies a 1-qubit gate using high-performance tensor contraction."""
    left_dim = 1 << target
    right_dim = 1 << (NUM_QUBITS - target - 1)
    
    tensor = state_vec.reshape((left_dim, 2, right_dim))
    tensor = jnp.einsum('ij,ajb->aib', gate_matrix, tensor)
    return tensor.reshape((-1,))

@functools.partial(jax.jit, static_argnums=(1, 2), donate_argnums=0)
def apply_cnot(state_vec, control, target):
    """
    Applies a CNOT gate using a single multi-dimensional tensor contraction.
    Eliminates slicing and stacking to enforce zero-copy buffer reuse.
    """
    # Construct CNOT as a rank-4 tensor layout: (out_control, out_target, in_control, in_target)
    cnot_tensor = jnp.array([
        [[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
        [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]
    ], dtype=jnp.complex64)
    
    if control < target:
        dim1 = 1 << control
        dim3 = 1 << (target - control - 1)
        dim5 = 1 << (NUM_QUBITS - target - 1)
        
        # Reshape into 5 core relational dimensions
        tensor = state_vec.reshape((dim1, 2, dim3, 2, dim5))
        # Contract directly over control (axis 1) and target (axis 3)
        tensor = jnp.einsum('CTct,acbtd->aCbTd', cnot_tensor, tensor)
        return tensor.reshape((-1,))
    else:
        dim1 = 1 << target
        dim3 = 1 << (control - target - 1)
        dim5 = 1 << (NUM_QUBITS - control - 1)
        
        # Reshape where target appears before control in memory layout
        tensor = state_vec.reshape((dim1, 2, dim3, 2, dim5))
        # Contract directly over target (axis 1) and control (axis 3)
        tensor = jnp.einsum('CTct,atbcd->aTbCd', cnot_tensor, tensor)
        return tensor.reshape((-1,))

# -------------------------------------------------------------------------
# 4. BENCHMARKING RUN & PERFORMANCE MONITORING
# -------------------------------------------------------------------------
print("\nStarting Benchmark Circuit...")
Hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)

qubit_latencies = []
gate_depth_times = []

# Overwrite references immediately in the warm-up to free up memory channels
print("Compiling XLA graph (Warm-up run)...")
state = apply_1q_gate(state, Hadamard, 2)
state = apply_cnot(state, 2, 5)
jax.block_until_ready(state)
print("Compilation complete. Executing benchmark...")

start_total = time.perf_counter()

# Run 1-Qubit gate benchmark
for q in range(NUM_QUBITS):
    t0 = time.perf_counter()
    state = apply_1q_gate(state, Hadamard, q)
    jax.block_until_ready(state)
    t1 = time.perf_counter()
    
    latency = (t1 - t0) * 1000  
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
# 5. GRAPH GENERATION METRICS
# -------------------------------------------------------------------------
print("\nGenerating performance diagnostic plots...")
os.makedirs("metrics", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.bar(range(NUM_QUBITS), qubit_latencies, color='royalblue', edgecolor='black', alpha=0.85)
plt.title("TPU v6e-4 Latency Profile by Qubit Index (32 Qubits)", fontsize=14, fontweight='bold')
plt.xlabel("Target Qubit Index", fontsize=12)
plt.ylabel("Execution Latency (ms)", fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("metrics/qubit_latency_profile.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(gate_depth_times) + 1), gate_depth_times, marker='o', color='forestgreen', linewidth=2)
plt.title("Total Simulation Run Time vs. Gate Depth", fontsize=14, fontweight='bold')
plt.xlabel("Gate Execution Step Depth", fontsize=12)
plt.ylabel("Cumulative Elapsed Time (s)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("metrics/runtime_scaling.png", dpi=300)
plt.close()

print("Performance graphs successfully generated and saved to the 'metrics/' folder.")
