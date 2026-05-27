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

NUM_QUBITS = 33
STATE_SIZE = 1 << NUM_QUBITS  # 2^33 elements
LOCAL_SIZE = STATE_SIZE // 4

print(f"Allocating 33-qubit state vector ({STATE_SIZE * 8 / 1e9:.2f} GB) via Host Streaming...")

# -------------------------------------------------------------------------
# 2. ZERO-OVERHEAD STREAMING FROM HOST TO CHIPS (OOM SOLUTION)
# -------------------------------------------------------------------------
t_init_start = time.perf_counter()

# Allocate a single 16 GiB chunk on host CPU RAM
print("Allocating 16 GiB buffer on Host CPU RAM...")
host_buffer = np.zeros((LOCAL_SIZE,), dtype=np.complex64)

dev_arrays = []

# Prepare data for Chip 0 (contains the |00...0> ground state component)
print("Streaming Shard 0 to TPU Chip 0...")
host_buffer[0] = 1.0 + 0.0j
dev_arrays.append(jax.device_put(host_buffer, devices[0]))

# Prepare data for Chips 1, 2, and 3 (pure zeros)
host_buffer[0] = 0.0 + 0.0j
for i in range(1, 4):
    print(f"Streaming Shard {i} to TPU Chip {i}...")
    dev_arrays.append(jax.device_put(host_buffer, devices[i]))

# Force clear host memory reference
del host_buffer

# Stitch the 4 isolated device allocations into one unified global sharded array
state = jax.make_array_from_single_device_arrays(
    shape=(STATE_SIZE,),
    sharding=state_sharding,
    arrays=dev_arrays
)
jax.block_until_ready(state)
print(f"State vector successfully loaded into TPU HBM pools. Time taken: {time.perf_counter() - t_init_start:.2f}s")

# -------------------------------------------------------------------------
# 3. IN-PLACE MEMORY-DONATED GATE OPERATIONS
# -------------------------------------------------------------------------
# By using donate_argnums=0, we tell XLA it can safely destroy/overwrite the
# incoming state_vec buffer, ensuring zero-copy mutations during execution.
@functools.partial(jax.jit, static_argnums=2, donate_argnums=0)
def apply_1q_gate(state_vec, gate_matrix, target):
    tensor = state_vec.reshape((2,) * NUM_QUBITS)
    tensor = jnp.moveaxis(tensor, target, 0)
    tensor = jnp.einsum('ij,j...->i...', gate_matrix, tensor)
    tensor = jnp.moveaxis(tensor, 0, target)
    return tensor.reshape((-1,))

@functools.partial(jax.jit, static_argnums=(2, 3), donate_argnums=0)
def apply_cnot(state_vec, control, target):
    tensor = state_vec.reshape((2,) * NUM_QUBITS)
    tensor = jnp.moveaxis(tensor, (control, target), (0, 1))
    
    # Construct CNOT tensor mapping (out_ctrl, out_tgt, in_ctrl, in_tgt)
    cnot_matrix = jnp.zeros((2, 2, 2, 2), dtype=jnp.complex64)
    cnot_matrix = cnot_matrix.at[0, 0, 0, 0].set(1.0)
    cnot_matrix = cnot_matrix.at[0, 1, 0, 1].set(1.0)
    cnot_matrix = cnot_matrix.at[1, 1, 1, 0].set(1.0)
    cnot_matrix = cnot_matrix.at[1, 0, 1, 1].set(1.0)
    
    tensor = jnp.einsum('abcd,cd...->ab...', cnot_matrix, tensor)
    tensor = jnp.moveaxis(tensor, (0, 1), (control, target))
    return tensor.reshape((-1,))

# -------------------------------------------------------------------------
# 4. BENCHMARKING RUN & PERFORMANCE MONITORING
# -------------------------------------------------------------------------
print("\nStarting Benchmark Circuit...")

Hadamard = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex64) / jnp.sqrt(2.0)

qubit_latencies = []
gate_depth_times = []

print("Compiling XLA graph (Warm-up run)...")
tmp = apply_1q_gate(state, Hadamard, 2)
tmp = apply_cnot(tmp, 2, 5)
jax.block_until_ready(tmp)
del tmp
print("Compilation complete. Executing benchmark...")

start_total = time.perf_counter()

# Run a sequence of 1-Qubit gates across all qubits
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
# 5. GRAPH GENERATION METRICS
# -------------------------------------------------------------------------
print("\nGenerating performance diagnostic plots...")
os.makedirs("metrics", exist_ok=True)

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
