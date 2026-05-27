import os
import time
import functools
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# 1. TPU INITIALIZATION & SHARDING CONFIGURATION
# -------------------------------------------------------------------------
print("Initializing TPU Environment...")
devices = jax.devices()
num_devices = len(devices)
assert num_devices == 4, f"Expected 4 TPU chips for v6e-4, found {num_devices}."

mesh = Mesh(mesh_utils.create_device_mesh((4,)), axis_names=('chips',))
# Shard along the state vector axis, keeping real/imaginary channels unified
state_sharding = NamedSharding(mesh, P(None, 'chips'))

NUM_QUBITS = 33
STATE_SIZE = 1 << NUM_QUBITS  
LOCAL_SIZE = STATE_SIZE // 4

print(f"Allocating 33-qubit state vector ({STATE_SIZE * 8 / 1e9:.2f} GB) via Float32 Host Streaming...")

# -------------------------------------------------------------------------
# 2. ZERO-COPY HOST STREAMING INITIALIZATION
# -------------------------------------------------------------------------
t_init_start = time.perf_counter()
# Allocate dual real/imaginary matrix directly on host CPU RAM
host_buffer = np.zeros((2, LOCAL_SIZE), dtype=np.float32)

# Set the real channel index 0 to 1.0 (Ground State |00...0>)
host_buffer[0, 0] = 1.0
dev_arrays = [jax.device_put(host_buffer, devices[0])]

# Remaining chips receive pure zeros
host_buffer[0, 0] = 0.0
for i in range(1, 4):
    dev_arrays.append(jax.device_put(host_buffer, devices[i]))

del host_buffer

state = jax.make_array_from_single_device_arrays(
    shape=(2, STATE_SIZE),
    sharding=state_sharding,
    arrays=dev_arrays
)
jax.block_until_ready(state)
print(f"State vector successfully loaded into TPU HBM. Time taken: {time.perf_counter() - t_init_start:.2f}s")

# -------------------------------------------------------------------------
# 3. MICRO-BATCHED GATES VIA SHARD_MAP (LOW-MEMORY SPACE CORING)
# -------------------------------------------------------------------------
def apply_1q_gate(state_vec, gate_matrix, target):
    U_real = jnp.real(gate_matrix).astype(jnp.float32)
    U_imag = jnp.imag(gate_matrix).astype(jnp.float32)
    
    # Qubits 31 and 32 partition data across devices; targets <= 30 run locally on-chip
    if target <= 30:
        @functools.partial(shard_map, mesh=mesh, in_specs=P(None, 'chips'), out_specs=P(None, 'chips'))
        def local_1q(local_state):
            # Process the 16 GB chip memory in 4 chunks of 4 GB to keep memory low
            if target < 29:
                tensor = local_state.reshape((2, 4, 1 << 29))
                def scan_fn(carry, idx):
                    br, bi = tensor[0, idx], tensor[1, idx]
                    left, right = 1 << target, 1 << (29 - target - 1)
                    br_v, bi_v = br.reshape((left, 2, right)), bi.reshape((left, 2, right))
                    
                    out_r = jnp.einsum('ij,ajb->aib', U_real, br_v) - jnp.einsum('ij,ajb->aib', U_imag, bi_v)
                    out_i = jnp.einsum('ij,ajb->aib', U_real, bi_v) + jnp.einsum('ij,ajb->aib', U_imag, br_v)
                    return carry, jnp.stack([out_r.reshape((-1,)), out_i.reshape((-1,))], axis=0)
                _, out = jax.lax.scan(scan_fn, None, jnp.arange(4))
                return jnp.transpose(out, (1, 0, 2)).reshape((2, -1))
            else:
                tensor = local_state.reshape((2, 1 << 29, 4))
                def scan_fn(carry, idx):
                    br, bi = tensor[0, :, idx], tensor[1, :, idx]
                    left, right = 1 << target, 1 << (29 - target - 1)
                    br_v, bi_v = br.reshape((left, 2, right)), bi.reshape((left, 2, right))
                    
                    out_r = jnp.einsum('ij,ajb->aib', U_real, br_v) - jnp.einsum('ij,ajb->aib', U_imag, bi_v)
                    out_i = jnp.einsum('ij,ajb->aib', U_real, bi_v) + jnp.einsum('ij,ajb->aib', U_imag, br_v)
                    return carry, jnp.stack([out_r.reshape((-1,)), out_i.reshape((-1,))], axis=0)
                _, out = jax.lax.scan(scan_fn, None, jnp.arange(4))
                return jnp.transpose(out, (1, 2, 0)).reshape((2, -1))
        return local_1q(state_vec)
    else:
        # Cross-chip partition gate operations
        @jax.jit
        def global_1q(s):
            left, right = 1 << target, 1 << (NUM_QUBITS - target - 1)
            tensor = s.reshape((2, left, 2, right))
            br, bi = tensor[0], tensor[1]
            out_r = jnp.einsum('ij,ajb->aib', U_real, br) - jnp.einsum('ij,ajb->aib', U_imag, bi)
            out_i = jnp.einsum('ij,ajb->aib', U_real, bi) + jnp.einsum('ij,ajb->aib', U_imag, br)
            return jnp.stack([out_r, out_i], axis=0).reshape((2, -1))
        return global_1q(state_vec)

@functools.partial(jax.jit, static_argnums=(1, 2))
def apply_cnot(state_vec, control, target):
    # Process CNOT globally using split float32 matrix layouts
    left_dim = 1 << min(control, target)
    mid_dim = 1 << (abs(control - target) - 1)
    right_dim = 1 << (NUM_QUBITS - max(control, target) - 1)
    
    if control < target:
        tensor = state_vec.reshape((2, left_dim, 2, mid_dim, 2, right_dim))
        out_c0_r, out_c0_i = tensor[0, :, 0, :, :, :], tensor[1, :, 0, :, :, :]
        out_c1_r, out_c1_i = tensor[0, :, 1, :, :, :], tensor[1, :, 1, :, :, :]
        # Flip target values when control bit is active
        combined_r = jnp.stack([out_c0_r, out_c1_r[:, :, :, ::-1]], axis=1)
        combined_i = jnp.stack([out_c0_i, out_c1_i[:, :, :, ::-1]], axis=1)
        return jnp.stack([combined_r, combined_i], axis=0).reshape((2, -1))
    else:
        tensor = state_vec.reshape((2, left_dim, 2, mid_dim, 2, right_dim))
        out_c0_r, out_c0_i = tensor[0, :, :, :, 0, :], tensor[1, :, :, :, 0, :]
        out_c1_r, out_c1_i = tensor[0, :, :, :, 1, :], tensor[1, :, :, :, 1, :]
        combined_r = jnp.stack([out_c0_r, out_c1_r[:, ::-1, :, :]], axis=3)
        combined_i = jnp.stack([out_c0_i, out_c1_i[:, ::-1, :, :]], axis=3)
        return jnp.stack([combined_r, combined_i], axis=0).reshape((2, -1))

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
plt.title("TPU v6e-4 Latency Profile by Qubit Index", fontsize=14, fontweight='bold')
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
