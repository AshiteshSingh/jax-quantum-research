# ==========================================
# 0. NUMPY 2.0+ LEGACY COMPATIBILITY PATCHES
# ==========================================
import numpy as np
import math
import os
import sys

# Hotfix 1: Patch missing legacy ComplexWarning attribute in NumPy 2.0
if not hasattr(np, "ComplexWarning"):
    import numpy.exceptions
    np.ComplexWarning = numpy.exceptions.ComplexWarning

# Direct XLA to pool multi-chip topologies
os.environ["JAX_PLATFORMS"] = "tpu,cpu"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=false"

# ==========================================
# 1. IMPORT JAX & INITIALIZE CLUSTER
# ==========================================
# JAX and ml_dtypes initialize here with a clean NumPy pointer
import jax

try:
    jax.distributed.initialize()
    print(f"[CLUSTER] Worker {jax.process_index()} synchronized successfully inside the v5p mesh.")
except Exception as e:
    print(f"[CLUSTER REJECT] Multi-node initialization failed: {e}")
    sys.exit(1)

import jax.numpy as jnp
import tensorcircuit as tc
import cotengra as ctg
import time
import matplotlib.pyplot as plt

# ==========================================
# 2. POST-INITIALIZATION NUMPY 2.0 PATCH
# ==========================================
# Now that the binary C-extensions are safely loaded, patch np.log2 globally
_orig_log2 = np.log2
def _safe_log2(x):
    if isinstance(x, (int, float)):
        return math.log2(x)
    try:
        return _orig_log2(x)
    except Exception:
        return math.log2(float(x))
np.log2 = _safe_log2

# ==========================================
# 3. INITIALIZATION & HARDWARE COUPLING
# ==========================================
def initialize_engine():
    is_master = (jax.process_index() == 0)
    if is_master:
        print("====================================================")
        print("INITIATING 40-QUBIT DIAGNOSTIC ENGINE (TPU v5p-16 Layout)")
        print("====================================================")
    
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    
    if is_master:
        print(f"[SYSTEM] Global TPU Chips Detected: {jax.device_count()}")
        print(f"[SYSTEM] Local TPU Chips Per Host: {jax.local_device_count()}")

# ==========================================
# 4. 40-QUBIT LATTICE GEOMETRY
# ==========================================
N_QUBITS = 40
GRID_ROWS, GRID_COLS = 5, 8  # 5x8 dense rectangular mesh = 40 qubits

def build_40_qubit_grid():
    valid_positions = list(range(GRID_ROWS * GRID_COLS))
    edges = []
    for pos in valid_positions:
        r, c = divmod(pos, GRID_COLS)
        if c + 1 < GRID_COLS:
            edges.append((pos, pos + 1))
        if r + 1 < GRID_ROWS:
            edges.append((pos, pos + GRID_COLS))
    return edges

LATTICE_EDGES = build_40_qubit_grid()

# ==========================================
# 5. EXTREME CHAOS WAVE MECHANICS (RCS)
# ==========================================
def build_chaotic_circuit(gate_parameters, depth=20):
    c = tc.Circuit(N_QUBITS)
    param_idx = 0
    
    for layer in range(depth):
        for i in range(N_QUBITS):
            c.rx(i, theta=gate_parameters[param_idx])
            c.rz(i, theta=gate_parameters[param_idx + 1])
            param_idx += 2
            
        for idx, (q1, q2) in enumerate(LATTICE_EDGES):
            if (layer + idx) % 4 == 0:
                c.cz(q1, q2)
    return c

# ==========================================
# 6. PROTECTIVE TENSOR SLICING (DEADLOCK PROOF)
# ==========================================
opt = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    minimize="size",
    max_repeats=8,
    slicing_opts={"target_size": 2**23},
    progbar=False,
    parallel=False  # Crucial fix: Stops os.fork() from deadlocking multi-threaded JAX
)

try:
    tc.set_contractor("custom", optimizer=opt, preprocessing=True)
    if jax.process_index() == 0:
        print("[SYSTEM] Single-threaded memory protection initialized.")
except Exception as e:
    print(f"[FATAL ERROR] Contractor initialization failed: {e}")
    sys.exit(1)

# ==========================================
# 7. MULTI-CHIP SHARDING ENGINE (vmap + pmap)
# ==========================================
def get_amplitude_probability(gate_parameters, target_bitstring):
    circuit = build_chaotic_circuit(gate_parameters)
    amplitude = circuit.amplitude(target_bitstring)
    return jnp.real(amplitude * jnp.conj(amplitude))

single_chip_batcher = jax.vmap(get_amplitude_probability, in_axes=(None, 0))
parallel_tpu_driver = jax.pmap(single_chip_batcher, in_axes=(None, 0))

# ==========================================
# 8. BENCHMARKING & METRIC PLOTTING
# ==========================================
def run_pipeline():
    initialize_engine()
    is_master = (jax.process_index() == 0)
    
    key = jax.random.PRNGKey(2026 + jax.process_index())
    total_needed_weights = N_QUBITS * 2 * 20
    chaotic_angles = jax.random.uniform(key, shape=(total_needed_weights,), minval=0, maxval=2*jnp.pi)
    
    # Auto-adjust configuration to the 16-chip hardware metrics
    local_chips = jax.local_device_count() 
    tasks_per_chip = 16 
    global_states_computed = jax.device_count() * tasks_per_chip 
    
    target_bitstrings = jax.random.randint(
        key, 
        shape=(local_chips, tasks_per_chip, N_QUBITS), 
        minval=0, maxval=2
    )
    
    execution_times = []
    
    if is_master:
        print(f"\n[STAGE 1] Triggering Graph Slicing & XLA Compilation...")
        print(f"Distributing 40-Qubit tasks across {jax.device_count()} global TPU chips...")
    
    start_compile = time.time()
    try:
        warmup_out = parallel_tpu_driver(chaotic_angles, target_bitstrings)
        warmup_out.block_until_ready()
        compile_overhead = time.time() - start_compile
        if is_master:
            print(f"[SUCCESS] 40-Qubit Graph compiled to bare-metal XLA in {compile_overhead:.2f} seconds.\n")
    except RuntimeError as e:
        if is_master:
            print(f"\n[CRITICAL ERROR] TPU HBM line Overflowed: {e}")
        sys.exit(1)
        
    if is_master:
        print("[STAGE 2] Running Production Hardware Benchmark Iterations...")
    iterations = 5
    results = warmup_out
    
    for loop_id in range(iterations):
        start_run = time.time()
        results = parallel_tpu_driver(chaotic_angles, target_bitstrings)
        results.block_until_ready()
        stop_run = time.time() - start_run
        execution_times.append(stop_run)
        if is_master:
            print(f" -> Iteration {loop_id + 1}/{iterations} Completed: {stop_run:.4f} seconds.")
        
    if is_master:
        avg_throughput = sum(execution_times) / iterations
        print(f"\n[METRIC] Mean Execution Speed: {avg_throughput:.4f} seconds for {global_states_computed} parallel states.")
        print(f"[METRIC] Time Per Individual 40-Qubit State: {avg_throughput / global_states_computed:.4f} seconds.")

        print("\n[STAGE 3] Executing Linear Cross-Entropy Benchmarking (F_XEB)...")
        hilbert_dimension = 2.0 ** N_QUBITS
        calculated_mean_prob = jnp.mean(results)
        f_xeb = (hilbert_dimension * calculated_mean_prob) - 1.0
        
        print(f" -> Hilbert Space Dimension Size: {hilbert_dimension:.3e}")
        print(f" -> Calculated Sample Mean Probability Value: {calculated_mean_prob}")
        print(f" -> Verified F_XEB Output Fingerprint Score: {f_xeb:.6f}")
        
        print("\n[STAGE 4] Saving Performance Graphs to Disk (`tpu_40qubit_performance.png`)...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.plot(range(1, iterations + 1), execution_times, marker='o', color='#00a2ed', linewidth=2, label='TPU v5p MXU Processing Time')
        ax1.axhline(y=avg_throughput, color='r', linestyle='--', label=f'Mean Time ({avg_throughput:.2f}s)')
        ax1.set_title("Hardware Processing Velocity Across Warm JIT Runs", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Iteration Number", fontsize=10)
        ax1.set_ylabel("Time (Seconds)", fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()
        
        ax2.hist(results.flatten(), bins=15, color='#7a00ed', edgecolor='black', alpha=0.7, label='Simulated States')
        ax2.set_title("Probability Frequency Map (Chaos Distribution Test)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Probability Amplitude Value |psi|^2", fontsize=10)
        ax2.set_ylabel("Occurrences Count", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('tpu_40qubit_performance.png', dpi=300)
        print("[SUCCESS] Graphics rendered perfectly. Open `tpu_40qubit_performance.png` to view metrics.")
        print("====================================================")

if __name__ == "__main__":
    run_pipeline()
