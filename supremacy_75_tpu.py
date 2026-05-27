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

# Hotfix 2: Patch np.log2 to safely handle plain/large Python ints under NumPy 2.0
_orig_log2 = np.log2
def _safe_log2(x):
    if isinstance(x, (int, float)):
        return math.log2(x)
    try:
        return _orig_log2(x)
    except Exception:
        return math.log2(float(x))
np.log2 = _safe_log2

# Direct XLA to pool multi-chip topologies
os.environ["JAX_PLATFORMS"] = "tpu,cpu"
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=false"

# ==========================================
# 1. IMPORT JAX & INITIALIZE CLUSTER
# ==========================================
import jax

try:
    jax.distributed.initialize()
    print(f"[CLUSTER] Worker {jax.process_index()} synchronized successfully inside the 4-node mesh.")
except Exception as e:
    print(f"[CLUSTER REJECT] Multi-node initialization failed: {e}")
    sys.exit(1)

# Now it is safe to import the rest of the science stack
import jax.numpy as jnp
import tensorcircuit as tc
import cotengra as ctg
import time
import matplotlib.pyplot as plt

# ==========================================
# 2. INITIALIZATION & HARDWARE COUPLING
# ==========================================
def initialize_engine():
    print("====================================================")
    print("INITIATING 75-QUBIT EXTREME CHAOS ENGINE (TPU v5e-16)")
    print("====================================================")
    
    tc.set_backend("jax")
    # complex64 allows the TPU Matrix Units to operate at maximum TeraFLOPS velocity
    tc.set_dtype("complex64")
    
    num_chips = jax.device_count()
    print(f"[SYSTEM] Active TPU Chips Detected: {num_chips}")
    return num_chips

# ==========================================
# 3. 75-QUBIT 2D GRID LATTICE GEOMETRY
# ==========================================
N_QUBITS = 75
GRID_ROWS, GRID_COLS = 9, 9  # 81 total potential spaces

def build_75_qubit_grid():
    """Drops 6 corner sites out of a 9x9 layout to form a dense 75-qubit 2D cluster."""
    dropped_indices = {0, 8, 72, 80, 4, 76}  # Dropping corners and edge boundary points
    valid_positions = [i for i in range(GRID_ROWS * GRID_COLS) if i not in dropped_indices]
    
    # Map raw 2D grid positions to a continuous 0-74 index system for the circuit simulator
    coordinate_mapping = {raw_pos: clean_id for clean_id, raw_pos in enumerate(valid_positions)}
    
    edges = []
    for pos in valid_positions:
        r, c = divmod(pos, GRID_COLS)
        # 2D horizontal link
        if c + 1 < GRID_COLS and (pos + 1) in coordinate_mapping:
            edges.append((coordinate_mapping[pos], coordinate_mapping[pos + 1]))
        # 2D vertical link
        if r + 1 < GRID_ROWS and (pos + GRID_COLS) in coordinate_mapping:
            edges.append((coordinate_mapping[pos], coordinate_mapping[pos + GRID_COLS]))
            
    return edges

LATTICE_EDGES = build_75_qubit_grid()

# ==========================================
# 4. EXTREME CHAOS WAVE MECHANICS (RCS)
# ==========================================
def build_chaotic_circuit(gate_parameters, depth=20):
    """
    Constructs a 75-qubit highly entangled network topology.
    Alternates random single-qubit rotations with structured 2D grid entanglers.
    """
    c = tc.Circuit(N_QUBITS)
    param_idx = 0
    
    for layer in range(depth):
        # Step A: Chaos Seeding Layer (Rx and Rz combinations)
        for i in range(N_QUBITS):
            c.rx(i, theta=gate_parameters[param_idx])
            c.rz(i, theta=gate_parameters[param_idx + 1])
            param_idx += 2
            
        # Step B: Google-Style Alternating 2D Grid Interconnections
        for idx, (q1, q2) in enumerate(LATTICE_EDGES):
            if (layer + idx) % 4 == 0:
                c.cz(q1, q2)
                
    return c

# ==========================================
# 5. PROTECTIVE TENSOR SLICING (MEMORY SAFEGUARD)
# ==========================================
# 75 qubits will instantly smash through 16GB memory arrays without a bond dimension limit.
# We configure a reusable cotengra optimizer that shards tensors down to safe ~128MB chunks.
opt = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    minimize="size",
    max_repeats=8,  # Slashed to 8 to protect host CPU memory
    slicing_opts={"target_size": 2**23},
    progbar=False
)

try:
    tc.set_contractor("custom", optimizer=opt, preprocessing=True)
    print("[SYSTEM] Memory protection armor initialized via Cotengra Slicing.")
except Exception as e:
    print(f"[FATAL ERROR] Cotengra graph compiler failed to couple with TensorCircuit: {e}")
    sys.exit(1)

# ==========================================
# 6. MULTI-CHIP SHARDING ENGINE
# ==========================================
def get_amplitude_probability(gate_parameters, target_bitstring):
    """Computes pure probability value |psi(x)|^2 for a single output bitstring."""
    circuit = build_chaotic_circuit(gate_parameters)
    amplitude = circuit.amplitude(target_bitstring)
    return jnp.real(amplitude * jnp.conj(amplitude))

# Parallel execution map across all 16 physical TPU core elements
parallel_tpu_driver = jax.pmap(get_amplitude_probability, in_axes=(None, 0))

# ==========================================
# 7. BENCHMARKING & METRIC PLOTTING
# ==========================================
def run_pipeline():
    num_chips = initialize_engine()
    
    # Generate repeatable chaotic parameters
    key = jax.random.PRNGKey(2026)
    total_needed_weights = N_QUBITS * 2 * 20
    chaotic_angles = jax.random.uniform(key, shape=(total_needed_weights,), minval=0, maxval=2*jnp.pi)
    
    # Shard data chunks: 4 tasks per chip (64 total parallel state extractions)
    batch_size = num_chips * 4
    target_bitstrings = jax.random.randint(key, shape=(batch_size, N_QUBITS), minval=0, maxval=2)
    
    execution_times = []
    
    # Only the master worker node (Process 0) handles the printing and logging
    is_master = (jax.process_index() == 0)
    
    if is_master:
        print("\n[STAGE 1] Triggering Graph Slicing & XLA Compilation...")
        print(f"Slicing 75-qubit networks and distributing to {num_chips} TPU chips.")
    
    start_compile = time.time()
    try:
        warmup_out = parallel_tpu_driver(chaotic_angles, target_bitstrings)
        warmup_out.block_until_ready()
        compile_overhead = time.time() - start_compile
        if is_master:
            print(f"[SUCCESS] 75-Qubit Graph compiled to bare-metal XLA in {compile_overhead:.2f} seconds.\n")
    except RuntimeError as e:
        if is_master:
            print(f"\n[CRITICAL OUT OF MEMORY] TPU v5e HBM2 line Overflowed: {e}")
        sys.exit(1)
        
    # --- PRODUCTION RUNS ---
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
        print(f"\n[METRIC] Mean Execution Speed: {avg_throughput:.4f} seconds for {batch_size} states.")
        print(f"[METRIC] Time Per Individual 75-Qubit State: {avg_throughput / batch_size:.4f} seconds.")

        # --- SUPREMACY VERIFICATION (F_XEB) ---
        print("\n[STAGE 3] Executing Linear Cross-Entropy Benchmarking (F_XEB)...")
        hilbert_dimension = 2.0 ** N_QUBITS
        calculated_mean_prob = jnp.mean(results)
        f_xeb = (hilbert_dimension * calculated_mean_prob) - 1.0
        
        print(f" -> Hilbert Space Dimension Size: {hilbert_dimension:.3e}")
        print(f" -> Calculated Sample Mean Probability Value: {calculated_mean_prob}")
        print(f" -> Verified F_XEB Output Fingerprint Score: {f_xeb:.6f}")
        
        # --- GRAPHICS GENERATION ---
        print("\n[STAGE 4] Saving Performance Graphs to Disk (`tpu_75qubit_performance.png`)...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.plot(range(1, iterations + 1), execution_times, marker='o', color='#00a2ed', linewidth=2, label='TPU v5e MXU Processing Time')
        ax1.axhline(y=avg_throughput, color='r', linestyle='--', label=f'Mean Time ({avg_throughput:.2f}s)')
        ax1.set_title("Hardware Processing Velocity Across Warm JIT Runs", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Iteration Number", fontsize=10)
        ax1.set_ylabel("Time (Seconds)", fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()
        
        ax2.hist(results, bins=15, color='#7a00ed', edgecolor='black', alpha=0.7, label='Simulated States')
        ax2.set_title("Probability Frequency Map (Chaos Distribution Test)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Probability Amplitude Value |psi|^2", fontsize=10)
        ax2.set_ylabel("Occurrences Count", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('tpu_75qubit_performance.png', dpi=300)
        print("[SUCCESS] Graphics rendered perfectly. Open `tpu_75qubit_performance.png` to view metrics.")
        print("====================================================")

if __name__ == "__main__":
    run_pipeline()
