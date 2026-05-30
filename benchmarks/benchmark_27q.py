"""
27-Qubit Performance Benchmark: jax_qsim vs PennyLane vs Qiskit vs Cirq vs TensorFlow Quantum
Statevector: 2^27 complex64 elements = 1 GB RAM/VRAM
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Optional Imports
try:
    import pennylane as qml
    has_pennylane = True
except ImportError:
    has_pennylane = False

try:
    import qiskit
    from qiskit_aer import Aer
    has_qiskit = True
except ImportError:
    has_qiskit = False

try:
    import cirq
    has_cirq = True
except ImportError:
    has_cirq = False

try:
    import tensorflow as tf
    import tensorflow_quantum as tfq
    has_tfq = True
except ImportError:
    has_tfq = False

# Ensure jax_qsim is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_qsim.circuit import Circuit
from jax_qsim.statevector import zero_state

NUM_QUBITS = 27
NUM_REPEATS = 3
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Detect JAX Devices
try:
    gpu_device = jax.devices("gpu")[0]
    print(f"[INFO] GPU detected: {gpu_device}")
except Exception:
    gpu_device = None
    print("[WARNING] No JAX GPU device detected. Falling back to CPU for native JAX.")

# ==============================================================================
# 1. build_jax_qsim
# ==============================================================================
def run_our_simulator(state, params):
    c = Circuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        c.h(q)
    for q in range(NUM_QUBITS - 1):
        c.cnot(q, q + 1)
    for q in range(NUM_QUBITS):
        c.ry(q, q)
        
    def run_fn(state, p):
        final_state = c.run(p, 'statevector', initial_state=state)
        state_flat = final_state.reshape(-1)
        probs = jnp.abs(state_flat) ** 2
        half = 1 << (NUM_QUBITS - 1)
        marginal_0 = jnp.sum(probs[:half])
        marginal_1 = jnp.sum(probs[half:])
        return jnp.real(marginal_0 - marginal_1)
        
    target_device = gpu_device if gpu_device else jax.devices("cpu")[0]
    return jax.jit(run_fn, device=target_device)

# ==============================================================================
# 2. build_pennylane
# ==============================================================================
def run_pennylane(device_name="default.qubit"):
    if "gpu" in device_name or "cuda" in device_name:
        dev = qml.device("lightning.gpu", wires=NUM_QUBITS)
    elif device_name == "lightning.qubit":
        dev = qml.device("lightning.qubit", wires=NUM_QUBITS)
    else:
        dev = qml.device("default.qubit", wires=NUM_QUBITS)
        
    @qml.qnode(dev, interface="jax" if "lightning" not in device_name else None)
    def circuit(p):
        for q in range(NUM_QUBITS):
            qml.Hadamard(wires=q)
        for q in range(NUM_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
        for q in range(NUM_QUBITS):
            qml.RY(p[q], wires=q)
        return qml.expval(qml.PauliZ(0))
        
    if "lightning" in device_name:
        return circuit
    return jax.jit(circuit)

# ==============================================================================
# 3. build_qiskit
# ==============================================================================
def run_qiskit(use_gpu=False):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    
    # Initialize Aer backend
    if use_gpu:
        backend = Aer.get_backend('statevector_simulator')
        backend.set_options(device='GPU')
    else:
        backend = Aer.get_backend('statevector_simulator')
        backend.set_options(device='CPU')
        
    def run_fn(p):
        qc = QuantumCircuit(NUM_QUBITS)
        for q in range(NUM_QUBITS):
            qc.h(q)
        for q in range(NUM_QUBITS - 1):
            qc.cx(q, q + 1)
        for q in range(NUM_QUBITS):
            qc.ry(p[q], q)
        
        # Save statevector expectation
        qc.save_statevector()
        job = backend.run(qc)
        result = job.result()
        sv = result.get_statevector(qc)
        
        # Expectation value of Z0
        op = SparsePauliOp.from_list([("I" * (NUM_QUBITS - 1) + "Z", 1.0)])
        return sv.expectation_value(op).real
        
    return run_fn

# ==============================================================================
# 4. build_cirq
# ==============================================================================
def run_cirq():
    qubits = cirq.LineQubit.range(NUM_QUBITS)
    sim = cirq.Simulator()
    observable = cirq.Z(qubits[0])
    
    def run_fn(p):
        qc = cirq.Circuit()
        for q in range(NUM_QUBITS):
            qc.append(cirq.H(qubits[q]))
        for q in range(NUM_QUBITS - 1):
            qc.append(cirq.CNOT(qubits[q], qubits[q + 1]))
        for q in range(NUM_QUBITS):
            qc.append(cirq.ry(p[q])(qubits[q]))
            
        res = sim.simulate(qc)
        return observable.expectation_from_state_vector(
            res.final_state_vector, 
            qubit_map={q: i for i, q in enumerate(qubits)}
        ).real
        
    return run_fn

# ==============================================================================
# 5. build_tfq
# ==============================================================================
def run_tfq():
    # Helper to build TFQ graph
    import sympy
    qubits = cirq.LineQubit.range(NUM_QUBITS)
    symbols = [sympy.Symbol(f'theta_{i}') for i in range(NUM_QUBITS)]
    
    # Define circuit in Cirq schema for TFQ conversion
    qc = cirq.Circuit()
    for q in range(NUM_QUBITS):
        qc.append(cirq.H(qubits[q]))
    for q in range(NUM_QUBITS - 1):
        qc.append(cirq.CNOT(qubits[q], qubits[q + 1]))
    for q in range(NUM_QUBITS):
        qc.append(cirq.ry(symbols[q])(qubits[q]))
        
    # Operator for expectation
    op = tfq.convert_to_tensor([cirq.Z(qubits[0])])
    circuit_tensor = tfq.convert_to_tensor([qc])
    
    # Create expectation layer
    expectation_layer = tfq.layers.Expectation()
    
    def run_fn(p):
        symbol_values = tf.convert_to_tensor([p], dtype=tf.float32)
        res = expectation_layer(circuit_tensor, symbol_names=[str(s) for s in symbols], symbol_values=symbol_values, operators=op)
        return float(res.numpy()[0][0])
        
    return run_fn

# ==============================================================================
# Main execution block
# ==============================================================================
def main():
    print("=" * 80)
    print(" COMPREHENSIVE 27-QUBIT SIMULATOR PERFORMANCE BENCHMARK ".center(80, "="))
    print("=" * 80)
    print("Statevector Size : 2^27 complex64 elements = 1 GB VRAM/RAM")
    print(f"JAX Device       : {gpu_device if gpu_device else 'CPU (Fallback)'}")
    print("-" * 80)
    
    params = jax.random.uniform(jax.random.PRNGKey(42), shape=(NUM_QUBITS,))
    params_np = np.array(params)
    
    benchmark_results = {}
    
    # --- 1. jax_qsim ---
    print("\nRunning jax_qsim (Our Pure JAX Simulator)...")
    try:
        state = zero_state(NUM_QUBITS)
        run_fn = run_our_simulator(state, params)
        t_start = time.time()
        val = run_fn(state, params).block_until_ready()
        t_comp = time.time() - t_start
        print(f"  Warmup / JIT compilation: {t_comp:.3f}s")
        
        times = []
        for _ in range(NUM_REPEATS):
            t0 = time.time()
            _ = run_fn(state, params).block_until_ready()
            times.append(time.time() - t0)
        our_mean = np.mean(times)
        print(f"  Average Execution Speed  : {our_mean:.3f}s")
        benchmark_results['our_jax_qsim'] = {'compilation': t_comp, 'execution': our_mean, 'status': 'success', 'expectation': float(val)}
    except Exception as e:
        print(f"  Failed: {str(e)}")
        benchmark_results['our_jax_qsim'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}
        
    # --- 2. PennyLane Lightning CPU ---
    if has_pennylane:
        print("\nRunning PennyLane Lightning (C++ CPU Engine)...")
        try:
            pl_light = run_pennylane("lightning.qubit")
            t_start = time.time()
            val_light = pl_light(params_np)
            t_comp = time.time() - t_start
            print(f"  Warmup execution         : {t_comp:.3f}s")
            
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = pl_light(params_np)
                times.append(time.time() - t0)
            light_mean = np.mean(times)
            print(f"  Average Execution Speed  : {light_mean:.3f}s")
            benchmark_results['pennylane_lightning_cpu'] = {'compilation': t_comp, 'execution': light_mean, 'status': 'success', 'expectation': float(val_light)}
        except Exception as e:
            print(f"  Failed: {str(e)}")
            benchmark_results['pennylane_lightning_cpu'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}

        # --- 3. PennyLane Lightning GPU ---
        print("\nRunning PennyLane Lightning GPU (lightning.gpu)...")
        try:
            pl_gpu = run_pennylane("lightning.gpu")
            t_start = time.time()
            val_gpu = pl_gpu(params_np)
            t_comp = time.time() - t_start
            print(f"  Warmup execution         : {t_comp:.3f}s")
            
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = pl_gpu(params_np)
                times.append(time.time() - t0)
            gpu_mean = np.mean(times)
            print(f"  Average Execution Speed  : {gpu_mean:.3f}s")
            benchmark_results['pennylane_lightning_gpu'] = {'compilation': t_comp, 'execution': gpu_mean, 'status': 'success', 'expectation': float(val_gpu)}
        except Exception as e:
            print(f"  Failed / Skipped: {str(e)}")
            benchmark_results['pennylane_lightning_gpu'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}

    # --- 4. Qiskit Aer CPU/GPU ---
    if has_qiskit:
        print("\nRunning Qiskit Aer (Statevector CPU)...")
        try:
            qk_fn = run_qiskit(use_gpu=False)
            t_start = time.time()
            val_qk = qk_fn(params_np)
            t_comp = time.time() - t_start
            print(f"  Warmup execution         : {t_comp:.3f}s")
            
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = qk_fn(params_np)
                times.append(time.time() - t0)
            qk_mean = np.mean(times)
            print(f"  Average Execution Speed  : {qk_mean:.3f}s")
            benchmark_results['qiskit_aer_cpu'] = {'compilation': t_comp, 'execution': qk_mean, 'status': 'success', 'expectation': float(val_qk)}
        except Exception as e:
            print(f"  Failed: {str(e)}")
            benchmark_results['qiskit_aer_cpu'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}

        # Aer GPU
        print("\nRunning Qiskit Aer GPU (CUDA-accelerated)...")
        try:
            qk_gpu_fn = run_qiskit(use_gpu=True)
            t_start = time.time()
            val_qk_gpu = qk_gpu_fn(params_np)
            t_comp = time.time() - t_start
            print(f"  Warmup execution         : {t_comp:.3f}s")
            
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = qk_gpu_fn(params_np)
                times.append(time.time() - t0)
            qkgpu_mean = np.mean(times)
            print(f"  Average Execution Speed  : {qkgpu_mean:.3f}s")
            benchmark_results['qiskit_aer_gpu'] = {'compilation': t_comp, 'execution': qkgpu_mean, 'status': 'success', 'expectation': float(val_qk_gpu)}
        except Exception as e:
            print(f"  Failed / Skipped: {str(e)}")
            benchmark_results['qiskit_aer_gpu'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}
    else:
        print("\nQiskit: Skipped (Not Installed)")
        benchmark_results['qiskit_aer_cpu'] = {'compilation': 0.0, 'execution': 0.0, 'status': 'skipped'}

    # --- 5. Cirq Simulator ---
    if has_cirq:
        print("\nRunning Cirq Simulator...")
        try:
            cirq_fn = run_cirq()
            t_start = time.time()
            val_cirq = cirq_fn(params_np)
            t_comp = time.time() - t_start
            print(f"  Warmup execution         : {t_comp:.3f}s")
            
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = cirq_fn(params_np)
                times.append(time.time() - t0)
            cirq_mean = np.mean(times)
            print(f"  Average Execution Speed  : {cirq_mean:.3f}s")
            benchmark_results['cirq_simulator'] = {'compilation': t_comp, 'execution': cirq_mean, 'status': 'success', 'expectation': float(val_cirq)}
        except Exception as e:
            print(f"  Failed: {str(e)}")
            benchmark_results['cirq_simulator'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}
    else:
        print("\nCirq: Skipped (Not Installed)")
        benchmark_results['cirq_simulator'] = {'compilation': 0.0, 'execution': 0.0, 'status': 'skipped'}

    # --- 6. TensorFlow Quantum ---
    if has_tfq:
        print("\nRunning TensorFlow Quantum...")
        try:
            tfq_fn = run_tfq()
            t_start = time.time()
            val_tfq = tfq_fn(params_np)
            t_comp = time.time() - t_start
            print(f"  Warmup execution         : {t_comp:.3f}s")
            
            times = []
            for _ in range(NUM_REPEATS):
                t0 = time.time()
                _ = tfq_fn(params_np)
                times.append(time.time() - t0)
            tfq_mean = np.mean(times)
            print(f"  Average Execution Speed  : {tfq_mean:.3f}s")
            benchmark_results['tensorflow_quantum'] = {'compilation': t_comp, 'execution': tfq_mean, 'status': 'success', 'expectation': float(val_tfq)}
        except Exception as e:
            print(f"  Failed: {str(e)}")
            benchmark_results['tensorflow_quantum'] = {'compilation': 0.0, 'execution': 0.0, 'status': f'failed: {str(e)}'}
    else:
        print("\nTensorFlow Quantum: Skipped (Not Installed/Unsupported on Python 3.12 Windows)")
        benchmark_results['tensorflow_quantum'] = {'compilation': 0.0, 'execution': 0.0, 'status': 'skipped'}

    # Save results
    json_path = os.path.join(results_dir, "27q_comprehensive_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
        
    # Plot results
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    
    labels = []
    exec_times = []
    comp_times = []
    
    for key, info in benchmark_results.items():
        if info['status'] == 'success':
            labels.append(key.replace('_', '\n'))
            exec_times.append(info['execution'])
            comp_times.append(info['compilation'])
            
    if exec_times:
        x = np.arange(len(labels))
        width = 0.35
        rects1 = ax.bar(x - width/2, exec_times, width, label='Execution Time (seconds)', color='#58a6ff', edgecolor='#30363d')
        rects2 = ax.bar(x + width/2, comp_times, width, label='Compilation / Warmup Time', color='#ff7b72', edgecolor='#30363d')
        
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', color='#58a6ff', fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', color='#ff7b72', fontweight='bold')
            
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color='#c9d1d9')
        
    ax.set_title("27-Qubit Comprehensive State Vector Simulator: CUDA & Hardware Comparison", fontsize=14, color='#e6edf3', fontweight='bold', pad=20)
    ax.set_ylabel("Time (seconds) - Lower is Better", color='#8b949e', labelpad=10)
    ax.grid(True, linestyle='--', color='#21262d', alpha=0.6)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    
    plot_path = os.path.join(results_dir, "27q_comprehensive_comparison.png")
    plt.savefig(plot_path, dpi=300, facecolor='#0d1117')
    plt.close()
    print(f"Comparative visualization saved to: {plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
