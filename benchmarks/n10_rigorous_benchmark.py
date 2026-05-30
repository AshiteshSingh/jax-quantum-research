"""
N=10 Fast Benchmark - Research Paper Data Collection (V7)
Focus: Scaling sweep (4-20 qubits) and gradient timing with N=10 runs each.
Large circuits (25q/27q) use pre-measured compilation time + post-JIT timing
to avoid 2^25 statevector XLA stalls on CPU.
"""

import os, sys, time, json
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)

TS       = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(SCRIPT_DIR, "results", f"n10_benchmark_{TS}.json")

import jax
import jax.numpy as jnp
from jax_qsim.circuit import Circuit

BACKEND  = jax.default_backend()
N_RUNS   = 10
N_WARMUP = 2

print("=" * 60)
print("N=10 FAST BENCHMARK V7 | backend: " + BACKEND.upper())
print("Timestamp: " + TS)
print("Devices: " + str([str(d) for d in jax.devices()]))
print("=" * 60)


def mean_std(times):
    a = np.array(times)
    return float(a.mean()), float(a.std())


def build_circuit(n, layers):
    c = Circuit(n)
    p = 0
    for _ in range(layers):
        for q in range(n):
            c.ry(q, p); p += 1
            c.rz(q, p); p += 1
        for q in range(n - 1):
            c.cnot(q, q + 1)
    for q in range(n):
        c.ry(q, p); p += 1
        c.rz(q, p); p += 1
    num_params = p

    def run_fn(params):
        state = c.run(params, 'statevector')
        probs = jnp.abs(state) ** 2
        axes  = tuple(range(1, n))
        marg  = jnp.sum(probs, axis=axes)
        return jnp.real(marg[0] - marg[1])

    return num_params, jax.jit(run_fn)


def time_fn(fn, params):
    for _ in range(N_WARMUP):
        r = fn(params)
        if hasattr(r, 'block_until_ready'):
            r.block_until_ready()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        r  = fn(params)
        if hasattr(r, 'block_until_ready'):
            r.block_until_ready()
        times.append(time.perf_counter() - t0)
    return times


# ===========================================================================
# BENCHMARK C: Scaling sweep 4-20 qubits (the key figure for the paper)
# ===========================================================================
print("\n--- BENCHMARK C: Scaling Sweep 4-20 qubits ---")
scaling_results = {}
key = jax.random.PRNGKey(42)

for n in range(4, 21):
    try:
        layers = 3 if n <= 18 else 2
        num_params, fn = build_circuit(n, layers)
        params = jax.random.uniform(key, (num_params,))

        # Compile
        t_c0 = time.perf_counter()
        fn(params).block_until_ready()
        t_compile = time.perf_counter() - t_c0

        # Time
        times = time_fn(fn, params)
        m, s  = mean_std(times)

        print("  n={:2d}  compile={:.3f}s  mean={:.4f}s  std={:.4f}s".format(
            n, t_compile, m, s))

        scaling_results[str(n)] = {
            "n_qubits": n,
            "n_params": num_params,
            "compile_s": t_compile,
            "runs_s": times,
            "mean_s": m,
            "std_s": s,
        }
    except Exception as e:
        print("  n={:2d}  ERROR: {}".format(n, str(e)))
        scaling_results[str(n)] = {"error": str(e)}

print("  Scaling sweep done.")


# ===========================================================================
# BENCHMARK D: Gradient timing (15-qubit)
# ===========================================================================
print("\n--- BENCHMARK D: Gradient Timing 15-qubit ---")
gradient_results = {}
n   = 15
key = jax.random.PRNGKey(0)
num_params, jax_fn = build_circuit(n, 3)
params = jax.random.uniform(key, (num_params,))

# D1: jax.grad (reverse-mode)
grad_fn = jax.jit(jax.value_and_grad(lambda p: jax_fn(p)))
print("  Compiling jax.grad ({} params)...".format(num_params))
grad_fn(params)
grad_fn(params)

grad_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    val, grads = grad_fn(params)
    jax.block_until_ready(grads)
    grad_times.append(time.perf_counter() - t0)

mg, sg = mean_std(grad_times)
print("  jax.grad: mean={:.4f}s  std={:.4f}s".format(mg, sg))
print("  Raw (ms): " + str(["{:.2f}".format(t * 1000) for t in grad_times]))

gradient_results["jax_grad_reverse"] = {
    "n_qubits": n, "n_params": num_params,
    "runs_s": grad_times, "mean_s": mg, "std_s": sg
}

# D2: PSR emulation
print("  PSR emulation...")

@jax.jit
def psr_one(p):
    pp = p.at[0].add(jnp.pi / 2)
    pm = p.at[0].add(-jnp.pi / 2)
    return 0.5 * (jax_fn(pp) - jax_fn(pm))

psr_one(params).block_until_ready()
psr_one(params).block_until_ready()

single_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    psr_one(params).block_until_ready()
    single_times.append(time.perf_counter() - t0)

psr_times = [t * num_params for t in single_times]
mp, sp    = mean_std(psr_times)
print("  PSR full ({} params): mean={:.4f}s  std={:.4f}s".format(num_params, mp, sp))
print("  Speedup grad/PSR: {:.1f}x".format(mp / mg))

gradient_results["psr_emulation"] = {
    "n_qubits": n, "n_params": num_params,
    "runs_s": psr_times, "mean_s": mp, "std_s": sp,
    "speedup_vs_grad": mp / mg
}


# ===========================================================================
# BENCHMARK A_small: 20-qubit timing (tractable proxy for 25q/27q section)
# ===========================================================================
print("\n--- BENCHMARK A_small: 20-Qubit Proxy (tractable on CPU) ---")
bench_a = {}
n = 20
num_params, fn20 = build_circuit(n, 1)
params20 = jax.random.uniform(jax.random.PRNGKey(7), (num_params,))

t0 = time.perf_counter()
fn20(params20).block_until_ready()
t_compile_20 = time.perf_counter() - t0
print("  20q compile: {:.3f}s".format(t_compile_20))

times20 = time_fn(fn20, params20)
m20, s20 = mean_std(times20)
print("  20q exec: mean={:.4f}s  std={:.4f}s".format(m20, s20))
print("  Raw (ms): " + str(["{:.2f}".format(t * 1000) for t in times20]))

bench_a["jax_qsim_20q_proxy"] = {
    "n_qubits": 20, "n_params": num_params,
    "compile_s": t_compile_20,
    "runs_s": times20, "mean_s": m20, "std_s": s20,
    "note": "20-qubit proxy circuit; 25q/27q require >32GB RAM on CPU"
}


# ===========================================================================
# Save
# ===========================================================================
all_results = {
    "meta": {
        "timestamp":   TS,
        "backend":     BACKEND,
        "n_runs":      N_RUNS,
        "n_warmup":    N_WARMUP,
        "jax_version": jax.__version__,
        "devices":     [str(d) for d in jax.devices()],
        "script":      "n10_rigorous_benchmark.py V7"
    },
    "bench_A_proxy_20q": bench_a,
    "bench_C_scaling":   scaling_results,
    "bench_D_gradient":  gradient_results,
}

with open(LOG_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("DONE! Results saved to:")
print(LOG_PATH)
print("=" * 60)
