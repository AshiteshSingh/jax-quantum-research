#!/usr/bin/env python3
"""
================================================================================
  Grover's Algorithm — 20-Qubit Full State-Vector Simulation
  JAX accelerated · CPU / GPU / TPU (PositionalSharding)

  This is a REAL quantum circuit simulation — not an analytical formula.
  Every step below manipulates 2^20 = 1,048,576 complex64 amplitudes.

  Circuit structure per Grover iteration:
    1. Phase oracle    : |x⟩ → -|x⟩ if x == MARKED, else |x⟩  (sign flip in state vector)
    2. Diffusion (Ds)  : 2|s⟩⟨s| - I  =  inversion about the global mean amplitude

  Initialization:
    Uniform superposition  ≡  H^⊗20 |0⟩^⊗20  →  all 2^20 amplitudes = 1/√(2^20)

  Hardware
  ────────
  Tested on:  CPU (single device, 8 MB state, runs in < 1 s)
              GPU (single device, same)
              TPU v5e-16 / v6e-64  (sharded across all chips)

  Key numbers
  ───────────
  Search space  : N = 2^20 = 1,048,576
  State vector  : complex64 · 8 MB
  Marked state  : |111...1⟩  (index N-1)
  k_opt         : 804 iterations → P(success) > 0.9999
  Grover speedup: √N ≈ 1,024×  (vs classical O(N) average)
================================================================================
"""

import os, sys, time, json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
from jax import config
config.update("jax_enable_x64", False)   # keep complex64 throughout
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, PositionalSharding
from jax.experimental.multihost_utils import process_allgather

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Output dirs
# ─────────────────────────────────────────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("grover_simulation/results", exist_ok=True)
os.makedirs("grover_simulation/plots",   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Quantum circuit parameters
# ─────────────────────────────────────────────────────────────────────────────
N_QUBITS  = 20
N_TOTAL   = 1 << N_QUBITS          # 2^20 = 1,048,576
MARKED    = N_TOTAL - 1            # |111...1⟩  (all-ones bitstring)

# ─────────────────────────────────────────────────────────────────────────────
# Device / sharding setup
# ─────────────────────────────────────────────────────────────────────────────
DEVICES  = jax.devices()
NUM_DEV  = len(DEVICES)
BACKEND  = jax.default_backend()
TPU_MESH = Mesh(np.array(DEVICES), ("dev",))
P_SPEC   = PartitionSpec("dev",)
SHARDING = PositionalSharding(DEVICES)

# ─────────────────────────────────────────────────────────────────────────────
# Grover theory
# ─────────────────────────────────────────────────────────────────────────────
theta    = np.arcsin(1.0 / np.sqrt(float(N_TOTAL)))
K_OPT    = int(np.round(np.pi / (4.0 * theta) - 0.5))
PROB_OPT = float(np.sin((2 * K_OPT + 1) * theta) ** 2)
SPEEDUP  = np.sqrt(float(N_TOTAL))
MEM_BYTES = N_TOTAL * 8   # complex64 = 8 bytes

def theory_prob(k):
    """Analytical success probability after k Grover iterations."""
    return np.sin((2 * k + 1) * theta) ** 2

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
def banner(msg):
    w = 76
    print("\n" + "═" * w)
    print(f"  {msg.center(w - 4)}")
    print("═" * w)

banner("Grover's Algorithm — 20-Qubit Real Statevector Simulation")
print(f"  Backend      : {BACKEND.upper()}")
print(f"  Devices      : {NUM_DEV}  ({[str(d) for d in DEVICES]})")
print(f"  Qubits       : {N_QUBITS}")
print(f"  Search space : N = {N_TOTAL:,}")
print(f"  State vector : {MEM_BYTES / 1e6:.1f} MB  (complex64, {N_TOTAL:,} amplitudes)")
print(f"  Marked state : |{'1'*N_QUBITS}⟩  =  index {MARKED:,}")
print(f"  k_opt        : {K_OPT:,} iterations")
print(f"  P(success)   : {PROB_OPT:.10f}  (theory)")
print(f"  Grover speedup: ≈ {SPEEDUP:,.0f}×  vs classical O(N) search")

# ─────────────────────────────────────────────────────────────────────────────
# Oracle — real quantum gate: flips sign of |marked⟩ amplitude
#
# In the flat 1-D state vector sharded across NUM_DEV devices:
#   MARKED = N_TOTAL - 1  lives at local_index = N_LOCAL - 1  on device NUM_DEV - 1
#
# oracle(|ψ⟩) = O_f|ψ⟩  where  O_f|x⟩ = (-1)^{f(x)}|x⟩,  f(x) = [x == MARKED]
# ─────────────────────────────────────────────────────────────────────────────
@partial(shard_map, mesh=TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)
def oracle(local_s):
    """Phase oracle: flips sign of the marked state |111...1⟩ amplitude."""
    d = jax.lax.axis_index("dev")
    # MARKED = N_TOTAL - 1 is the last element of the last shard
    is_marked_dev = (d == NUM_DEV - 1)
    sign = jnp.where(
        is_marked_dev,
        jnp.array(-1.0, dtype=jnp.complex64),
        jnp.array( 1.0, dtype=jnp.complex64),
    )
    return local_s.at[-1].multiply(sign)

# ─────────────────────────────────────────────────────────────────────────────
# Diffusion — real quantum gate: 2|s⟩⟨s| - I  =  inversion about the mean
#
# For state |ψ⟩ = Σ αᵢ|i⟩:
#   D|ψ⟩ᵢ  =  2 · mean(α)  -  αᵢ
#
# Requires ONE allreduce (psum) across devices — minimal communication.
# ─────────────────────────────────────────────────────────────────────────────
_INV_N = np.float32(1.0 / float(N_TOTAL))

@partial(shard_map, mesh=TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)
def diffusion(local_s):
    """Grover diffusion operator: inversion of all amplitudes about their mean."""
    local_sum  = jnp.sum(local_s)
    global_sum = lax.psum(local_sum, axis_name="dev")    # single allreduce
    global_mean = global_sum * _INV_N
    return jnp.complex64(2.0) * global_mean - local_s

# ─────────────────────────────────────────────────────────────────────────────
# One Grover iteration: oracle → diffusion
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def grover_step(state):
    state = oracle(state)
    state = diffusion(state)
    return state

# Runs k Grover steps using lax.fori_loop → O(1) XLA graph size for any k
@partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
def run_steps(state, k):
    return lax.fori_loop(0, k, lambda i, s: grover_step(s), state)

# ─────────────────────────────────────────────────────────────────────────────
# State initialization: uniform superposition ≡ H^⊗n |0⟩^⊗n
# All 2^20 amplitudes set to 1/√N — equivalent to applying Hadamard to every qubit
# ─────────────────────────────────────────────────────────────────────────────
INV_SQRT_N = np.float32(1.0 / np.sqrt(float(N_TOTAL)))

# Using shard_map to create the state in distributed fashion
# Each device creates its N_LOCAL slice directly — no host-side 8MB allocation needed
N_LOCAL = N_TOTAL // NUM_DEV

@partial(shard_map, mesh=TPU_MESH,
         in_specs=PartitionSpec("dev",), out_specs=PartitionSpec("dev",))
def _init_shard(dummy):
    return jnp.full((N_LOCAL,), INV_SQRT_N, dtype=jnp.complex64)

dummy_input = jnp.zeros(NUM_DEV)
state = _init_shard(dummy_input)

# ─────────────────────────────────────────────────────────────────────────────
# JIT warm-up (excludes from timing)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Warmup] Compiling JIT kernels ...", flush=True)
t_warm = time.perf_counter()
_ = run_steps(state, 1).block_until_ready()
print(f"  [Warmup] Done  ({time.perf_counter() - t_warm:.2f}s)")

# Reset state after warmup
state = _init_shard(dummy_input)

# ─────────────────────────────────────────────────────────────────────────────
# Main simulation — collect probability snapshots every SNAP_EVERY steps
# ─────────────────────────────────────────────────────────────────────────────
SNAP_EVERY = max(1, K_OPT // 100)   # ≈100 snapshots over the full run

snapshots_itr  = []
snapshots_prob = []

print(f"\n  [Run] Executing {K_OPT:,} Grover iterations (snapshots every {SNAP_EVERY}) ...\n")
t0   = time.perf_counter()
itr  = 0

while itr < K_OPT:
    chunk = min(SNAP_EVERY, K_OPT - itr)
    state = run_steps(state, chunk)
    state.block_until_ready()
    itr  += chunk
    # Read probability of measuring |MARKED⟩  (JAX handles sharded indexing)
    p = float(jnp.abs(state[MARKED]) ** 2)
    snapshots_itr.append(itr)
    snapshots_prob.append(p)
    if itr % (SNAP_EVERY * 10) == 0 or itr == K_OPT:
        print(f"    iter {itr:>6,}/{K_OPT:,}  P(|marked⟩) = {p:.6f}")

elapsed = time.perf_counter() - t0
final_prob = snapshots_prob[-1]

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print(f"  Qubits              : {N_QUBITS}")
print(f"  Search space        : {N_TOTAL:,}")
print(f"  Optimal iterations  : {K_OPT:,}")
print(f"  P(success) theory   : {PROB_OPT:.10f}")
print(f"  P(success) measured : {final_prob:.10f}")
print(f"  Accuracy (%)        : {final_prob * 100:.6f}%")
print(f"  Theoretical speedup : ≈ {SPEEDUP:,.0f}×  vs classical O(N)")
print(f"  Simulation time     : {elapsed:.4f} s  ({elapsed*1e3/K_OPT:.3f} ms/iter)")
print(f"  Backend             : {BACKEND.upper()},  {NUM_DEV} device(s)")
print("─" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme plots
# ─────────────────────────────────────────────────────────────────────────────
P = {
    "bg": "#0d1117", "panel": "#161b22", "border": "#30363d",
    "text": "#e6edf3", "sub": "#8b949e", "grid": "#21262d",
    "a1": "#58a6ff", "a2": "#3fb950", "a3": "#f78166",
    "a4": "#d2a8ff", "a5": "#ffa657",
}

def theme(fig, axes_list):
    fig.patch.set_facecolor(P["bg"])
    for ax in (axes_list if hasattr(axes_list, "__iter__") else [axes_list]):
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["text"], labelsize=9)
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["border"])
        ax.grid(True, color=P["grid"], ls="--", alpha=0.5, lw=0.6)

# Build theory curve (dense)
k_theory  = np.arange(0, K_OPT + 1)
p_theory  = theory_prob(k_theory)

fig = plt.figure(figsize=(18, 10), dpi=150)
fig.patch.set_facecolor(P["bg"])
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                       left=0.07, right=0.97, top=0.90, bottom=0.08)

# ── Panel 1: Full probability trajectory ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(k_theory, p_theory, color=P["sub"], lw=1.2, label="Theory sin²((2k+1)θ)", alpha=0.6)
ax1.plot(snapshots_itr, snapshots_prob, "o-", color=P["a1"], ms=3, lw=1.8,
         label="Simulated P(|marked⟩)", zorder=3)
ax1.axvline(K_OPT, color=P["a2"], ls="--", lw=1.5, label=f"k_opt = {K_OPT:,}")
ax1.scatter([K_OPT], [final_prob], color=P["a2"], s=80, zorder=5)
ax1.set_xlabel("Grover Iterations k")
ax1.set_ylabel("P(success)")
ax1.set_title(f"🔍  Grover Probability Growth — {N_QUBITS} Qubits\n(Real statevector, 2^{N_QUBITS} = {N_TOTAL:,} amplitudes)")
ax1.set_ylim(0, 1.05)
ax1.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
theme(fig, ax1)

# ── Panel 2: Zoom around k_opt ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
zoom_lo = max(0, K_OPT - 60)
zoom_hi = min(len(k_theory) - 1, K_OPT + 60)
ax2.plot(k_theory[zoom_lo:zoom_hi], p_theory[zoom_lo:zoom_hi],
         color=P["a4"], lw=2.2, label="Theory")
zoom_itr  = [i for i in snapshots_itr  if zoom_lo <= i <= zoom_hi]
zoom_prob = [p for i, p in zip(snapshots_itr, snapshots_prob) if zoom_lo <= i <= zoom_hi]
ax2.plot(zoom_itr, zoom_prob, "o", color=P["a1"], ms=5, label="Simulated")
ax2.axvline(K_OPT, color=P["a2"], ls="--", lw=1.5)
ax2.set_xlabel("Grover Iterations k")
ax2.set_ylabel("P(success)")
ax2.set_title(f"🔬  Zoom near k_opt = {K_OPT:,}\n(Theory vs measured statevector)")
ax2.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
theme(fig, ax2)

# ── Panel 3: Final measurement bar ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
labels = [f"|{'0'*N_QUBITS}⟩", f"|{'1'*N_QUBITS}⟩"]
values = [1.0 - final_prob, final_prob]
bars = ax3.bar(labels, values, color=[P["sub"], P["a2"]], width=0.5,
               edgecolor=P["border"])
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             val + 0.02, f"{val*100:.4f}%", ha="center",
             color=P["text"], fontsize=10, fontweight="bold")
ax3.set_ylim(0, 1.10)
ax3.set_title(f"📊  Final Measurement Probabilities\n(After k_opt = {K_OPT:,} iterations)")
ax3.set_ylabel("Probability")
theme(fig, ax3)

# ── Panel 4: Iteration cost comparison ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
qubit_range  = np.arange(4, N_QUBITS + 1)
grover_iters = [int(np.round(np.pi / 4 * np.sqrt(2**n) - 0.5)) for n in qubit_range]
classical    = [2**(n-1) for n in qubit_range]   # average classical = N/2
ax4.semilogy(qubit_range, classical,    "s--", color=P["a3"], lw=2,   ms=6, label="Classical (N/2 avg)")
ax4.semilogy(qubit_range, grover_iters, "o-",  color=P["a1"], lw=2.2, ms=6, label="Grover O(√N)")
ax4.axvline(N_QUBITS, color=P["a2"], ls=":", lw=1.5, label=f"This run ({N_QUBITS}q)")
ax4.set_xlabel("Qubits (n)")
ax4.set_ylabel("Oracle Calls (log scale)")
ax4.set_title("⚡  Query Complexity: Grover vs Classical\n(quadratic quantum speedup)")
ax4.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
theme(fig, ax4)

fig.suptitle(
    f"Grover's Algorithm — 20-Qubit Full Statevector Simulation  |  JAX {jax.__version__}  |  {BACKEND.upper()}"
    f"\nk_opt = {K_OPT:,}  ·  P(success) = {final_prob:.8f}  ·  Time = {elapsed:.3f}s  ·  {TS}",
    color=P["text"], fontsize=11, fontweight="bold", y=0.97,
)

plot_path = f"grover_simulation/plots/grover_20q_{TS}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close()
print(f"\n  📈 Plot saved  → {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Save JSON results
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "meta": {
        "timestamp": TS, "backend": BACKEND, "n_devices": NUM_DEV,
        "jax_version": jax.__version__, "script": "grover_simulation/20qubits.py",
        "simulation_type": "real_statevector_jax",
    },
    "circuit": {
        "n_qubits": N_QUBITS, "N_total": N_TOTAL,
        "marked_state": MARKED, "marked_bitstring": "1" * N_QUBITS,
        "state_vector_bytes": MEM_BYTES,
    },
    "theory": {
        "k_opt": K_OPT, "prob_opt_theory": PROB_OPT,
        "grover_speedup": float(SPEEDUP),
        "theta_rad": float(theta),
    },
    "simulation": {
        "k_opt_run": K_OPT, "final_prob_measured": final_prob,
        "elapsed_s": elapsed, "ms_per_iter": elapsed * 1e3 / K_OPT,
        "snapshots_itr": snapshots_itr, "snapshots_prob": snapshots_prob,
    },
}
json_path = f"grover_simulation/results/grover_20q_{TS}.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  📄 JSON saved  → {json_path}")
print()
