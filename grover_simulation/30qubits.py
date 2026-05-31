#!/usr/bin/env python3
"""
================================================================================
  Grover's Algorithm — 30-Qubit Full State-Vector Simulation
  JAX accelerated · TPU v5e-16 (recommended) · GPU (8+ GB VRAM) · CPU (8+ GB RAM)

  This is a REAL quantum circuit simulation — not an analytical formula.
  Every step below manipulates 2^30 = 1,073,741,824 complex64 amplitudes.

  Circuit structure per Grover iteration:
    1. Phase oracle    : flip sign of |MARKED⟩ amplitude  (1 element out of 1B)
    2. Diffusion (Ds)  : 2|s⟩⟨s| - I  =  inversion about global mean
                         → requires 1 allreduce (psum) across devices per step

  Initialization:
    Uniform superposition  ≡  H^⊗30 |0⟩^⊗30  →  all 1B amplitudes = 1/√(2^30)

  Hardware targets
  ────────────────
  TPU v5e-16  : 16 devices × 16 GB HBM = 256 GB total  →  512 MB/chip for state
  TPU v6e-64  : 64 devices × 32 GB HBM = 2 TB total    →  128 MB/chip for state
  GPU (single): needs ≥ 8 GB VRAM  (RTX 3090 / A100)
  CPU         : needs ≥ 8 GB RAM   (slow but correct)

  Key numbers
  ───────────
  Search space  : N = 2^30 = 1,073,741,824
  State vector  : complex64 · 8 GB
  Marked state  : |111...1⟩  (index N-1)
  k_opt         : 25,735 iterations → P(success) > 0.9999
  Grover speedup: √N = 32,768×  (vs classical O(N) average)
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
config.update("jax_enable_x64", False)
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
N_QUBITS  = 30
N_TOTAL   = 1 << N_QUBITS          # 2^30 = 1,073,741,824
MARKED    = N_TOTAL - 1            # |111...1⟩

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
MEM_BYTES = N_TOTAL * 8

def theory_prob(k):
    return np.sin((2 * k + 1) * theta) ** 2

# Memory check
mem_per_chip_gb = (MEM_BYTES / NUM_DEV) / 1e9

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
def banner(msg):
    w = 76
    print("\n" + "═" * w)
    print(f"  {msg.center(w - 4)}")
    print("═" * w)

banner("Grover's Algorithm — 30-Qubit Real Statevector Simulation")
print(f"  Backend       : {BACKEND.upper()}")
print(f"  Devices       : {NUM_DEV}  ({[str(d) for d in DEVICES]})")
print(f"  Qubits        : {N_QUBITS}")
print(f"  Search space  : N = {N_TOTAL:,}")
print(f"  State vector  : {MEM_BYTES / 1e9:.2f} GB  ({N_TOTAL:,} complex64 amplitudes)")
print(f"  Per-device    : {mem_per_chip_gb:.2f} GB / chip")
print(f"  Marked state  : |{'1'*N_QUBITS}⟩  =  index {MARKED:,}")
print(f"  k_opt         : {K_OPT:,} iterations")
print(f"  P(success)    : {PROB_OPT:.10f}  (theory)")
print(f"  Grover speedup: ≈ {SPEEDUP:,.0f}×")

# ─────────────────────────────────────────────────────────────────────────────
# Oracle — phase oracle on sharded flat 1-D state vector
# MARKED = N_TOTAL - 1  →  local index N_LOCAL-1  on device NUM_DEV-1
# ─────────────────────────────────────────────────────────────────────────────
@partial(shard_map, mesh=TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)
def oracle(local_s):
    """Phase oracle: flip sign of |111...1⟩ amplitude."""
    d = jax.lax.axis_index("dev")
    is_marked_dev = (d == NUM_DEV - 1)
    sign = jnp.where(
        is_marked_dev,
        jnp.array(-1.0, dtype=jnp.complex64),
        jnp.array( 1.0, dtype=jnp.complex64),
    )
    return local_s.at[-1].multiply(sign)

# ─────────────────────────────────────────────────────────────────────────────
# Diffusion — 2|s⟩⟨s| - I  via inversion about global mean
# One psum per iteration: minimal inter-chip communication
# ─────────────────────────────────────────────────────────────────────────────
_INV_N = np.float32(1.0 / float(N_TOTAL))

@partial(shard_map, mesh=TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)
def diffusion(local_s):
    """Grover diffusion: inversion of all amplitudes about the global mean."""
    local_sum   = jnp.sum(local_s)
    global_sum  = lax.psum(local_sum, axis_name="dev")   # 1 allreduce/iter
    global_mean = global_sum * _INV_N
    return jnp.complex64(2.0) * global_mean - local_s

# ─────────────────────────────────────────────────────────────────────────────
# One Grover step  +  block of k steps via lax.fori_loop (O(1) XLA graph)
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def grover_step(state):
    state = oracle(state)
    state = diffusion(state)
    return state

@partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
def run_steps(state, k):
    """Run k Grover iterations using lax.fori_loop — O(1) XLA graph size."""
    return lax.fori_loop(0, k, lambda i, s: grover_step(s), state)

# ─────────────────────────────────────────────────────────────────────────────
# State initialization — uniform superposition via shard_map
# Each device creates its N_LOCAL slice: no 8 GB host-side allocation needed
# ─────────────────────────────────────────────────────────────────────────────
INV_SQRT_N = np.float32(1.0 / np.sqrt(float(N_TOTAL)))
N_LOCAL    = N_TOTAL // NUM_DEV

@partial(shard_map, mesh=TPU_MESH,
         in_specs=PartitionSpec("dev",), out_specs=PartitionSpec("dev",))
def _init_shard(dummy):
    return jnp.full((N_LOCAL,), INV_SQRT_N, dtype=jnp.complex64)

def init_state():
    dummy = jnp.zeros(NUM_DEV)
    return _init_shard(dummy)

# ─────────────────────────────────────────────────────────────────────────────
# JIT warm-up
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Warmup] Compiling JIT kernels (oracle + diffusion + fori_loop) ...", flush=True)
state = init_state()
t_warm = time.perf_counter()
_ = run_steps(state, 1).block_until_ready()
t_warmup = time.perf_counter() - t_warm
print(f"  [Warmup] Done  ({t_warmup:.2f}s)")

# Estimate total time
estimated_total = t_warmup * K_OPT
print(f"  [Estimate] {K_OPT:,} iters × {t_warmup*1e3:.2f}ms/iter ≈ {estimated_total:.0f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# Run full K_OPT iterations in chunks of CHUNK_SIZE.
# Collect N_SNAPSHOTS probability measurements.
# ─────────────────────────────────────────────────────────────────────────────
N_SNAPSHOTS  = 100
CHUNK_SIZE   = max(1, K_OPT // N_SNAPSHOTS)

snapshots_itr  = []
snapshots_prob = []
chunk_times    = []

# Re-init clean state
state = init_state()

print(f"\n  [Run] {K_OPT:,} iterations in {N_SNAPSHOTS} chunks of {CHUNK_SIZE:,} ...\n")
t0  = time.perf_counter()
itr = 0
chunk_id = 0

while itr < K_OPT:
    chunk = min(CHUNK_SIZE, K_OPT - itr)
    tc0   = time.perf_counter()
    state = run_steps(state, chunk)
    state.block_until_ready()
    tc    = time.perf_counter() - tc0
    itr  += chunk
    chunk_times.append(tc)

    # Probability of measuring |MARKED⟩
    p = float(jnp.abs(state[MARKED]) ** 2)
    snapshots_itr.append(itr)
    snapshots_prob.append(p)
    chunk_id += 1

    if chunk_id % 10 == 0 or itr >= K_OPT:
        throughput = chunk / tc
        print(f"    Chunk {chunk_id:>4d}/{N_SNAPSHOTS}  "
              f"iter {itr:>7,}/{K_OPT:,}  "
              f"P(|marked⟩) = {p:.6f}  "
              f"({tc*1e3:.1f}ms,  {throughput:,.0f} iter/s)")

elapsed    = time.perf_counter() - t0
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
print(f"  Grover speedup      : ≈ {SPEEDUP:,.0f}×  vs O(N) classical")
print(f"  Total sim time      : {elapsed:.2f}s")
print(f"  Throughput          : {K_OPT / elapsed:,.0f} iterations/s")
print(f"  Per-iter latency    : {elapsed * 1e3 / K_OPT:.3f} ms")
print(f"  Backend             : {BACKEND.upper()},  {NUM_DEV} device(s)")
print("─" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme plots — 4 panels
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

# Theory curve (sampled for performance)
k_sample = np.linspace(0, K_OPT, 3000, dtype=int)
p_sample = theory_prob(k_sample)

fig = plt.figure(figsize=(18, 10), dpi=150)
fig.patch.set_facecolor(P["bg"])
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                       left=0.07, right=0.97, top=0.90, bottom=0.08)

# ── Panel 1: Full probability trajectory ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(k_sample, p_sample, color=P["sub"], lw=1.0, label="Theory sin²((2k+1)θ)", alpha=0.5)
ax1.plot(snapshots_itr, snapshots_prob, "o-", color=P["a1"], ms=4, lw=2.0,
         label="Simulated P(|marked⟩)", zorder=3)
ax1.axvline(K_OPT, color=P["a2"], ls="--", lw=1.5, label=f"k_opt = {K_OPT:,}")
ax1.scatter([K_OPT], [final_prob], color=P["a2"], s=80, zorder=5)
ax1.set_xlabel("Grover Iterations k")
ax1.set_ylabel("P(success)")
ax1.set_title(f"🔍  Grover Probability Growth — {N_QUBITS} Qubits\n"
              f"(Real statevector, {N_TOTAL:,} amplitudes, 8 GB)")
ax1.set_ylim(0, 1.05)
ax1.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
theme(fig, ax1)

# ── Panel 2: Zoom around k_opt ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
zoom_lo = max(0, K_OPT - 2 * CHUNK_SIZE)
zoom_hi = K_OPT + 2 * CHUNK_SIZE
k_zoom  = np.arange(zoom_lo, min(zoom_hi, 2 * K_OPT))
ax2.plot(k_zoom, theory_prob(k_zoom), color=P["a4"], lw=2.2, label="Theory")
zoom_itr  = [i for i in snapshots_itr  if zoom_lo <= i <= zoom_hi]
zoom_prob = [p for i, p in zip(snapshots_itr, snapshots_prob) if zoom_lo <= i <= zoom_hi]
ax2.plot(zoom_itr, zoom_prob, "o", color=P["a1"], ms=6, label="Simulated")
ax2.axvline(K_OPT, color=P["a2"], ls="--", lw=1.5, label=f"k_opt = {K_OPT:,}")
ax2.set_xlabel("Grover Iterations k")
ax2.set_ylabel("P(success)")
ax2.set_title(f"🔬  Zoom near k_opt = {K_OPT:,}\n(Theory vs measured statevector)")
ax2.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
theme(fig, ax2)

# ── Panel 3: Chunk timing bar chart ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
chunk_ids = np.arange(1, len(chunk_times) + 1)
ax3.bar(chunk_ids, [t * 1e3 for t in chunk_times], color=P["a5"],
        alpha=0.8, edgecolor=P["border"], width=0.8)
mean_ct = np.mean(chunk_times) * 1e3
ax3.axhline(mean_ct, color=P["a3"], ls="--", lw=2,
            label=f"Mean = {mean_ct:.1f} ms/{CHUNK_SIZE:,} iters")
ax3.set_xlabel("Chunk Index")
ax3.set_ylabel("Wall time (ms)")
ax3.set_title(f"⏱  Chunk Timing  ({CHUNK_SIZE:,} iters/chunk)\n"
              f"Total: {elapsed:.1f}s  |  {K_OPT/elapsed:.0f} iters/s")
ax3.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
theme(fig, ax3)

# ── Panel 4: Final measurement bar ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
labels = [f"|{'0'*N_QUBITS}⟩", f"|{'1'*N_QUBITS}⟩"]
values = [1.0 - final_prob, final_prob]
bars = ax4.bar(labels, values, color=[P["sub"], P["a2"]], width=0.5,
               edgecolor=P["border"])
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             val + 0.02, f"{val*100:.6f}%", ha="center",
             color=P["text"], fontsize=10, fontweight="bold")
ax4.set_ylim(0, 1.12)
ax4.set_title(f"📊  Final Measurement Probabilities\n"
              f"After {K_OPT:,} iterations  |  P(success) = {final_prob:.8f}")
ax4.set_ylabel("Probability")
theme(fig, ax4)

fig.suptitle(
    f"Grover's Algorithm — 30-Qubit Full Statevector Simulation  |  JAX {jax.__version__}  |  {BACKEND.upper()}\n"
    f"N = 2^30 = {N_TOTAL:,}  ·  8 GB state  ·  k_opt = {K_OPT:,}  ·  "
    f"P(success) = {final_prob:.8f}  ·  {elapsed:.1f}s  ·  {TS}",
    color=P["text"], fontsize=10, fontweight="bold", y=0.97,
)

plot_path = f"grover_simulation/plots/grover_30q_{TS}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close()
print(f"\n  📈 Plot saved  → {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Save JSON
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "meta": {
        "timestamp": TS, "backend": BACKEND, "n_devices": NUM_DEV,
        "jax_version": jax.__version__, "script": "grover_simulation/30qubits.py",
        "simulation_type": "real_statevector_jax",
    },
    "circuit": {
        "n_qubits": N_QUBITS, "N_total": N_TOTAL,
        "marked_state": MARKED, "state_vector_gb": MEM_BYTES / 1e9,
        "mem_per_device_gb": mem_per_chip_gb,
    },
    "theory": {
        "k_opt": K_OPT, "prob_opt_theory": PROB_OPT,
        "grover_speedup": float(SPEEDUP),
    },
    "simulation": {
        "k_opt_run": K_OPT, "final_prob_measured": final_prob,
        "elapsed_s": elapsed, "ms_per_iter": elapsed * 1e3 / K_OPT,
        "iters_per_sec": K_OPT / elapsed,
        "snapshots_itr": snapshots_itr, "snapshots_prob": snapshots_prob,
        "chunk_times_ms": [t * 1e3 for t in chunk_times],
    },
}
json_path = f"grover_simulation/results/grover_30q_{TS}.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  📄 JSON saved  → {json_path}")
print()