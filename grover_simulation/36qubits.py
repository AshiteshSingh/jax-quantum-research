#!/usr/bin/env python3
"""
================================================================================
  Grover's Algorithm — 36-Qubit Full State-Vector Simulation
  Google Cloud TPU v6e-64  (64 chips × 32 GB HBM = 2 TB aggregate)

  This is a REAL quantum circuit simulation — not an analytical formula.
  Every iteration below manipulates 2^36 = 68,719,476,736 complex64 amplitudes
  distributed across 64 TPU v6e chips (~8.59 GB per chip).

  XLA environment
  ───────────────
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

  Circuit structure per Grover iteration
  ───────────────────────────────────────
  1. Phase oracle: flip sign of |MARKED⟩ amplitude
       oracle(|ψ⟩)   →  (-1)^{f(x)} αₓ  for all x,  f(x) = [x == MARKED]
       Implementation: local_state.at[-1].multiply(-1)  on device NUM_DEV-1 only
       Communication: NONE (purely local operation)

  2. Diffusion: 2|s⟩⟨s| - I  =  inversion about mean
       D|ψ⟩ᵢ  =  2·mean(α)  −  αᵢ
       Implementation: psum(local_sum) → global_mean → 2·mean − local_state
       Communication: 1 × allreduce per iteration (psum over 64 chips)

  Key numbers
  ───────────
  Search space  : N  = 2^36 = 68,719,476,736  (≈ 6.87 × 10^10)
  State vector  : complex64 · 512 GB  (549.76 GB including padding)
  Per-chip      : 2^30 amplitudes · 8 GB / chip  (on 64-chip TPU)
  Marked state  : |111...1⟩  (index N − 1)
  k_opt (full)  : 205,887 iterations  →  P(success) > 0.9999
  This run      : K_RUN iterations (practical subset, see K_RUN below)
  Grover speedup: √N ≈ 262,144×  vs classical O(N) search

  Why K_RUN < K_OPT?
  ───────────────────
  Running all 205,887 iterations on a 512 GB state vector — even on TPU v6e-64 —
  takes several hours (diffusion requires a global allreduce per iteration).
  K_RUN demonstrates correct probability growth and validates the implementation.
  Full K_OPT execution is possible by setting K_RUN = K_OPT; use a TPU job queue.
================================================================================
"""

import os, sys, time, json
from datetime import datetime

# Force XLA to preallocate HBM to prevent runtime fragmentation on large states
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
from jax import config
config.update("jax_enable_x64", False)    # keep complex64
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
N_QUBITS = 36
N_TOTAL  = 1 << N_QUBITS      # 2^36 = 68,719,476,736
MARKED   = N_TOTAL - 1         # |111...1⟩

# How many Grover iterations to actually execute in this run.
# Set K_RUN = K_OPT to run the full algorithm (takes hours on TPU v6e-64).
# Default: 2,000  →  shows first ~1% of the probability growth curve.
K_RUN = 2_000

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
MEM_BYTES     = N_TOTAL * 8          # 512 GB
MEM_PER_CHIP  = MEM_BYTES / NUM_DEV  # bytes per device

def theory_prob(k):
    """Analytical success probability after k Grover iterations."""
    return np.sin((2 * np.asarray(k) + 1) * theta) ** 2

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
def banner(msg):
    w = 78
    print("\n" + "═" * w)
    print(f"  {msg.center(w - 4)}")
    print("═" * w)

banner("Grover's Algorithm — 36-Qubit Real Statevector Simulation")
print(f"  Backend       : {BACKEND.upper()}")
print(f"  Devices       : {NUM_DEV}  ({[str(d) for d in DEVICES]})")
print(f"  Qubits        : {N_QUBITS}")
print(f"  Search space  : N = 2^36 = {N_TOTAL:,}  (≈ 6.87 × 10¹⁰ states)")
print(f"  State vector  : {MEM_BYTES / 1e9:.2f} GB  ({N_TOTAL:,} complex64 amplitudes)")
print(f"  Per-device    : {MEM_PER_CHIP / 1e9:.2f} GB / chip")
print(f"  Marked state  : |{'1'*N_QUBITS}⟩  =  index {MARKED:,}")
print(f"  k_opt (full)  : {K_OPT:,} iterations")
print(f"  K_RUN (this)  : {K_RUN:,} iterations  ({K_RUN/K_OPT*100:.2f}% of k_opt)")
print(f"  P_theory(k_opt): {PROB_OPT:.10f}")
print(f"  Grover speedup: ≈ {SPEEDUP:,.0f}×  vs classical O(N) brute-force")

# ─────────────────────────────────────────────────────────────────────────────
# Hardware requirement check
# ─────────────────────────────────────────────────────────────────────────────
if MEM_PER_CHIP / 1e9 > 16.0 and BACKEND != "tpu":
    print(f"\n  ⚠️  WARNING: {MEM_PER_CHIP/1e9:.1f} GB per device required.")
    print(f"     Running on {BACKEND.upper()} with {NUM_DEV} device(s).")
    print(f"     Full 36-qubit simulation requires TPU v6e-64 (64 × 32 GB HBM).")
    print(f"     Proceeding — will OOM if insufficient device memory.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Oracle — phase oracle on sharded flat 1-D state vector
#
# MARKED = 2^36 - 1  is stored at:
#   device  : NUM_DEV - 1  (last shard)
#   local idx: N_LOCAL - 1  (last element of that shard)
#
# Communication cost: NONE — purely local write on one chip.
# ─────────────────────────────────────────────────────────────────────────────
@partial(shard_map, mesh=TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)
def oracle(local_s):
    """
    Phase oracle: flip sign of |111...1⟩ amplitude.
    Only the last element of the last shard is modified.
    All other elements are unchanged (multiply by +1).
    """
    d = jax.lax.axis_index("dev")
    is_marked_dev = (d == NUM_DEV - 1)
    sign = jnp.where(
        is_marked_dev,
        jnp.array(-1.0, dtype=jnp.complex64),
        jnp.array( 1.0, dtype=jnp.complex64),
    )
    return local_s.at[-1].multiply(sign)

# ─────────────────────────────────────────────────────────────────────────────
# Diffusion — 2|s⟩⟨s| - I  =  inversion about the global mean amplitude
#
# For uniform |s⟩: D|ψ⟩ᵢ = 2·(1/N Σⱼ αⱼ) - αᵢ = 2·mean(α) - αᵢ
#
# Implementation on sharded state:
#   1. Each chip sums its N_LOCAL amplitudes   →  local_sum
#   2. psum collects global_sum from all chips →  ONE allreduce per iteration
#   3. global_mean = global_sum / N_TOTAL
#   4. Each chip applies: 2·mean - local_state  (purely local, no more comm)
#
# Communication: 1 × allreduce (psum) per Grover iteration  — O(N_TOTAL bytes)
# This is the dominant communication cost of the 36-qubit simulation.
# ─────────────────────────────────────────────────────────────────────────────
_INV_N = np.float32(1.0 / float(N_TOTAL))

@partial(shard_map, mesh=TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)
def diffusion(local_s):
    """
    Grover diffusion operator: inversion about the global mean.
    Uses a single psum to compute the mean across all TPU chips.
    """
    local_sum   = jnp.sum(local_s)                        # local reduction
    global_sum  = lax.psum(local_sum, axis_name="dev")    # global allreduce
    global_mean = global_sum * _INV_N                     # scalar broadcast
    return jnp.complex64(2.0) * global_mean - local_s     # local update

# ─────────────────────────────────────────────────────────────────────────────
# One Grover step: oracle → diffusion
# Compiled once by XLA; runs as two XLA ops (one with no comm, one with psum)
# ─────────────────────────────────────────────────────────────────────────────
@jax.jit
def grover_step(state):
    state = oracle(state)
    state = diffusion(state)
    return state

# ─────────────────────────────────────────────────────────────────────────────
# Block of k steps via lax.fori_loop → O(1) XLA graph size regardless of k
#
# Critical for 36-qubit scale:
#   Without fori_loop:  XLA graph has K_OPT = 205,887 oracle+diffusion nodes
#                       → compiler OOM / multi-hour compile time
#   With fori_loop:     XLA graph has 1 oracle + 1 diffusion (body compiled once)
#                       → seconds to compile, then runs as XLA While loop
# ─────────────────────────────────────────────────────────────────────────────
@partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
def run_steps(state, k):
    """Run k Grover iterations via lax.fori_loop — O(1) XLA graph, any k."""
    return lax.fori_loop(0, k, lambda i, s: grover_step(s), state)

# ─────────────────────────────────────────────────────────────────────────────
# State initialization — uniform superposition in shards
#
# Uniform superposition = H^⊗36|0⟩^⊗36: each amplitude = 1/√(2^36)
# We CANNOT call jnp.full(2^36, ...) — that would require 512 GB on the host.
# Instead, each chip creates its own N_LOCAL = 2^30 element slice directly.
# ─────────────────────────────────────────────────────────────────────────────
INV_SQRT_N = np.float32(1.0 / np.sqrt(float(N_TOTAL)))
N_LOCAL    = N_TOTAL // NUM_DEV      # 2^36 / 64 = 2^30 elements per chip

@partial(shard_map, mesh=TPU_MESH,
         in_specs=PartitionSpec("dev",), out_specs=PartitionSpec("dev",))
def _init_shard(dummy):
    """Create uniform superposition slice on each device."""
    return jnp.full((N_LOCAL,), INV_SQRT_N, dtype=jnp.complex64)

def init_state():
    dummy = jnp.zeros(NUM_DEV)
    return _init_shard(dummy)

# ─────────────────────────────────────────────────────────────────────────────
# Probability extraction — P(|MARKED⟩) from sharded state
#
# |MARKED⟩ = index N-1 = last element on device NUM_DEV-1 (local index N_LOCAL-1)
# Using shard_map + psum avoids host-side array transfers.
# ─────────────────────────────────────────────────────────────────────────────
@partial(shard_map, mesh=TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)
def _extract_marked_prob(local_s):
    d    = jax.lax.axis_index("dev")
    p    = jnp.where(d == NUM_DEV - 1,
                     jnp.abs(local_s[-1]) ** 2,
                     jnp.float32(0.0))
    # psum broadcasts the result to all devices
    p_global = lax.psum(p, axis_name="dev")
    return jnp.full_like(local_s, p_global.real)

def get_marked_prob(state):
    """Return P(|MARKED⟩) as a Python float."""
    result = _extract_marked_prob(state)
    return float(result[0])   # all elements have the same replicated value

# ─────────────────────────────────────────────────────────────────────────────
# JIT warm-up — triggers XLA compilation for all kernels
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Warmup] Allocating 512 GB state + compiling JIT kernels ...", flush=True)
state = init_state()
state.block_until_ready()
print(f"  [Warmup] State initialized  ({N_TOTAL:,} amplitudes  ≈ {MEM_BYTES/1e9:.1f} GB)")

t_warm = time.perf_counter()
_ = run_steps(state, 1).block_until_ready()
t_warmup = time.perf_counter() - t_warm
print(f"  [Warmup] JIT compile done  ({t_warmup:.2f}s)")

# Latency estimate for full run
estimated_per_iter_ms = t_warmup * 1e3
estimated_k_run_s     = estimated_per_iter_ms / 1e3 * K_RUN
estimated_k_opt_h     = estimated_per_iter_ms / 1e3 * K_OPT / 3600
print(f"  [Estimate] ~{estimated_per_iter_ms:.1f}ms/iter  →  "
      f"K_RUN={K_RUN:,} iters ≈ {estimated_k_run_s:.0f}s  |  "
      f"K_OPT={K_OPT:,} iters ≈ {estimated_k_opt_h:.1f}h")

# ─────────────────────────────────────────────────────────────────────────────
# Main simulation — K_RUN iterations in chunks, collect 20 snapshots
# ─────────────────────────────────────────────────────────────────────────────
N_SNAPSHOTS = 20
CHUNK_SIZE  = max(1, K_RUN // N_SNAPSHOTS)

snapshots_itr  = []
snapshots_prob = []
chunk_times    = []

# Re-init clean state
state = init_state()

print(f"\n  [Run] {K_RUN:,} Grover iterations in chunks of {CHUNK_SIZE:,} ...\n")
t0  = time.perf_counter()
itr = 0

while itr < K_RUN:
    chunk = min(CHUNK_SIZE, K_RUN - itr)
    tc0   = time.perf_counter()
    state = run_steps(state, chunk)
    state.block_until_ready()
    tc    = time.perf_counter() - tc0
    itr  += chunk
    chunk_times.append(tc)

    p = get_marked_prob(state)
    snapshots_itr.append(itr)
    snapshots_prob.append(p)

    throughput = chunk / tc
    print(f"    itr {itr:>6,}/{K_RUN:,}  "
          f"P(|marked⟩) = {p:.8f}  "
          f"({tc*1e3:.0f}ms,  {throughput:.0f} iter/s)")

elapsed    = time.perf_counter() - t0
final_prob = snapshots_prob[-1]

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(f"  Qubits              : {N_QUBITS}")
print(f"  Search space        : {N_TOTAL:,}  (2^36)")
print(f"  State vector        : {MEM_BYTES/1e9:.2f} GB  ({MEM_PER_CHIP/1e9:.2f} GB/chip)")
print(f"  Marked state        : |{'1'*N_QUBITS}⟩  (index {MARKED:,})")
print(f"  k_opt (full)        : {K_OPT:,}")
print(f"  K_RUN (this run)    : {K_RUN:,}  ({K_RUN/K_OPT*100:.2f}% of k_opt)")
print(f"  P_theory @ K_RUN    : {theory_prob(K_RUN):.8f}")
print(f"  P_measured @ K_RUN  : {final_prob:.8f}")
print(f"  P_theory @ k_opt    : {PROB_OPT:.10f}")
print(f"  Simulation time     : {elapsed:.2f}s")
print(f"  Throughput          : {K_RUN/elapsed:.1f} iterations/s")
print(f"  Per-iter latency    : {elapsed*1e3/K_RUN:.2f} ms")
print(f"  Est. full k_opt time: {elapsed/K_RUN*K_OPT/3600:.1f} h")
print(f"  Backend             : {BACKEND.upper()},  {NUM_DEV} device(s)")
print(f"  Grover speedup      : ≈ {SPEEDUP:,.0f}×  vs classical O(N)")
print("─" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme plots — 4 panels
# ─────────────────────────────────────────────────────────────────────────────
P_COL = {
    "bg": "#0d1117", "panel": "#161b22", "border": "#30363d",
    "text": "#e6edf3", "sub": "#8b949e", "grid": "#21262d",
    "a1": "#58a6ff", "a2": "#3fb950", "a3": "#f78166",
    "a4": "#d2a8ff", "a5": "#ffa657",
}

def theme(fig, axes_list):
    fig.patch.set_facecolor(P_COL["bg"])
    for ax in (axes_list if hasattr(axes_list, "__iter__") else [axes_list]):
        ax.set_facecolor(P_COL["panel"])
        ax.tick_params(colors=P_COL["text"], labelsize=9)
        ax.xaxis.label.set_color(P_COL["text"])
        ax.yaxis.label.set_color(P_COL["text"])
        ax.title.set_color(P_COL["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P_COL["border"])
        ax.grid(True, color=P_COL["grid"], ls="--", alpha=0.5, lw=0.6)

# Theory curves
k_full_sample = np.linspace(0, K_OPT, 5000, dtype=int)
p_full_sample = theory_prob(k_full_sample)
k_run_sample  = np.linspace(0, K_RUN, 500, dtype=int)
p_run_sample  = theory_prob(k_run_sample)

fig = plt.figure(figsize=(20, 11), dpi=150)
fig.patch.set_facecolor(P_COL["bg"])
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                       left=0.07, right=0.97, top=0.89, bottom=0.08)

# ── Panel 1: Partial run vs full theory ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
# Full theory (dashed, full range)
ax1.plot(k_full_sample, p_full_sample, color=P_COL["sub"], lw=1.0,
         ls="--", label=f"Theory (full, k_opt={K_OPT:,})", alpha=0.5)
# Simulated region
ax1.plot(k_run_sample, p_run_sample, color=P_COL["a4"], lw=1.5,
         label=f"Theory (K_RUN={K_RUN:,})")
ax1.plot(snapshots_itr, snapshots_prob, "o", color=P_COL["a1"], ms=6, zorder=5,
         label="Simulated P(|marked⟩)")
ax1.axvline(K_RUN, color=P_COL["a5"], ls=":", lw=2, label=f"K_RUN = {K_RUN:,}")
ax1.axvline(K_OPT, color=P_COL["a2"], ls="--", lw=1.5, label=f"k_opt = {K_OPT:,}", alpha=0.6)
ax1.set_xlabel("Grover Iterations k")
ax1.set_ylabel("P(|marked⟩)")
ax1.set_title(f"🔍  Grover Probability Growth — {N_QUBITS} Qubits\n"
              f"(2^36 = {N_TOTAL:,} amplitudes · 512 GB state · TPU v6e-64)")
ax1.set_ylim(0, max(0.05, max(snapshots_prob) * 1.5))
ax1.legend(facecolor=P_COL["panel"], edgecolor=P_COL["border"],
           labelcolor=P_COL["text"], fontsize=7)
theme(fig, ax1)

# ── Panel 2: Probability measured vs theory (scatter) ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
theory_at_snaps = [float(theory_prob(k)) for k in snapshots_itr]
ax2.scatter(theory_at_snaps, snapshots_prob, color=P_COL["a1"], s=40, zorder=3,
            label="Simulation points")
diag_max = max(max(theory_at_snaps), max(snapshots_prob)) * 1.05
ax2.plot([0, diag_max], [0, diag_max], color=P_COL["a2"], ls="--", lw=1.5,
         label="Perfect agreement")
resid = np.mean(np.abs(np.array(snapshots_prob) - np.array(theory_at_snaps)))
ax2.set_xlabel("Theory P(success)")
ax2.set_ylabel("Simulated P(success)")
ax2.set_title(f"🎯  Theory vs Simulation Agreement\n"
              f"Mean |Δ| = {resid:.2e}  (numerical precision)")
ax2.legend(facecolor=P_COL["panel"], edgecolor=P_COL["border"],
           labelcolor=P_COL["text"], fontsize=8)
theme(fig, ax2)

# ── Panel 3: Chunk timing (per-step latency) ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
chunk_ids = np.arange(1, len(chunk_times) + 1)
ax3.bar(chunk_ids, [t * 1e3 for t in chunk_times], color=P_COL["a5"],
        alpha=0.85, edgecolor=P_COL["border"])
mean_ms = np.mean(chunk_times) * 1e3
ax3.axhline(mean_ms, color=P_COL["a3"], ls="--", lw=2,
            label=f"Mean = {mean_ms:.1f} ms/{CHUNK_SIZE} iters")
ax3.set_xlabel("Chunk Index")
ax3.set_ylabel("Wall time (ms)")
ax3.set_title(f"⏱  Per-chunk Timing  ({CHUNK_SIZE} iters/chunk)\n"
              f"Total: {elapsed:.1f}s  ·  {K_RUN/elapsed:.0f} iter/s  ·  "
              f"Est. full: {elapsed/K_RUN*K_OPT/3600:.0f}h")
ax3.legend(facecolor=P_COL["panel"], edgecolor=P_COL["border"],
           labelcolor=P_COL["text"], fontsize=8)
theme(fig, ax3)

# ── Panel 4: Memory & sharding layout ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
n_show = min(NUM_DEV, 16)    # show at most 16 chips
dev_labels = [f"Dev {i}" for i in range(n_show)]
mem_per_dev_gb = MEM_PER_CHIP / 1e9
bars = ax4.bar(dev_labels, [mem_per_dev_gb] * n_show,
               color=P_COL["a4"], edgecolor=P_COL["border"], alpha=0.85)
# Highlight last device (holds MARKED state)
bars[-1].set_color(P_COL["a2"])
bars[-1].set_alpha(1.0)
ax4.axhline(mem_per_dev_gb, color=P_COL["a3"], ls="--", lw=1.5,
            label=f"{mem_per_dev_gb:.2f} GB/chip")
ax4.set_xlabel(f"Device  (showing first {n_show} of {NUM_DEV})")
ax4.set_ylabel("HBM usage (GB)")
ax4.set_title(f"🖥️  State Sharding Layout\n"
              f"{NUM_DEV} × {mem_per_dev_gb:.2f} GB = {MEM_BYTES/1e9:.0f} GB total  |  "
              f"Green = holds |marked⟩")
ax4.tick_params(axis='x', rotation=45, labelsize=7)
ax4.legend(facecolor=P_COL["panel"], edgecolor=P_COL["border"],
           labelcolor=P_COL["text"], fontsize=8)
theme(fig, ax4)

fig.suptitle(
    f"Grover's Algorithm — 36-Qubit Full Statevector Simulation  |  JAX {jax.__version__}  |  {BACKEND.upper()}  |  {NUM_DEV} chips\n"
    f"N = 2^36 = {N_TOTAL:,}  ·  512 GB state  ·  K_RUN = {K_RUN:,} / k_opt = {K_OPT:,}  ·  "
    f"P(measured) = {final_prob:.6f}  ·  {elapsed:.1f}s  ·  {TS}",
    color=P_COL["text"], fontsize=9.5, fontweight="bold", y=0.97,
)

plot_path = f"grover_simulation/plots/grover_36q_{TS}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=P_COL["bg"])
plt.close()
print(f"\n  📈 Plot saved  → {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Save JSON results
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "meta": {
        "timestamp": TS, "backend": BACKEND, "n_devices": NUM_DEV,
        "devices": [str(d) for d in DEVICES],
        "jax_version": jax.__version__,
        "script": "grover_simulation/36qubits.py",
        "simulation_type": "real_statevector_jax_tpu",
        "hardware_target": "TPU v6e-64 (64 chips × 32 GB HBM)",
    },
    "circuit": {
        "n_qubits": N_QUBITS, "N_total": N_TOTAL,
        "marked_state": MARKED, "marked_bitstring": "1" * N_QUBITS,
        "state_vector_gb": MEM_BYTES / 1e9,
        "state_per_device_gb": MEM_PER_CHIP / 1e9,
    },
    "theory": {
        "theta_rad": float(theta),
        "k_opt": K_OPT,
        "prob_opt_theory": PROB_OPT,
        "grover_speedup": float(SPEEDUP),
    },
    "simulation": {
        "K_RUN": K_RUN,
        "fraction_of_k_opt": K_RUN / K_OPT,
        "final_prob_measured": final_prob,
        "theory_at_K_RUN": float(theory_prob(K_RUN)),
        "elapsed_s": elapsed,
        "ms_per_iter": elapsed * 1e3 / K_RUN,
        "iters_per_sec": K_RUN / elapsed,
        "estimated_k_opt_hours": elapsed / K_RUN * K_OPT / 3600,
        "snapshots_itr": snapshots_itr,
        "snapshots_prob": snapshots_prob,
        "chunk_times_ms": [t * 1e3 for t in chunk_times],
    },
}
json_path = f"grover_simulation/results/grover_36q_{TS}.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  📄 JSON saved  → {json_path}")
print()
print("  To run full k_opt = {:,} iterations, set K_RUN = K_OPT at line ~70.".format(K_OPT))
print("  Estimated time on TPU v6e-64: {:.1f} hours.".format(
    elapsed / K_RUN * K_OPT / 3600))
print()