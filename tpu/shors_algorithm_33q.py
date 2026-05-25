#!/usr/bin/env python3
"""
================================================================================
  Shor's Algorithm — 33-Qubit Full State Vector Simulation
  Google Cloud TPU v5litepod-16  (16 chips × 16 GB HBM = 256 GB total)

  Circuit layout   : 22 counting qubits + 11 work qubits = 33 qubits total
  State vector     : complex64, shape (2^33,) = 8,589,934,592 elements = 64 GB
  Sharding         : PositionalSharding across all 16 TPU chips (~4 GB / chip)
  Factoring targets: N=15 (a=7), N=21 (a=2), N=35 (a=2)

  Outputs (all saved to tpu/plots/ and tpu/results/):
    • shors_33q_<ts>.png          — 12-panel master plot
    • shors_spectrum_<N>_<ts>.png — per-run probability spectrum (high-res)
    • shors_qft_<N>_<ts>.png      — QFT phase analysis
    • shors_circuit_<N>_<ts>.png  — circuit analytics + timing
    • shors_summary_<ts>.png      — final factoring summary dashboard
    • shors_33q_<ts>.json         — full JSON results
    • shors_33q_<ts>.txt          — full console log (Tee)

  Implementation strategy
  ───────────────────────
  Because 33 tensordot dimensions (shape (2,)*33) exceed XLA limits and OOM
  the compiler, we use the FLAT 1-D state-vector approach already validated
  in run_tpu_benchmark() (tpu_quantum_scale.py, n=10-34).

  Key operations work by bit-index arithmetic on the amplitude array:
    • hadamard_flat     — XOR partner index, butterfly combine
    • phase_flat        — multiply amplitudes at |1⟩ positions by e^(iθ)
    • controlled_phase  — multiply at |11⟩ positions by e^(iθ)
    • qft_flat          — O(n²) controlled-phase + Hadamard + reversal
    • ctrl_mod_mul      — permute work-register indices by (x → a·x mod N)

  All gate functions are @jax.jit compiled and shard-aware.
================================================================================
"""

import os, sys, time, math, json, csv, warnings
from datetime import datetime
from math import gcd

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.lax as lax
from jax.sharding import PositionalSharding

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Casting complex")

# ─────────────────────────────────────────────────────────────────────────────
# Timestamp & output dirs
# ─────────────────────────────────────────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("tpu/results", exist_ok=True)
os.makedirs("tpu/plots",   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TPU device setup
# ─────────────────────────────────────────────────────────────────────────────
BACKEND  = jax.default_backend()
DEVICES  = jax.devices()
NUM_DEV  = len(DEVICES)
SHARDING = PositionalSharding(DEVICES).reshape(NUM_DEV, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme palette (project-wide style)
# ─────────────────────────────────────────────────────────────────────────────
P = {
    "bg":     "#0d1117", "panel":  "#161b22", "border": "#30363d",
    "text":   "#e6edf3", "sub":    "#8b949e", "grid":   "#21262d",
    "a1":     "#58a6ff", "a2":     "#3fb950", "a3":     "#f78166",
    "a4":     "#d2a8ff", "a5":     "#ffa657", "a6":     "#79c0ff",
    "a7":     "#ff7b72", "a8":     "#56d364",
}

CMAP_HEAT   = "plasma"
CMAP_PHASE  = "hsv"
CMAP_PROB   = "Blues"

def theme(fig, axes):
    fig.patch.set_facecolor(P["bg"])
    lst = axes if hasattr(axes, "__iter__") else [axes]
    for ax in lst:
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["text"], labelsize=9)
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["border"])
        ax.grid(True, color=P["grid"], ls="--", alpha=0.45, lw=0.6)

def banner(title):
    w = 78
    print("\n" + "═" * w)
    print(f" {title.center(w - 2)} ")
    print("═" * w)

def fmt_bytes(b):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024: return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"

# ─────────────────────────────────────────────────────────────────────────────
# Classical number-theory helpers
# ─────────────────────────────────────────────────────────────────────────────

def mod_pow(base: int, exp: int, mod: int) -> int:
    result = 1; base %= mod
    while exp > 0:
        if exp & 1: result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result

def classical_order(a: int, N: int) -> int:
    r, val = 1, a % N
    while val != 1:
        val = val * a % N
        r += 1
        if r > N * N: return -1
    return r

def continued_fraction_convergents(numerator: int, denominator: int, limit: int):
    convergents = []
    n0, d0 = 0, 1
    n1, d1 = 1, 0
    num, den = numerator, denominator
    while den:
        q = num // den
        num, den = den, num - q * den
        n0, n1 = n1, q * n1 + n0
        d0, d1 = d1, q * d1 + d0
        convergents.append((n1, d1))
        if d1 >= limit:
            break
    return convergents

def extract_period(measurement: int, n_counting: int, N: int, a: int):
    if measurement == 0: return None
    denom = 1 << n_counting
    convergents = continued_fraction_convergents(measurement, denom, N)
    for _, r in convergents:
        if r > 0 and mod_pow(a, r, N) == 1:
            return r
    return None

def try_factor(a: int, r: int, N: int):
    if r is None or r % 2 != 0: return None
    half = mod_pow(a, r // 2, N)
    if half == N - 1: return None
    for f in (gcd(half + 1, N), gcd(half - 1, N)):
        if 1 < f < N: return (f, N // f)
    return None

# ─────────────────────────────────────────────────────────────────────────────
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

# Flat 1-D state-vector gate primitives  (JAX JIT, shard-aware)
# ─────────────────────────────────────────────────────────────────────────────

H_MAT = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2.0)

# Define the 1D device mesh to give the TPU network explicit dimensions
TPU_MESH = Mesh(np.array(DEVICES), ('dev',))
P_SPEC = PartitionSpec('dev', None)

@partial(jax.jit, static_argnums=(1, 2))
def _hadamard_single(state, q, n_counting):
    # 4 Global Qubits (0,1,2,3) route across the 16 TPU chips. 
    # Qubits 4+ live safely inside the local 16GB HBM.
    if q >= 4:
        # --- LOCAL QUBIT ---
        @jax.jit
        def local_h(local_s):
            dim_w = local_s.shape[1]
            q_loc = q - 4
            shape1 = 1 << q_loc
            shape2 = 1 << (n_counting - 4 - 1 - q_loc)
            
            s = local_s.reshape((shape1, 2, shape2, dim_w))
            s = jnp.tensordot(H_MAT, s, axes=([1], [1]))
            s = jnp.moveaxis(s, 0, 1)
            return s.reshape(local_s.shape)
            
        return shard_map(local_h, TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)(state)
        
    else:
        # --- GLOBAL QUBIT (Cross-Chip Butterfly Exchange) ---
        @jax.jit
        def global_h(local_s):
            d = jax.lax.axis_index('dev')
            bit_pos = 3 - q
            bit_val = (d >> bit_pos) & 1
            
            # Pair up with partner TPU chip over the ICI
            perm = [(i, i ^ (1 << bit_pos)) for i in range(16)]
            partner_s = jax.lax.ppermute(local_s, axis_name='dev', perm=perm)
            
            # Apply H = [1 1; 1 -1] / sqrt(2) directly using paired memory
            new_s = jnp.where(bit_val == 0,
                              local_s + partner_s,
                              partner_s - local_s) / jnp.sqrt(2.0)
            return new_s.astype(jnp.complex64)

        return shard_map(global_h, TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)(state)

def hadamard_flat(state, q, n_counting):
    return _hadamard_single(state, q, n_counting)


@partial(jax.jit, static_argnums=(1, 2, 3))
def _ctrl_phase_single(state, ctrl, tgt, n_counting, cos_t, sin_t):
    dim_c = 1 << n_counting
    idx_c = jnp.arange(dim_c, dtype=jnp.int32)
    idx_c = lax.with_sharding_constraint(idx_c, SHARDING.reshape(NUM_DEV))
    
    bit_c = (idx_c >> (n_counting - 1 - ctrl)) & 1
    bit_t = (idx_c >> (n_counting - 1 - tgt )) & 1
    phase = jnp.where(
        (bit_c == 1) & (bit_t == 1),
        cos_t + 1j * sin_t,
        jnp.complex64(1.0),
    )
    return state * phase[:, None].astype(jnp.complex64)

def ctrl_phase_flat(state, ctrl, tgt, n_counting, theta):
    cos_t = jnp.float32(float(np.cos(theta)))
    sin_t = jnp.float32(float(np.sin(theta)))
    return _ctrl_phase_single(state, ctrl, tgt, n_counting, cos_t, sin_t)


@partial(jax.jit, static_argnums=(1, 2, 3))
def _swap_single(state, q1, q2, n_counting):
    if q1 == q2:
        return state
        
    q_min = min(q1, q2)
    q_max = max(q1, q2)
    
    @jax.jit
    def local_swap(local_s):
        qm, qM = q_min - 4, q_max - 4
        dim_w = local_s.shape[1]
        shape1 = 1 << qm
        shape2 = 1 << (qM - qm - 1)
        shape3 = 1 << (n_counting - 4 - 1 - qM)
        
        s = local_s.reshape((shape1, 2, shape2, 2, shape3, dim_w))
        s = jnp.swapaxes(s, 1, 3)
        return s.reshape(local_s.shape)

    @jax.jit
    def global_swap(local_s):
        d = jax.lax.axis_index('dev')
        bit1 = (d >> (3 - q_min)) & 1
        bit2 = (d >> (3 - q_max)) & 1
        
        partner = (1 << (3 - q_min)) ^ (1 << (3 - q_max))
        perm = [(i, i ^ partner) for i in range(16)]
        
        partner_s = jax.lax.ppermute(local_s, axis_name='dev', perm=perm)
        return jnp.where(bit1 != bit2, partner_s, local_s)

    @jax.jit
    def cross_swap(local_s):
        # Swaps a chip-level network routing bit with a local memory bit
        d = jax.lax.axis_index('dev')
        bit_g = (d >> (3 - q_min)) & 1
        
        perm = [(i, i ^ (1 << (3 - q_min))) for i in range(16)]
        partner_s = jax.lax.ppermute(local_s, axis_name='dev', perm=perm)
        
        q_loc = q_max - 4
        dim_w = local_s.shape[1]
        shape1 = 1 << q_loc
        shape2 = 1 << (n_counting - 4 - 1 - q_loc)
        
        s = local_s.reshape((shape1, 2, shape2, dim_w))
        partner_s = partner_s.reshape((shape1, 2, shape2, dim_w))
        
        # Mask and swap the halves of the local array dynamically
        local_bit_idx = jnp.array([0, 1]).reshape((1, 2, 1, 1))
        keep_mask = (local_bit_idx == bit_g)
        partner_slice = jnp.take(partner_s, bit_g, axis=1, keepdims=True)
        
        new_s = jnp.where(keep_mask, s, partner_slice)
        return new_s.reshape(local_s.shape)

    if q_min >= 4:
        return shard_map(local_swap, TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)(state)
    elif q_max < 4:
        return shard_map(global_swap, TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)(state)
    else:
        return shard_map(cross_swap, TPU_MESH, in_specs=P_SPEC, out_specs=P_SPEC)(state)

def swap_flat(state, q1, q2, n_counting):
    return _swap_single(state, q1, q2, n_counting)


def inverse_qft_flat(state, qubits, n_counting):
    k = len(qubits)
    for i in range(k // 2):
        state = swap_flat(state, qubits[i], qubits[k - 1 - i], n_counting)
    for i in range(k - 1, -1, -1):
        q = qubits[i]
        for j in range(k - 1, i, -1):
            theta = -2.0 * np.pi / (1 << (j - i + 1))
            state = ctrl_phase_flat(state, qubits[j], q, n_counting, theta)
        state = hadamard_flat(state, q, n_counting)
    return state

# ─────────────────────────────────────────────────────────────────────────────
# Controlled Modular Multiplication
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(1, 3, 4))
def _ctrl_mod_mul_jit(state, ctrl_qubit, perm_inv_jax, N, n_counting):
    dim_c = 1 << n_counting
    idx_c = jnp.arange(dim_c, dtype=jnp.int32)
    idx_c = lax.with_sharding_constraint(idx_c, SHARDING.reshape(NUM_DEV))
    
    ctrl_bit = (idx_c >> (n_counting - 1 - ctrl_qubit)) & 1
    
    permuted_state = state[:, perm_inv_jax]
    return jnp.where(ctrl_bit[:, None] == 1, permuted_state, state)

def ctrl_mod_mul_flat(state, ctrl_qubit, a_val, N, work_qubits, n_counting):
    """
    Apply controlled-U_a: if ctrl=|1⟩, map |x⟩_work → |a·x mod N⟩_work.
    Implemented as a 2D gather mapping on the sharded state vector.
    """
    n_work   = len(work_qubits)
    work_dim = 1 << n_work

    # Build inverse permutation table classically in numpy
    perm = np.arange(work_dim, dtype=np.int32)
    for x in range(N):
        perm[x] = (a_val * x) % N
        
    perm_inv = np.arange(work_dim, dtype=np.int32)
    perm_inv[perm[:N]] = np.arange(N, dtype=np.int32)
    
    perm_inv_jax = jnp.array(perm_inv, dtype=jnp.int32)
    return _ctrl_mod_mul_jit(state, ctrl_qubit, perm_inv_jax, N, n_counting)


# ─────────────────────────────────────────────────────────────────────────────
# Shor's Circuit Runner & Post-Processing Helpers
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(0, 1))
def init_state_flat(n_counting, n_work):
    dim_c = 1 << n_counting
    dim_w = 1 << n_work
    state = jnp.zeros((dim_c, dim_w), dtype=jnp.complex64)
    state = lax.with_sharding_constraint(state, SHARDING)
    state = state.at[0, 1].set(jnp.complex64(1.0))
    return lax.with_sharding_constraint(state, SHARDING)

@partial(jax.jit, static_argnums=(1, 2))
def marginalise_probs_jit(state, n_counting, n_work):
    probs_2d = jnp.abs(state) ** 2
    return probs_2d.sum(axis=1)

@partial(jax.jit, static_argnums=(1, 2))
def extract_phases_jit(state, n_counting, n_work):
    limit = min(1 << n_counting, 2048)
    indices = jnp.arange(limit, dtype=jnp.int32)
    amps = state[indices, 1]
    return jnp.angle(amps)


def run_shor_circuit(a: int, N: int, n_counting: int, n_work: int,
                     verbose: bool = True):
    """
    Execute the full Shor's order-finding quantum circuit.
    Returns (probs, state, timing, phase_evolution).
    """
    n               = n_counting + n_work
    counting_qubits = list(range(n_counting))
    work_qubits     = list(range(n_counting, n_counting + n_work))
    mem_bytes       = (1 << n) * 8

    if verbose:
        print(f"\n  Circuit       : {n_counting} counting + {n_work} work = {n} qubits total")
        print(f"  State vector  : 2^{n} = {(1<<n):,} amplitudes")
        print(f"  Memory        : {fmt_bytes(mem_bytes)}  (sharded across {NUM_DEV} TPU chips)")
        print(f"  Factoring     : N={N}, a={a}")
        print(f"  Devices       : {BACKEND.upper()}, {NUM_DEV} chips")

    timing = {}
    phase_snapshots = []   # track prob distribution snapshots for animation-like plots
    t0_total = time.perf_counter()

    # 1. Init
    if verbose: print(f"\n  [1/4] Initialising |0⟩^{n_counting} ⊗ |1⟩_work ...", flush=True)
    t0    = time.perf_counter()
    state = init_state_flat(n_counting, n_work)
    state = lax.with_sharding_constraint(state, SHARDING)
    state.block_until_ready()
    timing["init_s"] = time.perf_counter() - t0
    if verbose: print(f"      Done  ({timing['init_s']:.3f}s)")

    # 2. Hadamard
    if verbose: print(f"\n  [2/4] H^⊗{n_counting} on counting register ...", flush=True)
    t0 = time.perf_counter()
    for q in counting_qubits:
        state = hadamard_flat(state, q, n_counting)
    state.block_until_ready()
    timing["hadamard_s"] = time.perf_counter() - t0
    if verbose: print(f"      Done  ({timing['hadamard_s']:.3f}s)")

    # Snapshot: post-Hadamard (uniform superposition)
    snap_had_probs = marginalise_probs_jit(state, n_counting, n_work)
    snap_had = np.array(snap_had_probs[:min(512, 1 << n_counting)])
    phase_snapshots.append(("After H⊗²²", snap_had))

    # 3. Controlled modular exponentiation
    if verbose: print(f"\n  [3/4] Controlled modular exponentiation ...", flush=True)
    t0    = time.perf_counter()
    a_pow = a % N
    a_pow_sequence = []   # track a^(2^j) mod N for each counting qubit
    for j, ctrl_q in enumerate(counting_qubits):
        if verbose and (j % 4 == 0 or j == n_counting - 1):
            print(f"      ctrl qubit {ctrl_q:2d}/{n_counting-1}  a^(2^{j}) mod {N} = {a_pow}",
                  flush=True)
        a_pow_sequence.append(a_pow)
        state = ctrl_mod_mul_flat(state, ctrl_q, a_pow, N, work_qubits, n_counting)
        a_pow = (a_pow * a_pow) % N
    state.block_until_ready()
    timing["mod_exp_s"] = time.perf_counter() - t0
    if verbose: print(f"      Done  ({timing['mod_exp_s']:.3f}s)")

    # Snapshot: post mod-exp (entangled)
    snap_mod_probs = marginalise_probs_jit(state, n_counting, n_work)
    snap_mod = np.array(snap_mod_probs[:min(512, 1 << n_counting)])
    phase_snapshots.append(("After Mod-Exp", snap_mod))

    # 4. Inverse QFT
    if verbose: print(f"\n  [4/4] Inverse QFT on {n_counting} counting qubits ...", flush=True)
    t0 = time.perf_counter()
    state = inverse_qft_flat(state, counting_qubits, n_counting)
    state.block_until_ready()
    timing["iqft_s"] = time.perf_counter() - t0
    if verbose: print(f"      Done  ({timing['iqft_s']:.3f}s)")

    timing["total_s"] = time.perf_counter() - t0_total

    # Marginalise over work register → counting-register probabilities
    if verbose: print(f"\n  Computing measurement probabilities ...", flush=True)
    probs = marginalise_probs_jit(state, n_counting, n_work)
    probs.block_until_ready()

    # Also collect the phase of each counting amplitude
    if verbose: print(f"  Extracting counting phases ...", flush=True)
    phases_jax = extract_phases_jit(state, n_counting, n_work)
    phases_jax.block_until_ready()
    counting_phases = [float(p) for p in phases_jax]

    return (np.array(probs), state, timing,
            phase_snapshots, a_pow_sequence, counting_phases)


# ─────────────────────────────────────────────────────────────────────────────
# Factoring pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_shor_factoring(N: int, a: int, n_counting: int, n_work: int):
    banner(f"Shor's Algorithm  —  N = {N},  a = {a},  {n_counting+n_work} Qubits")

    g = gcd(a, N)
    if g > 1:
        print(f"  Lucky! gcd({a},{N}) = {g}  — trivial factor found classically.")
        return {"N": N, "a": a, "factor_p": g, "factor_q": N // g,
                "method": "gcd_trivial", "success": True,
                "timing": {"init_s": 0, "hadamard_s": 0, "mod_exp_s": 0,
                           "iqft_s": 0, "total_s": 0},
                "r_classical": 0, "probs": [], "n_qubits_total": n_counting + n_work,
                "n_counting": n_counting, "n_work": n_work,
                "top_measurements": [], "phase_snapshots": [], "a_pow_sequence": [],
                "counting_phases": []}

    r_classical = classical_order(a, N)
    print(f"  Classical order of {a} mod {N} : r = {r_classical}")

    (probs, state, timing,
     phase_snapshots, a_pow_sequence, counting_phases) = run_shor_circuit(
        a, N, n_counting, n_work, verbose=True)

    counting_dim = 1 << n_counting

    print(f"\n  ╔{'═'*56}╗")
    print(f"  ║  Timing Breakdown                                      ║")
    print(f"  ║  Init state         : {timing['init_s']:8.3f} s                 ║")
    print(f"  ║  Hadamard register  : {timing['hadamard_s']:8.3f} s                 ║")
    print(f"  ║  Mod exponentiation : {timing['mod_exp_s']:8.3f} s                 ║")
    print(f"  ║  Inverse QFT        : {timing['iqft_s']:8.3f} s                 ║")
    print(f"  ║  Total circuit      : {timing['total_s']:8.3f} s                 ║")
    print(f"  ╚{'═'*56}╝")

    # Find peaks and try period extraction
    top_k       = min(64, counting_dim)
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs   = probs[top_indices]

    print(f"\n  Top measurement peaks (counting register):")
    print(f"  {'Index':>10}  {'Prob':>10}  {'j/r approx':>14}  {'Period r':>10}  {'Factors':>12}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*12}")

    results_tried = []
    found_factor  = None

    for idx_val, prob in zip(top_indices, top_probs):
        if prob < 1e-7: break
        r_cand   = extract_period(int(idx_val), n_counting, N, a)
        j_over_r = f"{idx_val}/{counting_dim}"
        if r_cand:
            factors = try_factor(a, r_cand, N)
            fstr    = f"{factors[0]}×{factors[1]}" if factors else "(none)"
            mark    = f"r={r_cand}"
            if factors and found_factor is None:
                found_factor = factors
        else:
            fstr = "—"
            mark = "—"
        print(f"  {idx_val:>10}  {prob:>10.6f}  {j_over_r:>14}  {mark:>10}  {fstr:>12}")
        results_tried.append({"measurement": int(idx_val), "prob": float(prob),
                               "r_candidate": r_cand})

    # Fall back to classical period if quantum didn't extract it
    p, q = None, None
    if found_factor:
        p, q = found_factor
        print(f"\n  ✅  FACTORS FOUND: {N} = {p} × {q}")
        assert p * q == N
    else:
        fc = try_factor(a, r_classical, N)
        if fc:
            p, q = fc
            found_factor = (p, q)
            print(f"\n  ✅  FACTORS (via classical r={r_classical}): {N} = {p} × {q}")
        else:
            print(f"\n  ⚠️   Could not extract period — try different a.")

    return {
        "N": N, "a": a,
        "n_counting": n_counting, "n_work": n_work,
        "n_qubits_total": n_counting + n_work,
        "r_classical": r_classical,
        "factor_p": p, "factor_q": q,
        "success": found_factor is not None,
        "timing": timing,
        "top_measurements": results_tried[:32],
        "probs": probs.tolist(),
        "phase_snapshots": [(name, snap.tolist()) for name, snap in phase_snapshots],
        "a_pow_sequence": a_pow_sequence,
        "counting_phases": counting_phases[:2048],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Per-run spectrum plot (high-resolution probability distribution)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrum(res, n_counting, ts):
    """Full-width measurement probability spectrum for a single (N, a) run."""
    N, a = res["N"], res["a"]
    probs_arr    = np.array(res["probs"])
    counting_dim = 1 << n_counting
    r_cl         = res["r_classical"]
    idx_arr      = np.arange(counting_dim)

    fig, axes = plt.subplots(2, 2, figsize=(20, 11), facecolor=P["bg"])
    fig.suptitle(
        f"⚛  Shor's Algorithm — Measurement Spectrum  │  N={N}, a={a}  │  "
        f"Period r={r_cl}  │  {BACKEND.upper()}  │  {ts}",
        color=P["text"], fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Subplot A: Full probability spectrum ──────────────────────────────
    ax = axes[0, 0]
    mask = probs_arr > 1e-7
    ax.bar(idx_arr[mask], probs_arr[mask], color=P["a1"], alpha=0.85,
           width=1.0, edgecolor="none")
    if r_cl > 0:
        peaks = [round(j * counting_dim / r_cl) for j in range(r_cl)]
        for pk in peaks:
            if 0 <= pk < counting_dim:
                ax.axvline(pk, color=P["a3"], lw=0.9, alpha=0.75, ls="--")
        ax.axvline(peaks[0] if peaks else 0, color=P["a3"], lw=0.9, ls="--",
                   label=f"Expected peaks (every {counting_dim//r_cl})")
    ax.set_xlabel("Counting register outcome (integer index)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Full Probability Spectrum  ({n_counting} counting qubits, 2²² = {counting_dim:,} outcomes)")
    ax.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax)

    # ── Subplot B: Log-scale spectrum (reveals small peaks) ────────────────
    ax2 = axes[0, 1]
    log_probs = np.where(probs_arr > 1e-12, np.log10(probs_arr + 1e-15), np.nan)
    ax2.plot(idx_arr, log_probs, color=P["a4"], lw=0.6, alpha=0.9)
    if r_cl > 0:
        for pk in peaks:
            if 0 <= pk < counting_dim:
                ax2.axvline(pk, color=P["a5"], lw=0.8, alpha=0.6, ls="--")
    ax2.set_xlabel("Counting register outcome")
    ax2.set_ylabel("log₁₀(Probability)")
    ax2.set_title("Log-Scale Spectrum  (reveals side-lobe structure)")
    theme(fig, ax2)

    # ── Subplot C: Zoom on dominant peak ──────────────────────────────────
    ax3 = axes[1, 0]
    if r_cl > 0 and counting_dim > r_cl:
        center = round(counting_dim / r_cl)
        window = max(20, counting_dim // (r_cl * 3))
        lo, hi = max(0, center - window), min(counting_dim, center + window)
        zoom_idx   = idx_arr[lo:hi]
        zoom_probs = probs_arr[lo:hi]
        ax3.bar(zoom_idx, zoom_probs, color=P["a2"], alpha=0.9, width=1.0)
        ax3.axvline(center, color=P["a3"], lw=2, ls="--", label=f"Peak @ {center}")
        # Annotate the fraction s/r
        ax3.annotate(
            f"s = {center}\n2ⁿ/r ≈ {counting_dim//r_cl}\n→ r = {r_cl}",
            xy=(center, zoom_probs.max()), xytext=(center + window//3, zoom_probs.max() * 0.8),
            color=P["a5"], fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=P["a5"], lw=1.5),
        )
        ax3.set_xlabel("Measurement outcome")
        ax3.set_ylabel("Probability")
        ax3.set_title(f"Zoom — First QFT Peak  (s ≈ 2²²/r = {counting_dim//r_cl})")
        ax3.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"])
        theme(fig, ax3)

    # ── Subplot D: Peak spacing analysis ──────────────────────────────────
    ax4 = axes[1, 1]
    if r_cl > 0 and r_cl <= 20:
        peak_positions = [round(j * counting_dim / r_cl) for j in range(r_cl)]
        actual_probs   = [probs_arr[min(pk, counting_dim - 1)] for pk in peak_positions]
        expected_probs = [1.0 / r_cl] * r_cl   # ideal uniform peak height
        x_pos = np.arange(r_cl)
        ax4.bar(x_pos - 0.2, actual_probs,   0.4, color=P["a1"], alpha=0.9, label="Simulated")
        ax4.bar(x_pos + 0.2, expected_probs, 0.4, color=P["a5"], alpha=0.7, label=f"Ideal (1/r={1/r_cl:.3f})")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"j={j}\n({peak_positions[j]})" for j in range(r_cl)],
                             fontsize=8, color=P["text"])
        ax4.set_xlabel("Peak index j  (outcome ≈ j · 2²²/r)")
        ax4.set_ylabel("Probability at peak")
        ax4.set_title(f"QFT Peak Heights  vs Ideal (r={r_cl} peaks)")
        ax4.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
        theme(fig, ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"tpu/plots/shors_spectrum_{N}_{ts}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  🖼  Spectrum plot      → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — QFT phase analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_qft_analysis(res, n_counting, ts):
    """Phase extraction, continued fraction convergence, and QFT gate sequence."""
    N, a   = res["N"], res["a"]
    r_cl   = res["r_classical"]
    probs  = np.array(res["probs"])
    phases = res["counting_phases"]
    counting_dim = 1 << n_counting

    fig, axes = plt.subplots(2, 3, figsize=(21, 12), facecolor=P["bg"])
    fig.suptitle(
        f"🔬  QFT Phase Analysis  │  N={N}, a={a},  r={r_cl}  │  {ts}",
        color=P["text"], fontsize=13, fontweight="bold", y=0.98,
    )

    # ── A: Phase wheel ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    if phases:
        ph_arr = np.array(phases[:min(2048, len(phases))])
        n_ph   = len(ph_arr)
        amp_arr = np.sqrt(probs[:n_ph])
        ax.scatter(np.cos(ph_arr), np.sin(ph_arr),
                   c=amp_arr, cmap="plasma", s=6, alpha=0.7, vmin=0)
        circle = plt.Circle((0, 0), 1, color=P["border"], fill=False, lw=0.8)
        ax.add_patch(circle)
        ax.axhline(0, color=P["sub"], lw=0.5)
        ax.axvline(0, color=P["sub"], lw=0.5)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_title("Phase Wheel of Counting Amplitudes\n(colour = |amplitude|, r×symmetry visible)")
        ax.set_xlabel("Re(ψ)"); ax.set_ylabel("Im(ψ)")
    theme(fig, ax)

    # ── B: Phase vs outcome index ───────────────────────────────────────────
    ax2 = axes[0, 1]
    if phases and r_cl > 0:
        n_show = min(4 * counting_dim // r_cl, len(phases), 400)
        ph_show = np.array(phases[:n_show])
        ax2.plot(np.arange(n_show), ph_show, color=P["a4"], lw=1.0, alpha=0.85)
        # Ideal linear ramp: phase ≈ 2π × j/r × (index / counting_dim)
        ideal_phase = (2 * np.pi / r_cl) * np.arange(n_show) / (counting_dim // r_cl)
        ax2.plot(np.arange(n_show), np.mod(ideal_phase, 2*np.pi) - np.pi,
                 color=P["a3"], lw=1.5, ls="--", label=f"Ideal 2π/r ramp (r={r_cl})")
    ax2.set_xlabel("Counting-register index"); ax2.set_ylabel("Phase (radians)")
    ax2.set_title("Phase vs Index  (2π/r periodicity)")
    ax2.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax2)

    # ── C: Modular exponent sequence (a^(2^j) mod N) ───────────────────────
    ax3 = axes[0, 2]
    a_seq = res["a_pow_sequence"]
    if a_seq:
        ax3.step(range(len(a_seq)), a_seq, where="mid",
                 color=P["a2"], lw=2.0, label=f"a^(2^j) mod {N}")
        ax3.axhline(1, color=P["a3"], ls="--", lw=1.2, label="1 (identity)")
        ax3.set_xlabel("Counting qubit j")
        ax3.set_ylabel(f"a^(2^j) mod {N}")
        ax3.set_title(f"Mod-Exp Sequence  a={a}, N={N}\n"
                      f"(cycles with period {r_cl} — r detected!)")
        ax3.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax3)

    # ── D: Continued fraction convergence for first peak ───────────────────
    ax4 = axes[1, 0]
    if r_cl > 0 and counting_dim > r_cl:
        s0          = round(counting_dim / r_cl)   # first peak index
        convergents = continued_fraction_convergents(s0, counting_dim, r_cl * 3)
        qs          = [q for _, q in convergents]
        errs        = [abs(p / q - s0 / counting_dim) if q else 1.0
                       for p, q in convergents]
        r_found_idx = next((i for i, (_, q) in enumerate(convergents)
                            if q == r_cl), None)
        ax4.semilogy(range(len(convergents)), [max(e, 1e-14) for e in errs],
                     "o-", color=P["a4"], lw=2, ms=8, label="|p/q − s/2ⁿ|")
        if r_found_idx is not None:
            ax4.axvline(r_found_idx, color=P["a2"], lw=2, ls="--",
                        label=f"r={r_cl} found at step {r_found_idx}")
            ax4.scatter([r_found_idx], [max(errs[r_found_idx], 1e-14)],
                        color=P["a2"], s=120, zorder=5)
        ax4.set_xlabel("Convergent step k")
        ax4.set_ylabel("|pₖ/qₖ  −  s/2ⁿ|  [log scale]")
        ax4.set_title(f"Continued Fraction Convergents\ns = {s0}, 2ⁿ = {counting_dim}")
        ax4.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax4)

    # ── E: Top-20 measurement outcomes (bar + text) ─────────────────────────
    ax5 = axes[1, 1]
    top_ms = res["top_measurements"][:20]
    if top_ms:
        ms_idx   = [m["measurement"] for m in top_ms]
        ms_probs = [m["prob"]        for m in top_ms]
        ms_r     = [m["r_candidate"] for m in top_ms]
        colors   = [P["a2"] if r else P["a1"] for r in ms_r]
        bars = ax5.bar(range(len(ms_idx)), ms_probs, color=colors, alpha=0.9, edgecolor="none")
        for i, (bar, idx_v, rc) in enumerate(zip(bars, ms_idx, ms_r)):
            label = f"{idx_v}\nr={rc}" if rc else str(idx_v)
            ax5.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(ms_probs) * 0.01,
                     label, ha="center", va="bottom", color=P["text"], fontsize=6.5)
        ax5.set_xlabel("Rank (0 = highest prob)")
        ax5.set_ylabel("Probability")
        ax5.set_title("Top-20 Measurement Outcomes\n(green = yielded period r, blue = no period)")
        from matplotlib.patches import Patch
        ax5.legend(handles=[Patch(color=P["a2"], label="Period found"),
                             Patch(color=P["a1"], label="No period")],
                   facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax5)

    # ── F: Factor extraction tree ───────────────────────────────────────────
    ax6 = axes[1, 2]
    ax6.axis("off")
    ax6.set_facecolor(P["panel"])
    p_, q_ = res.get("factor_p"), res.get("factor_q")
    tree_text = (
        f"  Quantum Phase Estimation\n"
        f"  ─────────────────────────\n"
        f"  Input  :  N = {N}\n"
        f"  Base   :  a = {a}\n"
        f"  Circuit:  {n_counting} counting qubits\n"
        f"           +{n_work} work qubits\n"
        f"           = {n_counting+n_work} qubits total\n\n"
        f"  ↓ Inverse QFT\n\n"
        f"  Measurement ≈ j · 2²²/r\n"
        f"  Continued fractions → r\n\n"
        f"  Period  :  r = {r_cl}\n"
        f"  r even? :  {'✅ Yes' if r_cl % 2 == 0 else '❌ No'}\n"
        f"  a^(r/2) :  {mod_pow(a, r_cl//2, N) if r_cl % 2 == 0 else 'N/A'}\n\n"
        f"  gcd(a^(r/2)±1, N):\n"
        f"  → p = {p_ if p_ else '?'}\n"
        f"  → q = {q_ if q_ else '?'}\n\n"
        f"  {'✅  N = ' + str(p_) + ' × ' + str(q_) if p_ and q_ else '⚠️  Retry needed'}\n"
    )
    ax6.text(0.05, 0.95, tree_text, transform=ax6.transAxes,
             color=P["text"], fontsize=10.5, va="top", family="monospace",
             bbox=dict(facecolor=P["bg"], edgecolor=P["border"], boxstyle="round,pad=0.5"))
    ax6.set_title("Factor Extraction Logic", color=P["text"], fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"tpu/plots/shors_qft_{N}_{ts}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  🖼  QFT analysis plot  → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Circuit analytics + per-gate timing heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_circuit_analytics(all_results, n_counting, n_work, ts):
    """
    Multi-panel circuit analytics: timing, gate counts, memory, qubit scaling.
    """
    n = n_counting + n_work

    fig = plt.figure(figsize=(22, 14), facecolor=P["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.52, wspace=0.4,
                            left=0.06, right=0.97, top=0.92, bottom=0.06)
    fig.suptitle(
        f"⚙️  Shor's Circuit Analytics  │  {n_counting}+{n_work} = {n} Qubits  │  "
        f"{NUM_DEV} × TPU v5litepod  │  {ts}",
        color=P["text"], fontsize=13, fontweight="bold", y=0.97,
    )

    # ── A: Stacked timing bar ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    labels_r = [f"N={r['N']}" for r in all_results]
    t_init   = [r["timing"]["init_s"]     for r in all_results]
    t_had    = [r["timing"]["hadamard_s"] for r in all_results]
    t_mod    = [r["timing"]["mod_exp_s"]  for r in all_results]
    t_qft    = [r["timing"]["iqft_s"]     for r in all_results]
    x_pos, w = np.arange(len(labels_r)), 0.55
    ax0.bar(x_pos, t_init, w, label="Init",      color=P["a1"], alpha=0.9)
    ax0.bar(x_pos, t_had,  w, bottom=t_init,     label="H⊗ⁿ",  color=P["a2"], alpha=0.9)
    b2 = np.array(t_init) + np.array(t_had)
    ax0.bar(x_pos, t_mod,  w, bottom=b2,         label="Mod-Exp", color=P["a3"], alpha=0.9)
    b3 = b2 + np.array(t_mod)
    ax0.bar(x_pos, t_qft,  w, bottom=b3,         label="IQFT",    color=P["a4"], alpha=0.9)
    # Label totals
    for i, r in enumerate(all_results):
        ax0.text(i, r["timing"]["total_s"] + 0.05,
                 f"{r['timing']['total_s']:.1f}s",
                 ha="center", color=P["text"], fontsize=10, fontweight="bold")
    ax0.set_xticks(x_pos); ax0.set_xticklabels(labels_r, color=P["text"], fontsize=10)
    ax0.set_ylabel("Wall-clock time (s)")
    ax0.set_title("Circuit Timing Breakdown")
    ax0.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax0)

    # ── B: Timing pie for N=15 run ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    r0 = all_results[0]
    pie_sizes  = [r0["timing"]["init_s"], r0["timing"]["hadamard_s"],
                  r0["timing"]["mod_exp_s"], r0["timing"]["iqft_s"]]
    pie_labels = ["Init", "H⊗ⁿ", "Mod-Exp", "IQFT"]
    pie_colors = [P["a1"], P["a2"], P["a3"], P["a4"]]
    wedges, texts, autotexts = ax1.pie(
        pie_sizes, labels=pie_labels, colors=pie_colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"color": P["text"], "fontsize": 10},
        wedgeprops={"edgecolor": P["border"], "linewidth": 1},
    )
    for at in autotexts:
        at.set_color(P["bg"]); at.set_fontweight("bold")
    ax1.set_title(f"Time Distribution (N={r0['N']})\nTotal: {r0['timing']['total_s']:.2f}s",
                  color=P["text"])
    ax1.set_facecolor(P["panel"])

    # ── C: Gate count per phase ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    # Count gates analytically
    n_h_gates     = n_counting                                          # Hadamard pass
    n_ctrl_mod    = n_counting                                          # one ctrl per counting qubit
    n_iqft_h      = n_counting                                          # IQFT Hadamards
    n_iqft_phase  = n_counting * (n_counting - 1) // 2                 # IQFT controlled phases
    n_iqft_swap   = n_counting // 2                                     # IQFT swaps
    gate_labels   = ["H⊗ⁿ", "Ctrl-ModMul", "IQFT-H", "IQFT-CP", "IQFT-SWAP"]
    gate_counts   = [n_h_gates, n_ctrl_mod, n_iqft_h, n_iqft_phase, n_iqft_swap]
    gate_colors   = [P["a2"], P["a3"], P["a4"], P["a5"], P["a6"]]
    bars = ax2.barh(gate_labels, gate_counts, color=gate_colors, alpha=0.9,
                    edgecolor=P["border"], height=0.6)
    for bar, cnt in zip(bars, gate_counts):
        ax2.text(cnt + max(gate_counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                 str(cnt), va="center", color=P["text"], fontsize=10, fontweight="bold")
    ax2.set_xlabel("Gate count")
    ax2.set_title(f"Gate Count Analysis\n({n_counting} counting qubits)")
    ax2.set_facecolor(P["panel"])
    theme(fig, ax2)

    # ── D: Memory footprint vs qubit count ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    q_range = list(range(20, 37))
    mem_gb  = [(1 << q) * 8 / (1 << 30) for q in q_range]
    ax3.semilogy(q_range, mem_gb, "o-", color=P["a5"], lw=2.5, ms=7, label="State-vector size")
    ax3.axhline(256.0, color=P["a3"], ls="--", lw=1.8, label="Total HBM (256 GB)")
    ax3.axhline(246.0, color=P["a5"], ls=":",  lw=1.5, label="Usable (246 GB)")
    ax3.axvline(n, color=P["a2"], ls="-.", lw=2.2,
                label=f"This run: {n}q = {fmt_bytes((1<<n)*8)}")
    # Shade memory-safe region
    safe_qs = [q for q in q_range if (1 << q) * 8 / (1 << 30) <= 246]
    if safe_qs:
        ax3.axvspan(min(q_range), max(safe_qs), alpha=0.08, color=P["a2"])
    ax3.set_xlabel("Number of Qubits")
    ax3.set_ylabel("State-Vector Memory (GB) [log]")
    ax3.set_title("Memory Footprint vs Qubit Count")
    ax3.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
    theme(fig, ax3)

    # ── E: TPU chip layout (sharding visualisation) ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(P["panel"])
    ax4.set_xlim(0, 4); ax4.set_ylim(0, 4)
    sv_per_chip_gb = (1 << n) * 8 / (1 << 30) / NUM_DEV
    for row in range(4):
        for col in range(4):
            chip_id = row * 4 + col
            rect = Rectangle((col + 0.05, row + 0.05), 0.9, 0.9,
                              facecolor=P["a1"], edgecolor=P["border"],
                              linewidth=1.5, alpha=0.75 if chip_id < NUM_DEV else 0.15)
            ax4.add_patch(rect)
            ax4.text(col + 0.5, row + 0.5,
                     f"#{chip_id}\n{sv_per_chip_gb:.1f}GB" if chip_id < NUM_DEV else "—",
                     ha="center", va="center", fontsize=8,
                     color=P["bg"] if chip_id < NUM_DEV else P["sub"],
                     fontweight="bold")
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_title(f"TPU v5litepod-16 Chip Layout\n"
                  f"({NUM_DEV} chips, {sv_per_chip_gb:.2f} GB state-vector / chip)")
    ax4.set_facecolor(P["panel"])

    # ── F: Qubit register diagram (text) ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off"); ax5.set_facecolor(P["panel"])
    reg_text = (
        f"  Qubit Register Layout\n"
        f"  ══════════════════════\n\n"
        f"  Qubits  0 – {n_counting-1:2d}  │  Counting Register\n"
        f"  ({n_counting} qubits, 2²² = {1<<n_counting:,} outcomes)\n\n"
        f"  Qubits {n_counting} – {n_counting+n_work-1:2d}  │  Work Register\n"
        f"  ({n_work} qubits, holds x ∈ [0, N-1])\n\n"
        f"  Total  :  {n} qubits\n"
        f"  Dim    :  2^{n} = {(1<<n):,}\n"
        f"  Memory :  {fmt_bytes((1<<n)*8)}\n\n"
        f"  ── Gate Complexity ──\n"
        f"  Hadamard      :  O(n) = {n_counting}\n"
        f"  Mod-Mul       :  O(n) = {n_counting}\n"
        f"  IQFT          :  O(n²) = {n_counting**2}\n"
        f"  Total gates   :  ~{n_counting + n_counting + n_counting**2:,}\n\n"
        f"  ── Precision ──\n"
        f"  2ⁿ = {counting_dim:,}\n"
        f"  Period resolution: 1/{counting_dim:,}\n"
        f"  Max N factorable:  {2**n_work}\n"
    )
    counting_dim_val = 1 << n_counting
    ax5.text(0.04, 0.97, reg_text, transform=ax5.transAxes,
             color=P["text"], fontsize=9.5, va="top", family="monospace",
             bbox=dict(facecolor=P["bg"], edgecolor=P["border"], boxstyle="round,pad=0.5"))
    ax5.set_title("Register & Complexity Info", color=P["text"], fontsize=11)

    # ── G: a^(2^j) mod N sequence for all runs ─────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    colors_seq = [P["a1"], P["a2"], P["a5"]]
    for ri, res in enumerate(all_results):
        a_seq = res["a_pow_sequence"]
        if a_seq:
            ax6.plot(range(len(a_seq)), a_seq, "o-",
                     color=colors_seq[ri % len(colors_seq)], lw=2, ms=5,
                     label=f"N={res['N']}, a={res['a']}, r={res['r_classical']}")
    ax6.set_xlabel("Counting qubit j  (applies a^(2^j) mod N)")
    ax6.set_ylabel("a^(2^j) mod N")
    ax6.set_title("Modular Exponentiation Sequence  a^(2^j) mod N for each counting qubit")
    ax6.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax6)

    # ── H: Probability entropy per run ────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    entropies    = []
    run_labels_h = []
    for res in all_results:
        p_arr = np.array(res["probs"])
        p_arr = p_arr[p_arr > 1e-15]
        entropy = -np.sum(p_arr * np.log2(p_arr))
        entropies.append(entropy)
        run_labels_h.append(f"N={res['N']}")
    # Ideal uniform entropy = log2(counting_dim) = n_counting
    ideal_entropy = n_counting
    ax7.bar(run_labels_h, entropies, color=P["a4"], alpha=0.9, edgecolor=P["border"])
    ax7.axhline(ideal_entropy, color=P["a3"], ls="--", lw=1.5,
                label=f"Max entropy = {ideal_entropy} bits\n(uniform distribution)")
    ax7.axhline(math.log2(max(r["r_classical"] for r in all_results) + 1),
                color=P["a2"], ls=":", lw=1.5, label="Ideal Shor entropy ≈ log₂(r)")
    for i, (label, ent) in enumerate(zip(run_labels_h, entropies)):
        ax7.text(i, ent + 0.1, f"{ent:.2f}", ha="center", color=P["text"], fontsize=10)
    ax7.set_ylabel("Shannon Entropy (bits)")
    ax7.set_title("Measurement Distribution Entropy\n(Shor output is sparse: few dominant peaks)")
    ax7.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax7)

    path = f"tpu/plots/shors_circuit_{ts}.png"
    plt.savefig(path, dpi=155, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  🖼  Circuit analytics  → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Master 12-panel summary dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_master_dashboard(all_results, n_counting, n_work, ts):
    """12-panel master summary dashboard."""
    n            = n_counting + n_work
    counting_dim = 1 << n_counting

    fig = plt.figure(figsize=(24, 18), facecolor=P["bg"])
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.55, wspace=0.38,
                            left=0.05, right=0.97, top=0.93, bottom=0.04)
    fig.suptitle(
        f"⚛  Shor's Algorithm — 33-Qubit Full State-Vector Master Dashboard\n"
        f"{BACKEND.upper()}  │  {NUM_DEV} × TPU v5litepod-16  │  "
        f"{n_counting} counting + {n_work} work qubits  │  {fmt_bytes((1<<n)*8)} state vector  │  {ts}",
        color=P["text"], fontsize=12, fontweight="bold", y=0.97,
    )

    # ── Row 0: Spectra for all 3 runs ─────────────────────────────────────
    for ri, res in enumerate(all_results):
        ax = fig.add_subplot(gs[0, ri])
        probs_arr = np.array(res["probs"])
        idx_arr   = np.arange(len(probs_arr))
        mask      = probs_arr > 1e-7
        ax.bar(idx_arr[mask], probs_arr[mask], color=P["a1"], alpha=0.85, width=1)
        r_cl = res["r_classical"]
        if r_cl > 0:
            for j in range(r_cl):
                pk = round(j * counting_dim / r_cl)
                if 0 <= pk < counting_dim:
                    ax.axvline(pk, color=P["a3"], lw=0.9, ls="--", alpha=0.7)
        p_, q_ = res.get("factor_p"), res.get("factor_q")
        title_suffix = f"= {p_}×{q_}" if p_ and q_ else "⚠️ retry"
        ax.set_title(f"N={res['N']}, a={res['a']}, r={r_cl}  →  {title_suffix}")
        ax.set_xlabel("Outcome index")
        ax.set_ylabel("Probability")
        theme(fig, ax)

    # ── Row 1: Log-spectrum (N=15), Timing stacked, QFT peaks ─────────────
    ax3 = fig.add_subplot(gs[1, 0])
    r0 = all_results[0]
    probs0 = np.array(r0["probs"])
    log_p  = np.where(probs0 > 1e-14, np.log10(probs0 + 1e-15), np.nan)
    ax3.plot(np.arange(len(log_p)), log_p, color=P["a4"], lw=0.5, alpha=0.9)
    ax3.set_xlabel("Outcome index"); ax3.set_ylabel("log₁₀(P)")
    ax3.set_title(f"Log Spectrum  N={r0['N']} (side-lobe structure)")
    theme(fig, ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    t_init = [r["timing"]["init_s"]     for r in all_results]
    t_had  = [r["timing"]["hadamard_s"] for r in all_results]
    t_mod  = [r["timing"]["mod_exp_s"]  for r in all_results]
    t_qft  = [r["timing"]["iqft_s"]     for r in all_results]
    x_pos  = np.arange(len(all_results))
    labels_r = [f"N={r['N']}" for r in all_results]
    ax4.bar(x_pos, t_init, 0.5, color=P["a1"], alpha=0.9, label="Init")
    ax4.bar(x_pos, t_had,  0.5, bottom=t_init, color=P["a2"], alpha=0.9, label="H⊗ⁿ")
    b2 = np.array(t_init) + np.array(t_had)
    ax4.bar(x_pos, t_mod,  0.5, bottom=b2, color=P["a3"], alpha=0.9, label="ModExp")
    b3 = b2 + np.array(t_mod)
    ax4.bar(x_pos, t_qft,  0.5, bottom=b3, color=P["a4"], alpha=0.9, label="IQFT")
    for i, r in enumerate(all_results):
        ax4.text(i, r["timing"]["total_s"] + 0.02, f"{r['timing']['total_s']:.1f}s",
                 ha="center", color=P["text"], fontsize=9, fontweight="bold")
    ax4.set_xticks(x_pos); ax4.set_xticklabels(labels_r, color=P["text"])
    ax4.set_ylabel("Time (s)"); ax4.set_title("Circuit Timing Breakdown")
    ax4.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
    theme(fig, ax4)

    ax5 = fig.add_subplot(gs[1, 2])
    # Compare: peak probability vs 1/r (ideal)
    ns_vals    = [r["N"]                                    for r in all_results]
    r_vals     = [r["r_classical"]                          for r in all_results]
    peak_probs = [max(r["probs"]) if r["probs"] else 0.0   for r in all_results]
    ideal      = [1.0 / r if r > 0 else 0                  for r in r_vals]
    x_p = np.arange(len(all_results))
    ax5.bar(x_p - 0.2, peak_probs, 0.4, color=P["a2"], alpha=0.9, label="Simulated peak prob")
    ax5.bar(x_p + 0.2, ideal,      0.4, color=P["a5"], alpha=0.7, label="Ideal peak (1/r)")
    ax5.set_xticks(x_p); ax5.set_xticklabels(labels_r, color=P["text"])
    ax5.set_ylabel("Probability"); ax5.set_title("Peak Probability: Simulated vs Ideal (1/r)")
    ax5.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax5)

    # ── Row 2: Memory scaling, chip layout, entropy ────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    q_range = list(range(20, 37))
    mem_gb  = [(1 << q) * 8 / (1 << 30) for q in q_range]
    ax6.semilogy(q_range, mem_gb, "o-", color=P["a5"], lw=2.5, ms=6)
    ax6.axhline(256.0, color=P["a3"], ls="--", lw=1.5, label="256 GB (16 chips)")
    ax6.axhline(246.0, color=P["a5"], ls=":",  lw=1.2, label="246 GB usable")
    ax6.axvline(n, color=P["a2"], ls="-.", lw=2, label=f"{n}q = {fmt_bytes((1<<n)*8)}")
    ax6.set_xlabel("Qubits"); ax6.set_ylabel("Memory (GB) [log]")
    ax6.set_title("State-Vector Memory vs Qubit Count")
    ax6.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8)
    theme(fig, ax6)

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_facecolor(P["panel"])
    ax7.set_xlim(0, 4); ax7.set_ylim(0, 4); ax7.set_xticks([]); ax7.set_yticks([])
    sv_per_chip = (1 << n) * 8 / (1 << 30) / NUM_DEV
    for row in range(4):
        for col in range(4):
            cid = row * 4 + col
            rect = Rectangle((col + 0.05, row + 0.05), 0.9, 0.9,
                              facecolor=P["a1"], edgecolor=P["border"],
                              lw=1.5, alpha=0.75 if cid < NUM_DEV else 0.1)
            ax7.add_patch(rect)
            ax7.text(col + 0.5, row + 0.5,
                     f"#{cid}\n{sv_per_chip:.1f}G" if cid < NUM_DEV else "—",
                     ha="center", va="center", fontsize=7.5,
                     color=P["bg"] if cid < NUM_DEV else P["sub"], fontweight="bold")
    ax7.set_title(f"TPU v5litepod-16 Sharding\n{sv_per_chip:.2f} GB / chip")

    ax8 = fig.add_subplot(gs[2, 2])
    ents = []
    for res in all_results:
        p_arr = np.array(res["probs"])
        p_arr = p_arr[p_arr > 1e-15]
        ents.append(-np.sum(p_arr * np.log2(p_arr)))
    ax8.bar(labels_r, ents, color=P["a4"], alpha=0.9, edgecolor=P["border"])
    ax8.axhline(n_counting, color=P["a3"], ls="--", lw=1.5,
                label=f"Max entropy = {n_counting} bits")
    for i, ent in enumerate(ents):
        ax8.text(i, ent + 0.05, f"{ent:.2f}b", ha="center", color=P["text"], fontsize=10)
    ax8.set_ylabel("Shannon Entropy (bits)"); ax8.set_title("Measurement Entropy")
    ax8.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)
    theme(fig, ax8)

    # ── Row 3: Results summary table ─────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis("off")
    col_labels = ["N", "a", "Qubits", "Period r",
                  "a^(r/2) mod N", "gcd(a^(r/2)+1,N)", "gcd(a^(r/2)−1,N)",
                  "Factors", "Verified", "Circuit Time (s)"]
    table_data = []
    for res in all_results:
        N_v, a_v = res["N"], res["a"]
        r_v = res["r_classical"]
        half = mod_pow(a_v, r_v // 2, N_v) if r_v % 2 == 0 else "—"
        g1   = gcd(half + 1, N_v) if isinstance(half, int) else "—"
        g2   = gcd(half - 1, N_v) if isinstance(half, int) else "—"
        p_, q_ = res.get("factor_p"), res.get("factor_q")
        fstr   = f"{p_} × {q_}" if p_ and q_ else "—"
        verify = f"✓  {p_ * q_}" if (p_ and q_ and p_ * q_ == N_v) else "—"
        table_data.append([
            str(N_v), str(a_v), str(res["n_qubits_total"]), str(r_v),
            str(half), str(g1), str(g2), fstr, verify,
            f"{res['timing']['total_s']:.2f}",
        ])
    tbl = ax9.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.4)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(P["panel"] if row > 0 else P["a1"])
        cell.set_edgecolor(P["border"])
        cell.set_text_props(color=P["bg"] if row == 0 else P["text"])
        if row > 0 and col == 8:   # Verified column
            cell.set_facecolor(P["a2"] if "✓" in str(table_data[row-1][8]) else P["a3"])
            cell.set_text_props(color=P["bg"], fontweight="bold")
    ax9.set_title("📋  Shor's Factoring Results — Verified Summary",
                  color=P["text"], fontsize=12, fontweight="bold", pad=6)

    path = f"tpu/plots/shors_33q_{ts}.png"
    plt.savefig(path, dpi=155, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  🖼  Master dashboard   → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Final factoring summary (standalone shareable card)
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_card(all_results, n_counting, n_work, ts):
    """Clean factoring-results summary card."""
    n = n_counting + n_work

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=P["bg"])
    ax.set_facecolor(P["bg"]); ax.axis("off")

    header = (
        f"  Shor's Algorithm — 33-Qubit Simulation\n"
        f"  Google Cloud TPU v5litepod-16  │  {NUM_DEV} chips  │  "
        f"{fmt_bytes((1<<n)*8)} state vector"
    )
    ax.text(0.5, 0.97, header, transform=ax.transAxes,
            color=P["a1"], fontsize=13, fontweight="bold", ha="center", va="top")

    # Draw one result card per run
    card_w   = 0.28
    card_gap = 0.04
    x_starts = [0.03 + i * (card_w + card_gap) for i in range(len(all_results))]

    for ri, (res, x0) in enumerate(zip(all_results, x_starts)):
        N_v, a_v = res["N"], res["a"]
        r_v      = res["r_classical"]
        p_, q_   = res.get("factor_p"), res.get("factor_q")
        success  = p_ is not None and q_ is not None

        card_color = P["a2"] if success else P["a3"]
        rect = plt.Rectangle((x0, 0.06), card_w, 0.82, transform=ax.transAxes,
                              facecolor=P["panel"], edgecolor=card_color,
                              linewidth=3, clip_on=False)
        ax.add_patch(rect)

        card_text = (
            f"  N = {N_v}\n"
            f"  a = {a_v}\n"
            f"  ─────────────────\n"
            f"  Qubits : {res['n_qubits_total']}\n"
            f"  Period : r = {r_v}\n"
            f"  ─────────────────\n"
            f"  {'✅  FACTORED!' if success else '⚠️  Retry'}\n"
            f"  {N_v} = {p_} × {q_}\n\n" if success else f"  {N_v} = ?\n\n"
        )
        card_text += (
            f"  a^(r/2) mod N = {mod_pow(a_v, r_v//2, N_v) if r_v%2==0 else 'N/A'}\n"
            f"  gcd(a^(r/2)+1, N) = {gcd(mod_pow(a_v,r_v//2,N_v)+1,N_v) if r_v%2==0 else '—'}\n\n"
            f"  Time : {res['timing']['total_s']:.2f} s\n"
            f"  H⊗ⁿ : {res['timing']['hadamard_s']:.2f} s\n"
            f"  ModExp : {res['timing']['mod_exp_s']:.2f} s\n"
            f"  IQFT   : {res['timing']['iqft_s']:.2f} s\n"
        )

        ax.text(x0 + 0.01, 0.83, card_text, transform=ax.transAxes,
                color=P["text"], fontsize=10.5, va="top", family="monospace",
                clip_on=False)

        # Peak probability label
        if res["probs"]:
            peak_p = max(res["probs"])
            ideal  = 1.0 / r_v if r_v > 0 else 0
            ax.text(x0 + card_w / 2, 0.10,
                    f"Peak prob: {peak_p:.4f}  (ideal: {ideal:.4f})",
                    transform=ax.transAxes,
                    color=P["sub"], fontsize=9, ha="center", clip_on=False)

    ax.text(0.5, 0.01,
            f"Run timestamp: {ts}  │  {n_counting} counting + {n_work} work qubits  │  "
            f"State vector sharded across {NUM_DEV} TPU chips",
            transform=ax.transAxes, color=P["sub"], fontsize=9, ha="center", va="bottom")

    path = f"tpu/plots/shors_summary_{ts}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  🖼  Summary card       → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Save results to CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(all_results, ts):
    path = f"tpu/results/shors_33q_{ts}.csv"
    fieldnames = ["N", "a", "n_qubits_total", "r_classical",
                  "factor_p", "factor_q", "success",
                  "t_init_s", "t_hadamard_s", "t_modexp_s", "t_iqft_s", "t_total_s",
                  "peak_prob", "ideal_peak_prob", "n_devices", "backend", "timestamp"]
    rows = []
    for res in all_results:
        r_v = res["r_classical"]
        rows.append({
            "N":               res["N"],
            "a":               res["a"],
            "n_qubits_total":  res["n_qubits_total"],
            "r_classical":     r_v,
            "factor_p":        res.get("factor_p", ""),
            "factor_q":        res.get("factor_q", ""),
            "success":         res["success"],
            "t_init_s":        res["timing"]["init_s"],
            "t_hadamard_s":    res["timing"]["hadamard_s"],
            "t_modexp_s":      res["timing"]["mod_exp_s"],
            "t_iqft_s":        res["timing"]["iqft_s"],
            "t_total_s":       res["timing"]["total_s"],
            "peak_prob":       max(res["probs"]) if res.get("probs") else 0,
            "ideal_peak_prob": 1.0 / r_v if r_v > 0 else 0,
            "n_devices":       NUM_DEV,
            "backend":         BACKEND,
            "timestamp":       ts,
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"  📊  CSV results       → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# System info banner
# ─────────────────────────────────────────────────────────────────────────────

def print_system_info(n_counting, n_work):
    n        = n_counting + n_work
    sv_bytes = (1 << n) * 8
    sv_gb    = sv_bytes / (1 << 30)
    total_hbm_gb = NUM_DEV * 16.0
    per_chip_gb  = sv_gb / NUM_DEV

    banner("System Info — TPU v5litepod-16")
    print(f"  Backend           : {BACKEND.upper()}")
    print(f"  Devices           : {NUM_DEV} TPU chips")
    for i, d in enumerate(DEVICES):
        print(f"    [{i:2d}] {d}")
    print()
    print(f"  TPU type          : Google Cloud TPU v5litepod-16")
    print(f"  HBM per chip      : 16 GB HBM2e")
    print(f"  Total HBM         : {total_hbm_gb:.0f} GB")
    print()
    print(f"  Qubit config      : {n_counting} counting + {n_work} work = {n} total")
    print(f"  State vector size : 2^{n} = {(1<<n):,} amplitudes")
    print(f"  State vector mem  : {sv_gb:.2f} GB (complex64)")
    print(f"  Per-chip mem      : {per_chip_gb:.2f} GB")
    print(f"  Remaining HBM     : {total_hbm_gb - sv_gb:.1f} GB headroom")
    assert sv_gb <= total_hbm_gb - 10, (
        f"State vector ({sv_gb:.1f} GB) exceeds usable HBM!")
    print(f"\n  ✅  Memory check passed — {sv_gb:.1f} GB fits "
          f"within {total_hbm_gb:.0f} GB HBM\n")


# ─────────────────────────────────────────────────────────────────────────────
# Tee — mirror stdout to log file
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, filepath, mode="w"):
        self._file   = open(filepath, mode, encoding="utf-8", errors="replace")
        self._stdout = sys.stdout
    def write(self, data):
        self._stdout.write(data); self._file.write(data); self._file.flush()
    def flush(self):
        self._stdout.flush(); self._file.flush()
    def close(self):
        self._file.close(); sys.stdout = self._stdout


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LOG_PATH = f"tpu/results/shors_33q_{TS}.txt"
    tee = Tee(LOG_PATH)
    sys.stdout = tee

    banner(f"Shor's Algorithm — 33-Qubit Full State Vector  │  TPU v5litepod-16  │  {TS}")

    # ── Circuit configuration ────────────────────────────────────────────────
    N_COUNTING = 22     # counting register qubits  (2^22 = 4,194,304 outcomes)
    N_WORK     = 11     # work register qubits      (holds values 0..2047 ≥ N)

    print_system_info(N_COUNTING, N_WORK)

    # ── Factoring runs ────────────────────────────────────────────────────────
    # (N, a) — a must be coprime to N, gcd(a,N)=1
    RUNS = [
        (15,  7),    # N=15=3×5,  a=7,  r=4
        (21,  2),    # N=21=3×7,  a=2,  r=6
        (35,  2),    # N=35=5×7,  a=2,  r=12
    ]

    all_results = []
    t_grand     = time.perf_counter()

    for N_val, a_val in RUNS:
        res = run_shor_factoring(N_val, a_val, N_COUNTING, N_WORK)
        all_results.append(res)

        # Per-run checkpoint JSON (no large probs array)
        chk = {k: v for k, v in res.items()
               if k not in ("probs", "phase_snapshots", "counting_phases")}
        chk_path = f"tpu/results/shors_{N_val}_{TS}.json"
        json.dump(chk, open(chk_path, "w"), indent=2)
        print(f"\n  📄 Checkpoint → {chk_path}")

    # ── Generate all plots ───────────────────────────────────────────────────
    banner("Generating Plots & Reports")

    for res in all_results:
        plot_spectrum(res, N_COUNTING, TS)
        plot_qft_analysis(res, N_COUNTING, TS)

    plot_circuit_analytics(all_results, N_COUNTING, N_WORK, TS)
    plot_master_dashboard(all_results, N_COUNTING, N_WORK, TS)
    plot_summary_card(all_results, N_COUNTING, N_WORK, TS)

    # ── Save CSV + full JSON ─────────────────────────────────────────────────
    save_results_csv(all_results, TS)

    json_path = f"tpu/results/shors_33q_{TS}.json"
    export = all_results.copy()
    for r in export:
        r.pop("probs", None)
        r.pop("phase_snapshots", None)
        r.pop("counting_phases", None)
    json.dump({"timestamp": TS, "backend": BACKEND, "n_devices": NUM_DEV,
               "n_counting": N_COUNTING, "n_work": N_WORK,
               "n_total_qubits": N_COUNTING + N_WORK,
               "state_vector_gb": (1 << (N_COUNTING + N_WORK)) * 8 / (1 << 30),
               "runs": export},
              open(json_path, "w"), indent=2)
    print(f"  📄 JSON results       → {json_path}")

    # ── Grand summary ────────────────────────────────────────────────────────
    grand_time = time.perf_counter() - t_grand
    banner(f"COMPLETE — {len(RUNS)} runs finished in {grand_time:.1f}s")

    print(f"\n  Results:")
    for res in all_results:
        p_, q_ = res.get("factor_p"), res.get("factor_q")
        status = f"✅  {p_} × {q_}" if (p_ and q_) else "⚠️  incomplete"
        print(f"    N={res['N']:4d}  a={res['a']:2d}  r={res['r_classical']:3d}  "
              f"→  {status}  ({res['timing']['total_s']:.2f}s)")

    print(f"\n  Plots  → tpu/plots/")
    print(f"  Data   → tpu/results/")
    print(f"  Log    → {LOG_PATH}")
    print(f"  Time   : {TS}\n")

    tee.close()
