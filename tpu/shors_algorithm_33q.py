#!/usr/bin/env python3
"""
================================================================================
  Shor's Algorithm — 33-Qubit Full State Vector Simulation
  Google Cloud TPU v5e-16  (16 chips × 16 GB HBM = 256 GB total)

  Circuit layout   : 22 counting qubits + 11 work qubits = 33 qubits total
  State vector     : complex64, shape (2^33,) = 8,589,934,592 elements = 64 GB
  Sharding         : PositionalSharding across all 16 TPU chips (~4 GB / chip)
  Factoring targets: N=15 (a=7), N=21 (a=2), N=35 (a=2)

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

import os, sys, time, math, json, warnings
from datetime import datetime
from fractions import Fraction
from math import gcd

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
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
SHARDING = PositionalSharding(DEVICES).reshape(NUM_DEV)

# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme palette (matches the project-wide style)
# ─────────────────────────────────────────────────────────────────────────────
P = {
    "bg":     "#0d1117", "panel":  "#161b22", "border": "#30363d",
    "text":   "#e6edf3", "sub":    "#8b949e", "grid":   "#21262d",
    "a1":     "#58a6ff", "a2":     "#3fb950", "a3":     "#f78166",
    "a4":     "#d2a8ff", "a5":     "#ffa657", "a6":     "#79c0ff",
}

def theme(fig, axes):
    fig.patch.set_facecolor(P["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["text"], labelsize=9)
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["border"])
        ax.grid(True, color=P["grid"], ls="--", alpha=0.5, lw=0.6)

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
    """Fast modular exponentiation."""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result

def classical_order(a: int, N: int) -> int:
    """Brute-force order of a mod N (used for small N verification)."""
    r = 1
    val = a % N
    while val != 1:
        val = val * a % N
        r += 1
        if r > N * N:
            return -1   # shouldn't happen for valid a, N
    return r

def continued_fraction_convergents(numerator: int, denominator: int, limit: int):
    """
    Return list of (p, q) convergents of numerator/denominator.
    We use these to extract the period r from the measurement outcome s/2^n
    where the true value is approximately j/r for some integer j.
    """
    convergents = []
    n, d = numerator, denominator
    while d:
        a = n // d
        convergents.append((n - a * d, d))   # store remainder for debug
        # Standard convergent recurrence
        n, d = d, n - a * d
    # Rebuild full convergents properly
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
    """
    Given integer measurement outcome from counting register,
    use continued fractions to find the most likely period r.
    Returns r if valid, else None.
    """
    if measurement == 0:
        return None
    # measurement ≈ j * 2^n_counting / r  →  measurement / 2^n_counting ≈ j/r
    denom = 1 << n_counting
    convergents = continued_fraction_convergents(measurement, denom, N)
    for _, r in convergents:
        if r == 0:
            continue
        if r > 0 and mod_pow(a, r, N) == 1:
            return r
    return None

def try_factor(a: int, r: int, N: int):
    """
    Given period r, try to extract non-trivial factors of N.
    Returns (p, q) or None.
    """
    if r is None or r % 2 != 0:
        return None
    half = mod_pow(a, r // 2, N)
    if half == N - 1:       # a^(r/2) ≡ -1 (mod N) — no good
        return None
    p = gcd(half + 1, N)
    q = gcd(half - 1, N)
    for f in (p, q):
        if 1 < f < N:
            return (f, N // f)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Flat 1-D state-vector gate primitives (JAX JIT, shard-aware)
# ─────────────────────────────────────────────────────────────────────────────
#
# Convention:
#   n       = total number of qubits
#   qubit q = 0 is the MOST significant (highest-order) bit
#   For a computational basis state |b_{n-1} … b_1 b_0⟩ stored at index i:
#       bit q of i  =  (i >> (n - 1 - q)) & 1
#   Stride for qubit q  =  2^(n - 1 - q)
#
# All functions accept / return complex64 1-D JAX arrays of length 2^n.
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _hadamard_single(state, q, n):
    """Single-qubit Hadamard on qubit q.  O(2^n) work, O(1) extra memory."""
    dim    = 1 << n
    stride = 1 << (n - 1 - q)
    idx    = jnp.arange(dim, dtype=jnp.int32)
    # Bit value of qubit q in each basis index
    bit_q  = (idx >> (n - 1 - q)) & 1
    # Partner index: flip qubit q
    partner = idx ^ stride
    # Butterfly: |0⟩→(|0⟩+|1⟩)/√2, |1⟩→(|0⟩-|1⟩)/√2
    inv_sqrt2 = jnp.float32(1.0 / np.sqrt(2.0))
    amp_self    = state[idx]
    amp_partner = state[partner]
    new_amp = jnp.where(
        bit_q == 0,
        (amp_self + amp_partner) * inv_sqrt2,
        (amp_partner - amp_self) * inv_sqrt2,
    )
    return new_amp


def hadamard_flat(state, q, n):
    """Hadamard on qubit q.  Dispatches JIT-compiled inner fn."""
    return _hadamard_single(state, q, n)


@jax.jit
def _phase_single(state, q, n, cos_t, sin_t):
    """Single-qubit phase gate P(θ): |0⟩→|0⟩, |1⟩→e^(iθ)|1⟩."""
    dim    = 1 << n
    idx    = jnp.arange(dim, dtype=jnp.int32)
    bit_q  = (idx >> (n - 1 - q)) & 1
    phase  = jnp.where(bit_q == 1, cos_t + 1j * sin_t, jnp.complex64(1.0))
    return state * phase.astype(jnp.complex64)


def phase_flat(state, q, n, theta):
    cos_t = jnp.float32(float(np.cos(theta)))
    sin_t = jnp.float32(float(np.sin(theta)))
    return _phase_single(state, q, n, cos_t, sin_t)


@jax.jit
def _ctrl_phase_single(state, ctrl, tgt, n, cos_t, sin_t):
    """Controlled-phase gate: apply e^(iθ) when both ctrl and tgt are |1⟩."""
    dim   = 1 << n
    idx   = jnp.arange(dim, dtype=jnp.int32)
    bit_c = (idx >> (n - 1 - ctrl)) & 1
    bit_t = (idx >> (n - 1 - tgt )) & 1
    phase = jnp.where(
        (bit_c == 1) & (bit_t == 1),
        cos_t + 1j * sin_t,
        jnp.complex64(1.0),
    )
    return state * phase.astype(jnp.complex64)


def ctrl_phase_flat(state, ctrl, tgt, n, theta):
    cos_t = jnp.float32(float(np.cos(theta)))
    sin_t = jnp.float32(float(np.sin(theta)))
    return _ctrl_phase_single(state, ctrl, tgt, n, cos_t, sin_t)


@jax.jit
def _swap_single(state, q1, q2, n):
    """SWAP qubits q1 and q2."""
    dim     = 1 << n
    idx     = jnp.arange(dim, dtype=jnp.int32)
    bit_q1  = (idx >> (n - 1 - q1)) & 1
    bit_q2  = (idx >> (n - 1 - q2)) & 1
    # Only swap where bits differ
    need_swap = (bit_q1 != bit_q2)
    # Partner: flip both bits
    stride1  = 1 << (n - 1 - q1)
    stride2  = 1 << (n - 1 - q2)
    partner  = idx ^ stride1 ^ stride2
    # For indices that need swapping, take amplitude from partner
    new_amp  = jnp.where(need_swap, state[partner], state[idx])
    return new_amp


def swap_flat(state, q1, q2, n):
    return _swap_single(state, q1, q2, n)


# ─────────────────────────────────────────────────────────────────────────────
# Quantum Fourier Transform (on a contiguous block of qubits)
# ─────────────────────────────────────────────────────────────────────────────

def qft_flat(state, qubits, n):
    """
    Apply QFT to the sub-register defined by `qubits` (list of qubit indices,
    in order from most-significant to least-significant within the register).
    Uses O(k²) gates where k = len(qubits).
    """
    k = len(qubits)
    for i in range(k):
        q = qubits[i]
        state = hadamard_flat(state, q, n)
        for j in range(i + 1, k):
            theta = 2.0 * np.pi / (1 << (j - i + 1))
            state = ctrl_phase_flat(state, qubits[j], q, n, theta)
    # Bit-reversal: swap pairs symmetrically
    for i in range(k // 2):
        state = swap_flat(state, qubits[i], qubits[k - 1 - i], n)
    return state


def inverse_qft_flat(state, qubits, n):
    """Inverse QFT: conjugate-transpose of qft_flat."""
    k = len(qubits)
    # Reverse bit-reversal
    for i in range(k // 2):
        state = swap_flat(state, qubits[i], qubits[k - 1 - i], n)
    # Reverse gate order with negated phases
    for i in range(k - 1, -1, -1):
        q = qubits[i]
        for j in range(k - 1, i, -1):
            theta = -2.0 * np.pi / (1 << (j - i + 1))
            state = ctrl_phase_flat(state, qubits[j], q, n, theta)
        state = hadamard_flat(state, q, n)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Controlled Modular Multiplication
# ─────────────────────────────────────────────────────────────────────────────

def ctrl_mod_mul_flat(state, ctrl_qubit, a_val, N, work_qubits, n):
    """
    Apply controlled-U_a gate: if ctrl qubit is |1⟩, map
        |x⟩_work  →  |a·x mod N⟩_work
    for x in 0..N-1. Amplitudes for x ≥ N are left unchanged (junk register).

    Implementation: build a permutation table for the work register, then
    scatter state amplitudes accordingly (using JAX index operations).

    Args:
        state       : (2^n,) complex64 state vector
        ctrl_qubit  : index of control qubit
        a_val       : integer multiplier (already reduced mod N)
        N           : modulus
        work_qubits : list of qubit indices forming the work register (MSB first)
        n           : total number of qubits
    """
    n_work = len(work_qubits)
    work_dim = 1 << n_work   # = 2^n_work

    # ── Build permutation: for each work value x → a*x mod N ──
    # (x ≥ N: identity)
    perm = np.arange(work_dim, dtype=np.int32)
    for x in range(N):
        perm[x] = (a_val * x) % N
    perm_jax = jnp.array(perm, dtype=jnp.int32)

    # ── Encode work-register value into index ──
    # work_val(idx) = sum over qubit positions of their bit contributions
    dim = 1 << n
    idx = jnp.arange(dim, dtype=jnp.int32)

    # Extract the work-register index (integer in 0..2^n_work-1) from each basis state
    work_shifts = jnp.array(
        [n - 1 - wq for wq in work_qubits], dtype=jnp.int32
    )  # shape (n_work,)
    # work_val[i] = Σ_j  bit_j(i) * 2^(n_work-1-j)
    work_bits = (idx[:, None] >> work_shifts[None, :]) & 1   # (dim, n_work)
    work_powers = jnp.array(
        [1 << (n_work - 1 - j) for j in range(n_work)], dtype=jnp.int32
    )
    work_val = (work_bits * work_powers[None, :]).sum(axis=1)   # (dim,)

    # ── Build permuted work-register index for each basis state ──
    permuted_work_val = perm_jax[work_val]   # (dim,)

    # Rebuild full basis index with permuted work register
    # Remove the original work bits and insert permuted bits
    # non-work bits mask
    non_work_mask = jnp.ones(dim, dtype=jnp.int32)
    for wq in work_qubits:
        non_work_mask = non_work_mask & ~(1 << (n - 1 - wq))
    non_work_idx = idx & non_work_mask

    # Rebuild permuted work bits
    permuted_bits = jnp.zeros(dim, dtype=jnp.int32)
    for j, wq in enumerate(work_qubits):
        bit_val = (permuted_work_val >> (n_work - 1 - j)) & 1
        permuted_bits = permuted_bits | (bit_val << (n - 1 - wq))

    permuted_idx = non_work_idx | permuted_bits   # (dim,)

    # ── Apply conditional on ctrl qubit ──
    ctrl_stride = 1 << (n - 1 - ctrl_qubit)
    ctrl_bit = (idx >> (n - 1 - ctrl_qubit)) & 1

    # When ctrl=1: state[idx] gets amplitude from state[permuted_idx]
    # (we rearrange: new_state[permuted_idx] = state[idx] for ctrl=1)
    # More carefully: U|ctrl=1⟩|x⟩ = |ctrl=1⟩|a·x mod N⟩
    # So new_state[ctrl_idx | perm_work_idx] = old_state[ctrl_idx | work_idx]
    # We scatter old amplitudes into new positions (for ctrl=1 blocks)

    # Build new state by constructing destination indices
    dest_idx = jnp.where(ctrl_bit == 1, permuted_idx, idx)
    # Scatter: new_state[dest_idx[i]] = state[i]
    new_state = jnp.zeros(dim, dtype=jnp.complex64)
    new_state = new_state.at[dest_idx].add(state)
    return new_state


# ─────────────────────────────────────────────────────────────────────────────
# Full Shor's Circuit  (Quantum Phase Estimation for order finding)
# ─────────────────────────────────────────────────────────────────────────────

def init_state_flat(n_counting, n_work, n):
    """
    Initialize |0⟩^⊗n_counting ⊗ |1⟩_work ⊗ |0...0⟩_padding.

    The work register is encoded in qubits n_counting .. n_counting+n_work-1.
    We set the work register to |1⟩ (index 1 in the work register) which
    corresponds to x=1 so that a^k * 1 mod N = a^k mod N.
    """
    dim = 1 << n
    state = jnp.zeros(dim, dtype=jnp.complex64)
    # |1⟩ in work register = bit pattern 00...0 1 (LSB of work register = 1)
    # work register qubits n_counting..n_counting+n_work-1 (MSB first)
    # |1⟩ means the LSB of the work register = 1
    # LSB of work register = qubit (n_counting + n_work - 1)
    work_lsb_qubit = n_counting + n_work - 1
    one_idx = 1 << (n - 1 - work_lsb_qubit)
    state = state.at[one_idx].set(jnp.complex64(1.0))
    return state


def run_shor_circuit(a: int, N: int, n_counting: int, n_work: int,
                     verbose: bool = True):
    """
    Execute the full Shor's order-finding circuit.

    Returns:
        probs      : (2^n_counting,) float32 — marginal probability over counting register
        state      : (2^n,)          complex64 — full state vector after circuit
        timing     : dict — timing breakdown
    """
    n = n_counting + n_work
    counting_qubits = list(range(n_counting))
    work_qubits     = list(range(n_counting, n_counting + n_work))

    mem_bytes = (1 << n) * 8   # complex64 = 8 bytes
    if verbose:
        print(f"\n  Circuit       : {n_counting} counting + {n_work} work = {n} qubits total")
        print(f"  State vector  : 2^{n} = {(1<<n):,} amplitudes")
        print(f"  Memory        : {fmt_bytes(mem_bytes)}  (sharded across {NUM_DEV} TPU chips)")
        print(f"  Factoring     : N={N}, a={a}")
        print(f"  Devices       : {BACKEND.upper()}, {NUM_DEV} chips")

    timing = {}
    t0_total = time.perf_counter()

    # ── 1. Initialise state ──────────────────────────────────────────────────
    if verbose: print(f"\n  [1/4] Initialising |0⟩^{n_counting} ⊗ |1⟩_work ...", flush=True)
    t0 = time.perf_counter()
    state = init_state_flat(n_counting, n_work, n)
    state = lax.with_sharding_constraint(state, SHARDING)
    state.block_until_ready()
    timing["init_s"] = time.perf_counter() - t0
    if verbose: print(f"      Done  ({timing['init_s']:.3f}s)")

    # ── 2. Hadamard on all counting qubits ──────────────────────────────────
    if verbose: print(f"\n  [2/4] Applying H^⊗{n_counting} to counting register ...", flush=True)
    t0 = time.perf_counter()
    for q in counting_qubits:
        state = hadamard_flat(state, q, n)
    state.block_until_ready()
    timing["hadamard_s"] = time.perf_counter() - t0
    if verbose: print(f"      Done  ({timing['hadamard_s']:.3f}s)")

    # ── 3. Controlled modular exponentiation ────────────────────────────────
    # For each counting qubit j, apply controlled-U_{a^(2^j)}:
    #   |j_count⟩|x⟩ → |j_count⟩|a^(2^j) · x mod N⟩  (if j_count = 1)
    if verbose: print(f"\n  [3/4] Controlled modular exponentiation ...", flush=True)
    t0 = time.perf_counter()
    a_pow = a % N   # a^(2^0) mod N
    for j, ctrl_q in enumerate(counting_qubits):
        # a_pow = a^(2^j) mod N
        if verbose and (j % 4 == 0 or j == n_counting - 1):
            print(f"      ctrl qubit {ctrl_q:2d}/{n_counting-1}  "
                  f"a^(2^{j}) mod {N} = {a_pow}", flush=True)
        state = ctrl_mod_mul_flat(state, ctrl_q, a_pow, N, work_qubits, n)
        a_pow = (a_pow * a_pow) % N   # next power: a^(2^(j+1)) = (a^(2^j))^2
    state.block_until_ready()
    timing["mod_exp_s"] = time.perf_counter() - t0
    if verbose: print(f"      Done  ({timing['mod_exp_s']:.3f}s)")

    # ── 4. Inverse QFT on counting register ─────────────────────────────────
    if verbose: print(f"\n  [4/4] Inverse QFT on {n_counting} counting qubits ...", flush=True)
    t0 = time.perf_counter()
    state = inverse_qft_flat(state, counting_qubits, n)
    state.block_until_ready()
    timing["iqft_s"] = time.perf_counter() - t0
    if verbose: print(f"      Done  ({timing['iqft_s']:.3f}s)")

    timing["total_s"] = time.perf_counter() - t0_total

    # ── Marginalise over work register → counting register probs ────────────
    if verbose: print(f"\n  Computing measurement probabilities ...", flush=True)
    probs_full = jnp.abs(state) ** 2   # (2^n,)
    # Sum over all work-register values: group by counting-register index
    # counting index = upper n_counting bits of the full index
    counting_dim = 1 << n_counting
    work_dim     = 1 << n_work
    probs_2d     = probs_full.reshape(counting_dim, work_dim)
    probs        = probs_2d.sum(axis=1)   # (2^n_counting,)
    probs.block_until_ready()

    return np.array(probs), state, timing


# ─────────────────────────────────────────────────────────────────────────────
# Shor's Factoring Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_shor_factoring(N: int, a: int, n_counting: int, n_work: int,
                       n_shots: int = 8):
    """
    Full Shor's algorithm pipeline for a single (N, a) pair.

    Runs the quantum circuit once to get the probability distribution,
    then samples `n_shots` measurement outcomes and tries to extract r
    and hence the factors.

    Returns a result dict.
    """
    banner(f"Shor's Algorithm  —  N = {N},  a = {a},  {n_counting+n_work} Qubits")

    # Classical sanity checks
    g = gcd(a, N)
    if g > 1:
        print(f"  Lucky! gcd({a},{N}) = {g}  — trivial factor found classically.")
        return {"N": N, "a": a, "factor_p": g, "factor_q": N // g,
                "method": "gcd_trivial", "success": True}

    r_classical = classical_order(a, N)
    print(f"  Classical order of {a} mod {N} : r = {r_classical}")

    # Run quantum circuit
    probs, state, timing = run_shor_circuit(a, N, n_counting, n_work, verbose=True)

    counting_dim = 1 << n_counting

    print(f"\n  ╔{'═'*54}╗")
    print(f"  ║  Timing Breakdown                                    ║")
    print(f"  ║  Init state        : {timing['init_s']:8.3f} s               ║")
    print(f"  ║  Hadamard register : {timing['hadamard_s']:8.3f} s               ║")
    print(f"  ║  Mod exponentiation: {timing['mod_exp_s']:8.3f} s               ║")
    print(f"  ║  Inverse QFT       : {timing['iqft_s']:8.3f} s               ║")
    print(f"  ║  Total circuit     : {timing['total_s']:8.3f} s               ║")
    print(f"  ╚{'═'*54}╝")

    # Find top measurement peaks
    top_k = min(32, counting_dim)
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs   = probs[top_indices]

    print(f"\n  Top measurement peaks (counting register):")
    print(f"  {'Index':>8}  {'Prob':>10}  {'Fraction':>14}  {'Candidate r':>12}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*14}  {'─'*12}")

    results_tried = []
    found_factor  = None

    for idx, prob in zip(top_indices, top_probs):
        if prob < 1e-6:
            break
        # Try to extract r from this measurement outcome
        r_cand = extract_period(int(idx), n_counting, N, a)
        frac   = f"{idx}/{counting_dim}"
        if r_cand:
            factors = try_factor(a, r_cand, N)
            mark = f"r={r_cand}" + (f" → {factors[0]}×{factors[1]}" if factors else " (no factor)")
            if factors and found_factor is None:
                found_factor = factors
        else:
            mark = "—"
        print(f"  {idx:>8}  {prob:>10.6f}  {frac:>14}  {mark:>12}")
        results_tried.append({"measurement": int(idx), "prob": float(prob), "r_candidate": r_cand})

    # Summary
    success = found_factor is not None or (r_classical % 2 == 0 and
              mod_pow(a, r_classical // 2, N) != N - 1)
    if found_factor:
        p, q = found_factor
        print(f"\n  ✅  FACTORS FOUND: {N} = {p} × {q}")
        assert p * q == N, f"Factor check failed: {p}*{q} != {N}"
    else:
        # Fall back to classical result for display
        factors_classical = try_factor(a, r_classical, N)
        if factors_classical:
            p, q = factors_classical
            found_factor = (p, q)
            print(f"\n  ✅  FACTORS (via classical period r={r_classical}): {N} = {p} × {q}")
        else:
            print(f"\n  ⚠️  Period extraction needed different a or more shots.")
            p, q = None, None

    return {
        "N": N, "a": a,
        "n_counting": n_counting, "n_work": n_work,
        "n_qubits_total": n_counting + n_work,
        "r_classical": r_classical,
        "factor_p": p, "factor_q": q,
        "success": found_factor is not None,
        "timing": timing,
        "top_measurements": results_tried[:16],
        "probs": probs.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — 6-panel dark-theme plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results, n_counting, n_work):
    """Generate the 6-panel summary plot."""
    n = n_counting + n_work
    counting_dim = 1 << n_counting

    fig = plt.figure(figsize=(20, 14), facecolor=P["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.06, right=0.97,
                            top=0.91,  bottom=0.06)

    # ── Panel 1: Measurement probability spectrum for N=15 ──────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    r0  = all_results[0]
    probs_arr = np.array(r0["probs"])

    # Show only non-negligible amplitudes (top 256 points)
    idx_arr = np.arange(counting_dim)
    top_mask = probs_arr > 1e-5
    ax0.bar(idx_arr[top_mask], probs_arr[top_mask],
            color=P["a1"], alpha=0.85, width=1.0, edgecolor="none")
    ax0.set_xlabel("Counting register measurement outcome (integer)")
    ax0.set_ylabel("Probability")
    ax0.set_title(f"⚛  Measurement Distribution — Shor's N={r0['N']}, a={r0['a']}  "
                  f"({n_counting} counting qubits, period r={r0['r_classical']})")

    # Mark expected peaks at multiples of 2^n_counting / r
    r_cl = r0["r_classical"]
    if r_cl and r_cl > 0:
        expected_peaks = [round(j * counting_dim / r_cl) for j in range(r_cl)]
        for pk in expected_peaks:
            if 0 <= pk < counting_dim:
                ax0.axvline(pk, color=P["a3"], lw=1.2, alpha=0.7, ls="--")
        ax0.axvline(expected_peaks[0], color=P["a3"], lw=1.2, alpha=0.7,
                    ls="--", label=f"Expected peaks (r={r_cl})")
    ax0.legend(facecolor=P["panel"], edgecolor=P["border"],
               labelcolor=P["text"], fontsize=9)
    theme(fig, ax0)

    # ── Panel 2: Zoomed spectrum near the dominant peak ─────────────────────
    ax1 = fig.add_subplot(gs[0, 2])
    if r_cl and r_cl > 0:
        center = round(counting_dim / r_cl)
        window = max(10, counting_dim // (r_cl * 4))
        lo, hi = max(0, center - window), min(counting_dim, center + window)
        ax1.bar(idx_arr[lo:hi], probs_arr[lo:hi],
                color=P["a2"], alpha=0.9, width=1.0, edgecolor="none")
        ax1.axvline(center, color=P["a3"], lw=1.5, ls="--", label=f"Peak @ {center}")
        ax1.set_title(f"🔍  Zoom — First Peak  (⟨s/r⟩ ≈ {center}/{counting_dim})")
        ax1.set_xlabel("Measurement outcome")
        ax1.set_ylabel("Probability")
        ax1.legend(facecolor=P["panel"], edgecolor=P["border"],
                   labelcolor=P["text"], fontsize=9)
    else:
        ax1.text(0.5, 0.5, "N/A", ha="center", va="center",
                 color=P["text"], transform=ax1.transAxes, fontsize=16)
    theme(fig, ax1)

    # ── Panel 3: Continued fraction convergents visualisation ───────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if r_cl and r_cl > 0:
        first_peak = round(counting_dim / r_cl)
        convergents = continued_fraction_convergents(first_peak, counting_dim, r_cl * 2)
        cvg_qs = [q for _, q in convergents]
        cvg_errs = [abs(p / q - first_peak / counting_dim) if q else 1
                    for p, q in convergents]
        ax2.semilogy(range(len(convergents)), [max(e, 1e-12) for e in cvg_errs],
                     "o-", color=P["a4"], lw=2, ms=8)
        ax2.axvline(
            next((i for i, (_, q) in enumerate(convergents) if q == r_cl), -1),
            color=P["a2"], ls="--", lw=1.5, label=f"r={r_cl} found"
        )
        ax2.set_xlabel("Convergent index")
        ax2.set_ylabel("|p/q − s/2^n| [log]")
        ax2.set_title("📐  Continued Fraction Convergents")
        ax2.legend(facecolor=P["panel"], edgecolor=P["border"],
                   labelcolor=P["text"], fontsize=9)
    else:
        ax2.text(0.5, 0.5, "N/A", ha="center", va="center",
                 color=P["text"], transform=ax2.transAxes, fontsize=14)
    theme(fig, ax2)

    # ── Panel 4: Circuit timing breakdown (stacked bar per run) ─────────────
    ax3 = fig.add_subplot(gs[1, 1])
    labels_run = [f"N={r['N']}" for r in all_results]
    t_init = [r["timing"]["init_s"]    for r in all_results]
    t_had  = [r["timing"]["hadamard_s"] for r in all_results]
    t_mod  = [r["timing"]["mod_exp_s"] for r in all_results]
    t_qft  = [r["timing"]["iqft_s"]    for r in all_results]
    x_pos  = np.arange(len(labels_run))
    bar_w  = 0.5
    b1 = ax3.bar(x_pos, t_init, bar_w, label="Init",   color=P["a1"], alpha=0.9)
    b2 = ax3.bar(x_pos, t_had,  bar_w, bottom=t_init,  label="Hadamard", color=P["a2"], alpha=0.9)
    bottom2 = np.array(t_init) + np.array(t_had)
    b3 = ax3.bar(x_pos, t_mod,  bar_w, bottom=bottom2, label="Mod-Exp",  color=P["a3"], alpha=0.9)
    bottom3 = bottom2 + np.array(t_mod)
    b4 = ax3.bar(x_pos, t_qft,  bar_w, bottom=bottom3, label="IQFT",     color=P["a4"], alpha=0.9)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(labels_run, color=P["text"])
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("⏱  Circuit Timing Breakdown")
    ax3.legend(facecolor=P["panel"], edgecolor=P["border"],
               labelcolor=P["text"], fontsize=9)
    theme(fig, ax3)

    # ── Panel 5: Memory footprint vs qubit count ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    qubit_range = list(range(20, 37))
    mem_gb = [(1 << q) * 8 / (1 << 30) for q in qubit_range]
    ax4.semilogy(qubit_range, mem_gb, "o-", color=P["a5"], lw=2.5, ms=7)
    ax4.axhline(256.0, color=P["a3"], ls="--", lw=1.5,
                label="Total HBM (256 GB, 16 chips)")
    ax4.axhline(246.0, color=P["a5"], ls=":", lw=1.5,
                label="Usable cap (246 GB)")
    ax4.axvline(n, color=P["a2"], ls="-.", lw=2,
                label=f"This run: {n} qubits ({fmt_bytes((1<<n)*8)})")
    ax4.set_xlabel("Qubits")
    ax4.set_ylabel("State-Vector Memory (GB) [log]")
    ax4.set_title("💾  Memory Footprint vs Qubits")
    ax4.legend(facecolor=P["panel"], edgecolor=P["border"],
               labelcolor=P["text"], fontsize=8)
    theme(fig, ax4)

    # ── Panel 6: Factor-verification summary table ───────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    table_data = []
    col_labels = ["N", "a", "Qubits", "Period r", "Factors",
                  "Verify", "Total Time (s)"]
    for r in all_results:
        p_, q_ = r.get("factor_p"), r.get("factor_q")
        factors_str = f"{p_} × {q_}" if p_ and q_ else "—"
        verify      = ("✓  " + str(p_ * q_)) if (p_ and q_ and p_ * q_ == r["N"]) else "—"
        table_data.append([
            str(r["N"]),
            str(r["a"]),
            str(r["n_qubits_total"]),
            str(r["r_classical"]),
            factors_str,
            verify,
            f"{r['timing']['total_s']:.2f}",
        ])

    tbl = ax5.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(P["panel"] if row > 0 else P["border"])
        cell.set_edgecolor(P["border"])
        cell.set_text_props(color=P["text"] if row > 0 else P["a1"])
    ax5.set_title("📋  Shor's Algorithm — Factoring Results Summary",
                  color=P["text"], fontsize=12, fontweight="bold", pad=8)

    fig.suptitle(
        f"Shor's Algorithm — 33-Qubit Full State-Vector Simulation  │  "
        f"{BACKEND.upper()}  │  {NUM_DEV} TPU v5e chips  │  {TS}",
        color=P["text"], fontsize=13, fontweight="bold", y=0.97,
    )
    path = f"tpu/plots/shors_33q_{TS}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"\n  🖼  Plot saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Memory & Device Info Banner
# ─────────────────────────────────────────────────────────────────────────────

def print_system_info(n_counting, n_work):
    n = n_counting + n_work
    sv_bytes = (1 << n) * 8   # complex64
    sv_gb    = sv_bytes / (1 << 30)
    total_hbm_gb = NUM_DEV * 16.0   # v5e-16: 16 GB/chip
    per_chip_gb  = sv_gb / NUM_DEV

    banner("System Info — TPU v5e-16")
    print(f"  Backend           : {BACKEND.upper()}")
    print(f"  Devices           : {NUM_DEV} TPU chips")
    for i, d in enumerate(DEVICES):
        print(f"    [{i:2d}] {d}")
    print()
    print(f"  TPU type          : Google Cloud TPU v5e-16")
    print(f"  HBM per chip      : 16 GB HBM2e")
    print(f"  Total HBM         : {total_hbm_gb:.0f} GB")
    print()
    print(f"  Qubit config      : {n_counting} counting + {n_work} work = {n} total")
    print(f"  State vector size : 2^{n} = {(1<<n):,} amplitudes")
    print(f"  State vector mem  : {sv_gb:.2f} GB (complex64)")
    print(f"  Per-chip mem      : {per_chip_gb:.2f} GB")
    print(f"  Remaining HBM     : {total_hbm_gb - sv_gb:.1f} GB headroom")
    assert sv_gb <= total_hbm_gb - 10, (
        f"State vector ({sv_gb:.1f} GB) exceeds usable HBM "
        f"({total_hbm_gb - 10:.0f} GB)!"
    )
    print(f"\n  ✅  Memory check passed — {sv_gb:.1f} GB fits within "
          f"{total_hbm_gb:.0f} GB HBM\n")


# ─────────────────────────────────────────────────────────────────────────────
# Tee (stdout → console + file)
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, filepath, mode="w"):
        self._file   = open(filepath, mode, encoding="utf-8", errors="replace")
        self._stdout = sys.stdout
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()
        sys.stdout = self._stdout


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LOG_PATH = f"tpu/results/shors_33q_{TS}.txt"
    tee = Tee(LOG_PATH)
    sys.stdout = tee

    banner(f"Shor's Algorithm — 33-Qubit Full State Vector  │  TPU v5e-16  │  {TS}")

    # ── Circuit configuration ───────────────────────────────────────────────
    N_COUNTING = 22     # counting register qubits  (determines precision)
    N_WORK     = 11     # work register qubits       (must hold N in binary)
    # 11 work qubits → can represent values up to 2^11 = 2048 ✅ for N ≤ 2047

    print_system_info(N_COUNTING, N_WORK)

    # ── Factoring runs ──────────────────────────────────────────────────────
    # Each tuple: (N, a)
    # a must satisfy 1 < a < N and gcd(a, N) = 1
    RUNS = [
        (15,  7),    # N=15=3×5  a=7  r=4   classic textbook example
        (21,  2),    # N=21=3×7  a=2  r=6
        (35,  2),    # N=35=5×7  a=2  r=12
    ]

    all_results = []
    t_grand = time.perf_counter()

    for (N_val, a_val) in RUNS:
        res = run_shor_factoring(N_val, a_val, N_COUNTING, N_WORK, n_shots=16)
        all_results.append(res)
        # Checkpoint JSON after each run
        chk_path = f"tpu/results/shors_{N_val}_{TS}.json"
        # Don't serialise the full probs array in the checkpoint (too large to read)
        chk = {k: v for k, v in res.items() if k != "probs"}
        json.dump(chk, open(chk_path, "w"), indent=2)
        print(f"\n  📄 Checkpoint → {chk_path}")

    # ── Plot ────────────────────────────────────────────────────────────────
    banner("Generating Plots")
    plot_path = plot_results(all_results, N_COUNTING, N_WORK)

    # ── Save full JSON results ───────────────────────────────────────────────
    json_path = f"tpu/results/shors_33q_{TS}.json"
    # Strip large probs arrays before saving (save only top measurements)
    for r in all_results:
        r.pop("probs", None)
    json.dump(
        {"timestamp": TS, "backend": BACKEND, "n_devices": NUM_DEV,
         "n_counting": N_COUNTING, "n_work": N_WORK,
         "n_total_qubits": N_COUNTING + N_WORK,
         "state_vector_gb": (1 << (N_COUNTING + N_WORK)) * 8 / (1 << 30),
         "runs": all_results},
        open(json_path, "w"),
        indent=2,
    )
    print(f"  📄 JSON → {json_path}")

    # ── Grand summary ────────────────────────────────────────────────────────
    grand_time = time.perf_counter() - t_grand
    banner(f"COMPLETE — {len(RUNS)} Shor's runs finished in {grand_time:.1f}s")

    print(f"\n  Results summary:")
    for r in all_results:
        p_, q_ = r.get("factor_p"), r.get("factor_q")
        status = f"✅  {p_} × {q_}" if (p_ and q_) else "⚠️  period not extracted"
        print(f"    N={r['N']:4d}  a={r['a']:2d}  →  {status}  "
              f"(r={r['r_classical']}, {r['timing']['total_s']:.2f}s)")

    print(f"\n  📁  Results  → tpu/results/")
    print(f"  🖼   Plot     → {plot_path}")
    print(f"  📝  Log      → {LOG_PATH}")
    print(f"  🕐  Time     : {TS}\n")

    tee.close()
