#!/usr/bin/env python3
"""
================================================================================
  JAX Quantum Research — 6 Examples Runner  (TPU-Ready, Self-Contained)
  -----------------------------------------------------------------------
  Exactly mirrors the 6 example files in examples/ but without jax_qsim.
  Every quantum primitive is implemented inline with pure JAX.

  Experiments (in order):
    1. State Preparation  — GHZ 3-qubit  (01_state_preparation.py)
    2. VQC Classifier     — XOR problem   (02_vqc_classification.py)
    3. VQE H2 Molecule    — Ground state  (04_vqe_h2_molecule.py)
    4. QAOA MaxCut        — 6-node graph  (05_qaoa_maxcut.py)
    5. Noise Simulation   — Monte Carlo   (04_quantum_noise_simulation.py)
    6. Barren Plateaus    — Width & depth (06_barren_plateaus.py)

  Usage (on TPU VM):
    source ~/tpu_env/bin/activate
    cd ~/jax-quantum-research
    python3 tpu_examples_runner.py 2>&1 | tee results/examples_run_$(date +%Y%m%d_%H%M%S).txt
================================================================================
"""

import os, sys, time, json, warnings
from datetime import datetime
from functools import partial

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp

warnings.filterwarnings("ignore", category=jnp.ComplexWarning)
warnings.filterwarnings("ignore", message="Casting complex values")
warnings.filterwarnings("ignore", message="Glyph")
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Global setup
# ─────────────────────────────────────────────────────────────────────────────
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKEND = jax.default_backend()
DEVICES = jax.devices()

os.makedirs("results",        exist_ok=True)
os.makedirs("examples/plots", exist_ok=True)

# Dark theme palette (shared by all plots)
P = {
    "bg":     "#0d1117", "panel":  "#161b22", "border": "#30363d",
    "text":   "#e6edf3", "sub":    "#8b949e", "a1":     "#58a6ff",
    "a2":     "#3fb950", "a3":     "#f78166", "a4":     "#d2a8ff",
    "a5":     "#ffa657", "grid":   "#21262d",
}

def theme(fig, axes):
    fig.patch.set_facecolor(P["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["text"], labelsize=10)
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(P["border"])
        ax.grid(True, color=P["grid"], ls="--", alpha=0.6, lw=0.7)

def banner(title):
    w = 78
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)

# ─────────────────────────────────────────────────────────────────────────────
# Pure-JAX Quantum Simulator Primitives  (no jax_qsim)
# ─────────────────────────────────────────────────────────────────────────────

def zero_state(n):
    """Return |0>^n as complex64 tensor of shape (2,)*n."""
    s = jnp.zeros((2,) * n, dtype=jnp.complex64)
    return s.at[(0,) * n].set(1.0)

def apply_1q(state, gate, t, n):
    """Apply a 2x2 gate to qubit t of an n-qubit state tensor."""
    gate = gate.astype(jnp.complex64)
    out  = jnp.tensordot(gate, state, axes=((1,), (t,)))
    axes = list(range(1, n)); axes.insert(t, 0)
    return jnp.transpose(out, axes)

_CNOT_T = jnp.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=jnp.complex64,
).reshape(2, 2, 2, 2)

def apply_cnot(state, c, t, n):
    """Apply CNOT gate with control c, target t."""
    out  = jnp.tensordot(_CNOT_T, state, axes=((2, 3), (c, t)))
    dest = [None] * n
    dest[c] = 0; dest[t] = 1
    k = 2
    for i in range(n):
        if dest[i] is None:
            dest[i] = k; k += 1
    return jnp.transpose(out, dest)

# Gate constructors
def Hgate():   return jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2.0)
def Xgate():   return jnp.array([[0, 1], [1, 0]],  dtype=jnp.complex64)
def RX(t):     c = jnp.cos(t / 2); s = -1j * jnp.sin(t / 2); return jnp.array([[c, s], [s, c]], dtype=jnp.complex64)
def RY(t):     c = jnp.cos(t / 2); s = jnp.sin(t / 2);       return jnp.array([[c, -s], [s, c]], dtype=jnp.complex64)
def RZ(t):     e = jnp.exp(-1j * t / 2); return jnp.array([[e, 0], [0, jnp.conj(e)]], dtype=jnp.complex64)

def state_flat(state):
    """Flatten state tensor to 1-D vector."""
    return state.reshape(-1)

def pauli_z_expect(state, qubit, n):
    """<Z_qubit> of n-qubit state tensor."""
    probs  = jnp.abs(state) ** 2
    axes   = tuple(i for i in range(n) if i != qubit)
    marg   = probs.sum(axis=axes)          # shape (2,)
    return marg[0] - marg[1]              # P(0) - P(1)

def pauli_x_expect(state, qubit, n):
    """<X_qubit> via basis rotation."""
    h = Hgate()
    s_rot = apply_1q(state, h, qubit, n)
    return pauli_z_expect(s_rot, qubit, n)

def hamiltonian_expect(state, terms, n):
    """
    Compute <H> for a sum of Pauli string terms.
    terms: list of (coeff, list_of_(qubit, op) pairs)
    op is 'I', 'X', 'Y', 'Z'
    """
    total = 0.0
    for coeff, ops_list in terms:
        if not ops_list:               # identity
            total = total + coeff
            continue
        s = state
        for q, op in ops_list:
            if op == "X":
                s = apply_1q(s, Hgate(), q, n)
            elif op == "Y":
                # <Y> = i * <XZ> — measure in Y basis: rotate by S†H
                sdg = jnp.array([[1, 0], [0, -1j]], dtype=jnp.complex64)
                s   = apply_1q(s, jnp.matmul(Hgate(), sdg), q, n)
            # Z: no rotation needed
        # Measure <Z_0 Z_1 ...> of all relevant qubits
        qubit_list = [q for q, _ in ops_list]
        ev = _multi_z_expect(s, qubit_list, n)
        total = total + coeff * ev
    return jnp.real(total)

def _multi_z_expect(state, qubits, n):
    """<prod Z_q for q in qubits> of state tensor."""
    probs = jnp.abs(state) ** 2          # shape (2,)*n
    # Flatten to (2^n,), loop over bitstrings
    flat  = probs.reshape(-1)
    size  = 2 ** n
    vals  = jnp.zeros(size, dtype=jnp.float32)
    # Build sign vector: +1 if even number of qubits in |1>, -1 otherwise
    idxs  = jnp.arange(size)
    sign  = jnp.ones(size, dtype=jnp.float32)
    for q in qubits:
        bit_q = (idxs >> (n - 1 - q)) & 1   # bit at qubit q
        sign  = sign * (1 - 2 * bit_q.astype(jnp.float32))
    return jnp.sum(flat * sign)

def adam_update(params, grads, m, v, t, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    t   = t + 1
    m   = b1 * m + (1 - b1) * grads
    v   = b2 * v + (1 - b2) * grads ** 2
    mh  = m / (1 - b1 ** t)
    vh  = v / (1 - b2 ** t)
    return params - lr * mh / (jnp.sqrt(vh) + eps), m, v, t


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 1 — GHZ State Preparation
#  Mirrors: examples/01_state_preparation.py
# ─────────────────────────────────────────────────────────────────────────────

def run_state_prep():
    banner("Experiment 1 — GHZ State Preparation  (01_state_preparation.py)")
    N = 3

    target = jnp.zeros(2 ** N, dtype=jnp.complex64)
    target = target.at[0].set(1.0 / jnp.sqrt(2.0))
    target = target.at[7].set(1.0 / jnp.sqrt(2.0))

    # Circuit: 3 layers of [RX, RY, RZ] + CNOT(0,1) + CNOT(1,2)  (9 params)
    def circuit(params):
        s = zero_state(N)
        # Layer 1
        s = apply_1q(s, RX(params[0]), 0, N)
        s = apply_1q(s, RY(params[1]), 1, N)
        s = apply_1q(s, RZ(params[2]), 2, N)
        s = apply_cnot(s, 0, 1, N)
        s = apply_cnot(s, 1, 2, N)
        # Layer 2
        s = apply_1q(s, RX(params[3]), 0, N)
        s = apply_1q(s, RY(params[4]), 1, N)
        s = apply_1q(s, RZ(params[5]), 2, N)
        s = apply_cnot(s, 0, 1, N)
        s = apply_cnot(s, 1, 2, N)
        # Layer 3
        s = apply_1q(s, RX(params[6]), 0, N)
        s = apply_1q(s, RY(params[7]), 1, N)
        s = apply_1q(s, RZ(params[8]), 2, N)
        return state_flat(s)

    def loss_fn(params):
        sv    = circuit(params)
        overlap   = jnp.vdot(target, sv)
        fidelity  = jnp.abs(overlap) ** 2
        return 1.0 - fidelity

    @jax.jit
    def train_step(params, m, v, t):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        params, m, v, t = adam_update(params, grads, m, v, t, lr=0.05)
        return params, m, v, t, loss_val

    key    = jax.random.PRNGKey(42)
    params = jax.random.normal(key, shape=(9,)) * 0.1
    m = jnp.zeros_like(params); v = jnp.zeros_like(params); t = 0

    print("\nTraining state preparation circuit (100 epochs)...")
    loss_history = []
    for epoch in range(1, 101):
        params, m, v, t, lv = train_step(params, m, v, t)
        loss_history.append(float(lv))
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {lv:.6f} | Fidelity: {1-lv:.6%}")

    final_sv = circuit(params)
    print("\n  Target State:  ", jnp.round(target, 4))
    print("  Prepared State:", jnp.round(final_sv, 4))

    # Plot
    epochs_r   = range(1, 101)
    fidelities = 1.0 - jnp.array(loss_history)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_r, loss_history, label="Loss (1-Fidelity)", color=P["a3"], lw=2.5)
    ax.plot(epochs_r, fidelities,   label="Fidelity",          color=P["a2"], lw=2.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.set_title(f"01 — GHZ State Preparation Convergence  [{BACKEND.upper()}  |  {TS}]")
    ax.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"])
    theme(fig, ax)
    path = f"examples/plots/01_state_prep_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"\n  Plot saved -> {path}")

    json_path = f"results/state_prep_{TS}.json"
    with open(json_path, "w") as f:
        json.dump({"loss_history": loss_history,
                   "final_fidelity": float(fidelities[-1]),
                   "backend": BACKEND}, f, indent=2)
    print(f"  JSON saved -> {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 2 — VQC XOR Classifier
#  Mirrors: examples/02_vqc_classification.py
# ─────────────────────────────────────────────────────────────────────────────

def run_vqc():
    banner("Experiment 2 — VQC XOR Classifier  (02_vqc_classification.py)")
    N = 2  # 2-qubit circuit

    # Build dataset
    key = jax.random.PRNGKey(24)
    key, sk1, sk2 = jax.random.split(key, 3)
    X_data = jax.random.uniform(sk1, shape=(200, 2), minval=-1.5, maxval=1.5)
    Y_data = jnp.where(X_data[:, 0] * X_data[:, 1] < 0, 1.0, 0.0)

    # 2-qubit VQC: 2 input (RX) + 6 trainable (RY) = 8 total params
    # full_params = [x0, x1, theta0..theta5]
    def circuit_single(full_params):
        s = zero_state(N)
        s = apply_1q(s, RX(full_params[0]), 0, N)  # encode x0
        s = apply_1q(s, RX(full_params[1]), 1, N)  # encode x1
        s = apply_1q(s, RY(full_params[2]), 0, N)
        s = apply_1q(s, RY(full_params[3]), 1, N)
        s = apply_cnot(s, 0, 1, N)
        s = apply_1q(s, RY(full_params[4]), 0, N)
        s = apply_1q(s, RY(full_params[5]), 1, N)
        s = apply_cnot(s, 0, 1, N)
        s = apply_1q(s, RY(full_params[6]), 0, N)
        s = apply_1q(s, RY(full_params[7]), 1, N)
        return pauli_z_expect(s, 1, N)  # measure Z on qubit 1

    def predict_single(params, x):
        full_params = jnp.concatenate([x, params])
        return circuit_single(full_params)

    predict_batch = jax.vmap(predict_single, in_axes=(None, 0))

    def loss_fn(params, Xb, Yb):
        preds   = predict_batch(params, Xb)
        targets = Yb * 2.0 - 1.0    # map {0,1} -> {-1,+1}
        return jnp.mean((preds - targets) ** 2)

    @jax.jit
    def train_step(params, m, v, t, Xb, Yb):
        lv, grads = jax.value_and_grad(loss_fn)(params, Xb, Yb)
        params, m, v, t = adam_update(params, grads, m, v, t, lr=0.03)
        return params, m, v, t, lv

    params = jax.random.normal(sk2, shape=(6,)) * 0.1
    m = jnp.zeros_like(params); v = jnp.zeros_like(params); t = 0

    print("\nTraining VQC classifier (150 epochs)...")
    for epoch in range(1, 151):
        params, m, v, t, lv = train_step(params, m, v, t, X_data, Y_data)
        if epoch == 1 or epoch % 15 == 0:
            preds      = predict_batch(params, X_data)
            pred_class = jnp.where(preds > 0.0, 1.0, 0.0)
            acc        = jnp.mean(pred_class == Y_data)
            print(f"  Epoch {epoch:3d} | Loss: {lv:.6f} | Accuracy: {acc:.2%}")

    # Decision boundary
    print("\nGenerating decision boundary plot...")
    gs = 50
    gx = jnp.linspace(-1.8, 1.8, gs)
    gy = jnp.linspace(-1.8, 1.8, gs)
    xx, yy = jnp.meshgrid(gx, gy)
    grid_pts  = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_pred = predict_batch(params, grid_pts).reshape(gs, gs)

    fig, ax = plt.subplots(figsize=(8, 8))
    cf = ax.contourf(xx, yy, grid_pred, levels=50, cmap="coolwarm", alpha=0.85)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.ax.tick_params(labelsize=10, colors=P["text"])
    cbar.ax.set_ylabel("Z expectation", color=P["text"], labelpad=8)
    s0 = X_data[Y_data == 0]; s1 = X_data[Y_data == 1]
    ax.scatter(s0[:, 0], s0[:, 1], color=P["a1"], label="Class 0", edgecolors=P["bg"], s=50, alpha=0.9)
    ax.scatter(s1[:, 0], s1[:, 1], color=P["a3"], label="Class 1", edgecolors=P["bg"], s=50, alpha=0.9)
    ax.set_xlabel("x0"); ax.set_ylabel("x1")
    ax.set_title(f"02 — VQC XOR Decision Boundary  [{BACKEND.upper()}  |  {TS}]")
    ax.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"])
    theme(fig, ax)
    path = f"examples/plots/02_vqc_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"\n  Plot saved -> {path}")

    json_path = f"results/vqc_{TS}.json"
    final_acc = float(jnp.mean(jnp.where(predict_batch(params, X_data) > 0.0, 1.0, 0.0) == Y_data))
    with open(json_path, "w") as f:
        json.dump({"final_accuracy": final_acc, "backend": BACKEND}, f, indent=2)
    print(f"  JSON saved -> {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 3 — VQE H2 Molecule Ground State
#  Mirrors: examples/04_vqe_h2_molecule.py
# ─────────────────────────────────────────────────────────────────────────────

# H2 Hamiltonian terms: (coefficient, list_of_(qubit, pauli_op))
H2_TERMS = [
    (-0.81054, []),
    ( 0.17120, [(0, "Z")]),
    (-0.22278, [(1, "Z")]),
    (-0.22278, [(2, "Z")]),
    ( 0.17120, [(3, "Z")]),
    ( 0.12091, [(0, "Z"), (1, "Z")]),
    ( 0.16862, [(0, "Z"), (2, "Z")]),
    ( 0.17434, [(1, "Z"), (2, "Z")]),
    ( 0.04532, [(0, "Z"), (3, "Z")]),
    ( 0.16862, [(1, "Z"), (3, "Z")]),
    ( 0.12091, [(2, "Z"), (3, "Z")]),
    ( 0.04532, [(0, "X"), (1, "X"), (2, "Y"), (3, "Y")]),
    (-0.04532, [(0, "Y"), (1, "X"), (2, "X"), (3, "Y")]),
    (-0.04532, [(0, "X"), (1, "Y"), (2, "Y"), (3, "X")]),
    ( 0.04532, [(0, "Y"), (1, "Y"), (2, "X"), (3, "X")]),
]
FCI_ENERGY = -1.1372

PES_DATA = [
    (0.40, -0.8527), (0.50, -1.0284), (0.60, -1.0994), (0.70, -1.1279),
    (0.735,-1.1372), (0.80, -1.1378), (0.90, -1.1311), (1.00, -1.1186),
    (1.20, -1.0882), (1.50, -1.0374), (2.00, -0.9877), (2.50, -0.9694),
]

def _y_basis_gate():
    """Rotate from Y basis: S†H  = diag(1,-i) @ H"""
    sdg = jnp.array([[1, 0], [0, -1j]], dtype=jnp.complex64)
    return jnp.matmul(Hgate(), sdg)

def h2_energy(params):
    """HEA ansatz on 4 qubits, 3 layers + final layer = 32 params."""
    N   = 4
    s   = zero_state(N)
    # HF reference |0011>
    s   = apply_1q(s, Xgate(), 2, N)
    s   = apply_1q(s, Xgate(), 3, N)
    idx = 0
    for _ in range(3):   # 3 layers
        for q in range(N):
            s = apply_1q(s, RY(params[idx]), q, N); idx += 1
            s = apply_1q(s, RZ(params[idx]), q, N); idx += 1
        for q in range(N):
            s = apply_cnot(s, q, (q + 1) % N, N)
    # Final layer
    for q in range(N):
        s = apply_1q(s, RY(params[idx]), q, N); idx += 1
        s = apply_1q(s, RZ(params[idx]), q, N); idx += 1

    # Compute <H>
    total = jnp.array(0.0, dtype=jnp.float32)
    for coeff, ops_list in H2_TERMS:
        if not ops_list:
            total = total + coeff
            continue
        sv = s
        for q, op in ops_list:
            if op == "X":
                sv = apply_1q(sv, Hgate(), q, N)
            elif op == "Y":
                sv = apply_1q(sv, _y_basis_gate(), q, N)
        qs   = [q for q, _ in ops_list]
        ev   = _multi_z_expect(sv, qs, N)
        total = total + coeff * ev
    return jnp.real(total)

def run_vqe():
    banner("Experiment 3 — VQE H2 Molecule Ground State  (04_vqe_h2_molecule.py)")
    NUM_PARAMS = 32   # 4 qubits * (3 layers * 2 + 2 final) * 2 ops

    print(f"\n  FCI reference : {FCI_ENERGY} Hartree")
    print(f"  Ansatz        : HEA 3 layers, {NUM_PARAMS} parameters")
    print(f"  Backend       : {BACKEND.upper()}")
    print(f"  Hamiltonian   : {len(H2_TERMS)} terms (JW mapping)")
    print()

    value_and_grad = jax.jit(jax.value_and_grad(h2_energy))

    key    = jax.random.PRNGKey(42)
    params = jax.random.normal(key, shape=(NUM_PARAMS,)) * 0.05
    m = jnp.zeros_like(params); v = jnp.zeros_like(params); t = 0

    history = []
    t_start = time.perf_counter()
    prev_e  = None

    print(f"  {'Epoch':>6}  {'Energy (Ha)':>14}  {'dE':>12}  {'|grad|':>10}  {'Time':>8}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*12}  {'─'*10}  {'─'*8}")

    for epoch in range(1, 401):
        energy, grads   = value_and_grad(params)
        params, m, v, t = adam_update(params, grads, m, v, t, lr=5e-3)
        ev    = float(energy)
        gn    = float(jnp.linalg.norm(grads))
        de    = ev - prev_e if prev_e is not None else float("nan")
        elapsed = time.perf_counter() - t_start
        history.append({"epoch": epoch, "energy": ev, "delta_e": de,
                         "grad_norm": gn, "elapsed_s": elapsed})
        if epoch == 1 or epoch % 40 == 0 or epoch == 400:
            mark = " [OK]" if abs(ev - FCI_ENERGY) < 1.6e-3 else ""
            print(f"  {epoch:>6}  {ev:>14.8f}  {de:>+12.2e}  {gn:>10.6f}  {elapsed:>8.2f}{mark}")
        prev_e = ev

    final_e = history[-1]["energy"]
    err_mha = abs(final_e - FCI_ENERGY) * 1000
    chem_acc = "YES (<1.6 mHa)" if err_mha < 1.6 else f"NO ({err_mha:.2f} mHa)"
    print(f"\n  VQE energy    : {final_e:+.8f} Ha")
    print(f"  FCI reference : {FCI_ENERGY:+.8f} Ha")
    print(f"  Error         : {err_mha:.4f} mHartree")
    print(f"  Chemical acc. : {chem_acc}")

    # Plot
    epochs  = [h["epoch"]     for h in history]
    energies= [h["energy"]    for h in history]
    gnorms  = [h["grad_norm"] for h in history]
    deltas  = [abs(h["delta_e"]) if not np.isnan(h["delta_e"]) else np.nan
               for h in history]

    fig = plt.figure(figsize=(16, 10), facecolor=P["bg"])
    gs_plt = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                               left=0.08, right=0.97, top=0.91, bottom=0.07)
    ax0 = fig.add_subplot(gs_plt[0, 0])
    ax0.plot(epochs, energies, "-", color=P["a1"], lw=2, label="VQE Energy")
    ax0.axhline(FCI_ENERGY, color=P["a3"], ls="--", lw=1.5,
                label=f"FCI Ref ({FCI_ENERGY:.4f} Ha)")
    ax0.axhspan(FCI_ENERGY - 1.6e-3, FCI_ENERGY + 1.6e-3,
                color=P["a2"], alpha=0.1, label="Chem. accuracy")
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("Energy (Hartree)")
    ax0.set_title("VQE Energy Convergence -- H2")
    ax0.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

    ax1 = fig.add_subplot(gs_plt[0, 1])
    ax1.semilogy(epochs, gnorms, "-", color=P["a4"], lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("|grad| [log]")
    ax1.set_title("Gradient Norm Decay")

    ax2 = fig.add_subplot(gs_plt[1, 0])
    ax2.semilogy(epochs[1:], deltas[1:], "-", color=P["a5"], lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("|dE| [log]")
    ax2.set_title("Energy Change per Step")

    ax3 = fig.add_subplot(gs_plt[1, 1])
    r_fci, e_fci = zip(*PES_DATA)
    ax3.plot(r_fci, e_fci, "o-", color=P["a2"], lw=2, ms=6, label="FCI/STO-3G")
    ax3.axvline(0.735, color=P["a3"], ls=":", lw=1.5, label="Equilibrium")
    ax3.scatter([0.735], [final_e], color=P["a1"], s=120, marker="*", zorder=6,
                label=f"VQE ({final_e:.5f} Ha)")
    ax3.set_xlabel("Bond Length (A)"); ax3.set_ylabel("Energy (Ha)")
    ax3.set_title("H2 Potential Energy Surface")
    ax3.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

    theme(fig, [ax0, ax1, ax2, ax3])
    fig.suptitle(
        f"VQE -- H2 Ground State | {BACKEND.upper()} | {TS}",
        color=P["text"], fontsize=13, fontweight="bold", y=0.97,
    )
    path = f"examples/plots/03_vqe_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"\n  Plot saved -> {path}")

    json_path = f"results/vqe_{TS}.json"
    with open(json_path, "w") as f:
        json.dump({"fci_energy": FCI_ENERGY, "history": history, "backend": BACKEND}, f, indent=2)
    print(f"  JSON saved -> {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 4 — QAOA MaxCut
#  Mirrors: examples/05_qaoa_maxcut.py
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_EDGES = [
    (0, 1, 1.5), (1, 2, 2.0), (2, 3, 1.0), (3, 4, 1.5),
    (4, 5, 2.0), (5, 0, 1.0), (0, 3, 0.5), (1, 4, 0.5), (2, 5, 0.5),
]
NUM_NODES        = 6
CLASSICAL_MAXCUT = 9.0

def _classical_maxcut():
    best_cut, best_mask = 0, 0
    for mask in range(1 << NUM_NODES):
        cut = sum(w for u, v, w in GRAPH_EDGES
                  if bool(mask >> u & 1) != bool(mask >> v & 1))
        if cut > best_cut:
            best_cut, best_mask = cut, mask
    return best_cut, best_mask

def _build_qaoa_circuit_fn(p):
    """Return a function: params -> state_tensor for QAOA depth p."""
    n = NUM_NODES
    def qaoa(params):
        s = zero_state(n)
        h = Hgate()
        for q in range(n):
            s = apply_1q(s, h, q, n)
        for layer in range(p):
            gamma_idx = layer * 2
            beta_idx  = layer * 2 + 1
            for (u, v, w) in GRAPH_EDGES:
                s = apply_cnot(s, u, v, n)
                s = apply_1q(s, RZ(params[gamma_idx]), v, n)
                s = apply_cnot(s, u, v, n)
            for q in range(n):
                s = apply_1q(s, RX(params[beta_idx]), q, n)
        return s
    return qaoa

def _cost_expect(state, n=NUM_NODES):
    """<H_cost> = -0.5 * sum_{edges} w * (1 - <Z_u Z_v>)  (we minimise this)."""
    total = 0.0
    for u, v, w in GRAPH_EDGES:
        zuv = _multi_z_expect(state, [u, v], n)
        total = total + (-w / 2) * zuv + (w / 2)
    return total

def _run_qaoa_depth(p, epochs=250):
    qaoa_fn = _build_qaoa_circuit_fn(p)
    key     = jax.random.PRNGKey(42 + p)
    params  = jax.random.uniform(key, shape=(p * 2,), minval=0.0, maxval=2 * jnp.pi)

    def cost_fn(params):
        s = qaoa_fn(params)
        return -_cost_expect(s)   # negate: maximise cut

    vag = jax.jit(jax.value_and_grad(cost_fn))
    m = jnp.zeros_like(params); v = jnp.zeros_like(params); t = 0
    history = []
    for _ in range(epochs):
        neg_cut, grads  = vag(params)
        params, m, v, t = adam_update(params, grads, m, v, t, lr=0.05)
        history.append(-float(neg_cut))

    # Sample bitstrings from final state
    final_state = qaoa_fn(params)
    flat_probs  = jnp.abs(state_flat(final_state)) ** 2
    sk          = jax.random.PRNGKey(999)
    samples     = jax.random.choice(sk, 2 ** NUM_NODES, shape=(2048,), p=flat_probs)

    def cut_from_mask(mask):
        return sum(w for u, v, w in GRAPH_EDGES
                   if bool(mask >> u & 1) != bool(mask >> v & 1))

    cut_vals       = [cut_from_mask(int(s)) for s in np.array(samples)]
    best_cut_found = max(cut_vals)
    mean_cut       = np.mean(cut_vals)

    return {
        "p": p, "history": history,
        "final_expectation": history[-1],
        "best_cut_sampled":  best_cut_found,
        "mean_cut_sampled":  mean_cut,
        "approx_ratio":      best_cut_found / CLASSICAL_MAXCUT,
    }

def run_qaoa():
    banner("Experiment 4 — QAOA MaxCut  (05_qaoa_maxcut.py)")
    n = NUM_NODES

    classical_cut, best_mask = _classical_maxcut()
    best_part = ["A" if bool(best_mask >> q & 1) else "B" for q in range(n)]
    print(f"\n  Graph        : {n} nodes, {len(GRAPH_EDGES)} edges (weighted)")
    print(f"  Classical opt: {classical_cut:.2f}")
    print(f"  Best partition: {best_part}")
    print(f"  Backend      : {BACKEND.upper()}\n")

    all_results = []
    hdr = ("p", "E[cut]", "Best cut", "Approx ratio")
    print(f"  {'  '.join(str(h).ljust(14) for h in hdr)}")
    print(f"  {'  '.join('─'*14 for _ in hdr)}")

    for p in range(1, 6):
        t0  = time.perf_counter()
        res = _run_qaoa_depth(p, epochs=250)
        dt  = time.perf_counter() - t0
        all_results.append(res)
        print(f"  {p:<14d}  {res['final_expectation']:<14.4f}  "
              f"{res['best_cut_sampled']:<14.2f}  {res['approx_ratio']:<14.4f}  ({dt:.1f}s)")

    print(f"\n  Best QAOA result (p=5): {all_results[-1]['best_cut_sampled']:.2f}  "
          f"(approx ratio {all_results[-1]['approx_ratio']:.4f})")

    # Plot
    COLORS = [P["a1"], P["a2"], P["a3"], P["a4"], P["a5"]]
    fig    = plt.figure(figsize=(16, 10), facecolor=P["bg"])
    gp     = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                               left=0.08, right=0.97, top=0.91, bottom=0.07)

    ax0 = fig.add_subplot(gp[0, 0])
    for i, res in enumerate(all_results):
        ax0.plot(res["history"], color=COLORS[i], lw=1.8,
                 label=f"p = {res['p']}", alpha=0.9)
    ax0.axhline(classical_cut, color=P["a3"], ls="--", lw=1.5,
                label=f"Classical MaxCut ({classical_cut})")
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("Cut Value")
    ax0.set_title("QAOA Convergence per Circuit Depth p")
    ax0.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

    ax1 = fig.add_subplot(gp[0, 1])
    ps  = [r["p"] for r in all_results]
    ars = [r["approx_ratio"] for r in all_results]
    bars = ax1.bar(ps, ars, color=P["a1"], alpha=0.85, edgecolor=P["border"])
    for bar, ar in zip(bars, ars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{ar:.3f}", ha="center", va="bottom", color=P["text"], fontsize=10)
    ax1.axhline(1.0, color=P["a2"], ls="--", lw=1.5, label="Optimal (=1.0)")
    ax1.set_ylim(0.5, 1.05)
    ax1.set_xlabel("Circuit depth p"); ax1.set_ylabel("Approximation Ratio")
    ax1.set_title("Approximation Ratio vs QAOA Depth")
    ax1.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

    ax2 = fig.add_subplot(gp[1, 0])
    best_cuts = [r["best_cut_sampled"] for r in all_results]
    mean_cuts = [r["mean_cut_sampled"] for r in all_results]
    w = 0.35
    ax2.bar([x - w/2 for x in ps], best_cuts, width=w, color=P["a2"], alpha=0.85,
            edgecolor=P["border"], label="Best sampled")
    ax2.bar([x + w/2 for x in ps], mean_cuts, width=w, color=P["a4"], alpha=0.85,
            edgecolor=P["border"], label="Mean sampled")
    ax2.axhline(classical_cut, color=P["a3"], ls="--", lw=1.5,
                label=f"Classical ({classical_cut})")
    ax2.set_xlabel("Circuit depth p"); ax2.set_ylabel("Cut Value")
    ax2.set_title("Sampled Cut Quality vs Depth (2048 shots)")
    ax2.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

    ax3 = fig.add_subplot(gp[1, 1])
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xpos   = np.cos(angles); ypos = np.sin(angles)
    for u, v, w in GRAPH_EDGES:
        ax3.plot([xpos[u], xpos[v]], [ypos[u], ypos[v]],
                 color=P["sub"], lw=1 + w, alpha=0.7)
        ax3.text((xpos[u]+xpos[v])/2, (ypos[u]+ypos[v])/2,
                 f"{w}", color=P["a5"], fontsize=9, ha="center")
    ax3.scatter(xpos, ypos, s=400, color=P["a1"], zorder=5, edgecolors=P["border"], lw=1.5)
    for i, (x, y) in enumerate(zip(xpos, ypos)):
        ax3.text(x, y, str(i), ha="center", va="center",
                 color=P["bg"], fontsize=11, fontweight="bold")
    ax3.set_xlim(-1.4, 1.4); ax3.set_ylim(-1.4, 1.4)
    ax3.set_aspect("equal"); ax3.axis("off")
    ax3.set_title(f"MaxCut Graph ({n} nodes, {len(GRAPH_EDGES)} edges)")
    ax3.set_facecolor(P["panel"])

    theme(fig, [ax0, ax1, ax2])
    fig.suptitle(f"QAOA MaxCut | {BACKEND.upper()} | {TS}",
                 color=P["text"], fontsize=13, fontweight="bold", y=0.97)
    path = f"examples/plots/04_qaoa_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"\n  Plot saved -> {path}")

    json_path = f"results/qaoa_{TS}.json"
    with open(json_path, "w") as f:
        json.dump({"classical_maxcut": CLASSICAL_MAXCUT, "graph_edges": GRAPH_EDGES,
                   "results": all_results, "backend": BACKEND}, f, indent=2)
    print(f"  JSON saved -> {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 5 — Quantum Noise Simulation (Monte Carlo Trajectories)
#  Mirrors: examples/04_quantum_noise_simulation.py
# ─────────────────────────────────────────────────────────────────────────────

# Kraus operators (pure JAX)
def _amplitude_damping(gamma):
    K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=jnp.complex64)
    K1 = jnp.array([[0, jnp.sqrt(gamma)], [0, 0]], dtype=jnp.complex64)
    return K0, K1

def _phase_damping(gamma):
    K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=jnp.complex64)
    K1 = jnp.array([[0, 0], [0, jnp.sqrt(gamma)]], dtype=jnp.complex64)
    return K0, K1

def _depolarizing(p):
    # D(rho) = (1-p)rho + p/3*(X rho X + Y rho Y + Z rho Z)
    # Kraus: sqrt(1-p) I, sqrt(p/3) X, sqrt(p/3) Y, sqrt(p/3) Z
    sp  = jnp.sqrt(p / 3.0 + 1e-12).astype(jnp.complex64)
    sip = jnp.sqrt(jnp.maximum(1 - p, 0.0)).astype(jnp.complex64)
    I   = jnp.eye(2, dtype=jnp.complex64)
    X   = Xgate()
    Y   = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    Z   = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    return sip * I, sp * X, sp * Y, sp * Z

def _apply_kraus_stochastic(state_1q, kraus_ops, key):
    """Apply a random Kraus operator to single-qubit state vector."""
    # Compute probabilities: p_k = ||K_k |psi>||^2
    state_flat_1q = state_1q.reshape(2)
    probs = jnp.array([jnp.real(jnp.vdot(
        jnp.matmul(K, state_flat_1q),
        jnp.matmul(K, state_flat_1q)
    )) for K in kraus_ops])
    probs = jnp.abs(probs)
    probs = probs / (probs.sum() + 1e-12)
    idx   = jax.random.choice(key, len(kraus_ops), p=probs)
    # Apply chosen Kraus operator
    new_states = jnp.stack([
        jnp.matmul(K, state_flat_1q) / jnp.sqrt(jnp.abs(probs[i]) + 1e-12)
        for i, K in enumerate(kraus_ops)
    ])
    return new_states[idx]

def _simulate_amplitude_traj(key, gamma, init_sv):
    K0, K1 = _amplitude_damping(gamma)
    result  = _apply_kraus_stochastic(init_sv, [K0, K1], key)
    # <Z> = |a|^2 - |b|^2
    return jnp.real(jnp.abs(result[0])**2 - jnp.abs(result[1])**2)

def _simulate_phase_traj(key, gamma, init_sv):
    K0, K1 = _phase_damping(gamma)
    result  = _apply_kraus_stochastic(init_sv, [K0, K1], key)
    # <X> via Hadamard
    h  = Hgate()
    rv = jnp.matmul(h, result)
    return jnp.real(jnp.abs(rv[0])**2 - jnp.abs(rv[1])**2)

def _simulate_depol_traj(key, p, init_sv):
    K0, K1, K2, K3 = _depolarizing(p)
    result = _apply_kraus_stochastic(init_sv, [K0, K1, K2, K3], key)
    h  = Hgate()
    rv = jnp.matmul(h, result)
    return jnp.real(jnp.abs(rv[0])**2 - jnp.abs(rv[1])**2)

def run_noise_sim():
    banner("Experiment 5 — Quantum Noise Simulation  (04_quantum_noise_simulation.py)")
    print(f"\n  Backend: {BACKEND.upper()}")

    noise_vals      = np.linspace(0.0, 1.0, 30)
    traj_counts     = [10, 100, 1000]
    traj_colors     = {10: P["a3"], 100: P["a1"], 1000: P["a2"]}

    # 1-qubit init states
    init_1  = jnp.array([0.0, 1.0], dtype=jnp.complex64)           # |1>
    init_p  = jnp.array([1.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)  # |+>

    base_key = jax.random.PRNGKey(101)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=P["bg"])
    plt.subplots_adjust(wspace=0.3)

    # ── Panel 1: Amplitude damping ──────────────────────────────────────────
    print("\n  1. Amplitude damping (|1> -> |0>)...")
    ax1 = axes[0]; ax1.set_facecolor(P["panel"])
    exact_amp = 1.0 - noise_vals
    ax1.plot(noise_vals, exact_amp, label="Exact Analytical", color="#f9e2af", lw=3, zorder=5)
    for num_trajs in traj_counts:
        subkeys = jax.random.split(base_key, num_trajs)
        avg_pops = []
        for g in noise_vals:
            z_vals = [float(_simulate_amplitude_traj(subkeys[i], float(g), init_1))
                      for i in range(num_trajs)]
            pop1 = [(1.0 - z) / 2.0 for z in z_vals]
            avg_pops.append(np.mean(pop1))
        ax1.scatter(noise_vals, avg_pops, label=f"{num_trajs} Traj.",
                    color=traj_colors[num_trajs], alpha=0.8, s=40)
    ax1.set_title("Amplitude Damping (|1> Relaxation)", fontsize=13, fontweight="bold",
                  color=P["text"], pad=12)
    ax1.set_xlabel("Damping Rate"); ax1.set_ylabel("Population |1>")
    ax1.tick_params(colors=P["text"]); ax1.xaxis.label.set_color(P["text"])
    ax1.yaxis.label.set_color(P["text"])
    ax1.grid(True, ls="--", color=P["grid"], alpha=0.4)
    ax1.legend(facecolor=P["bg"], edgecolor=P["a4"], labelcolor=P["text"])

    # ── Panel 2: Phase damping ───────────────────────────────────────────────
    print("  2. Phase damping (|+> dephasing)...")
    ax2 = axes[1]; ax2.set_facecolor(P["panel"])
    exact_phase = np.sqrt(np.maximum(1.0 - noise_vals, 0.0))
    ax2.plot(noise_vals, exact_phase, label="Exact Analytical", color="#f9e2af", lw=3, zorder=5)
    for num_trajs in traj_counts:
        subkeys = jax.random.split(base_key, num_trajs)
        avg_xs  = []
        for g in noise_vals:
            x_vals = [float(_simulate_phase_traj(subkeys[i], float(g), init_p))
                      for i in range(num_trajs)]
            avg_xs.append(np.mean(x_vals))
        ax2.scatter(noise_vals, avg_xs, label=f"{num_trajs} Traj.",
                    color=traj_colors[num_trajs], alpha=0.8, s=40)
    ax2.set_title("Phase Damping (|+> Dephasing)", fontsize=13, fontweight="bold",
                  color=P["text"], pad=12)
    ax2.set_xlabel("Dephasing Rate"); ax2.set_ylabel("<X>")
    ax2.tick_params(colors=P["text"]); ax2.xaxis.label.set_color(P["text"])
    ax2.yaxis.label.set_color(P["text"])
    ax2.grid(True, ls="--", color=P["grid"], alpha=0.4)
    ax2.legend(facecolor=P["bg"], edgecolor=P["a4"], labelcolor=P["text"])

    # ── Panel 3: Depolarizing ────────────────────────────────────────────────
    print("  3. Depolarizing noise (|+>)...")
    ax3 = axes[2]; ax3.set_facecolor(P["panel"])
    exact_depol = 1.0 - (4.0 / 3.0) * noise_vals
    ax3.plot(noise_vals, exact_depol, label="Exact Analytical", color="#f9e2af", lw=3, zorder=5)
    for num_trajs in traj_counts:
        subkeys = jax.random.split(base_key, num_trajs)
        avg_xs  = []
        for g in noise_vals:
            x_vals = [float(_simulate_depol_traj(subkeys[i], float(g), init_p))
                      for i in range(num_trajs)]
            avg_xs.append(np.mean(x_vals))
        ax3.scatter(noise_vals, avg_xs, label=f"{num_trajs} Traj.",
                    color=traj_colors[num_trajs], alpha=0.8, s=40)
    ax3.set_title("Depolarizing Noise on |+>", fontsize=13, fontweight="bold",
                  color=P["text"], pad=12)
    ax3.set_xlabel("Depolarization prob"); ax3.set_ylabel("<X>")
    ax3.tick_params(colors=P["text"]); ax3.xaxis.label.set_color(P["text"])
    ax3.yaxis.label.set_color(P["text"])
    ax3.grid(True, ls="--", color=P["grid"], alpha=0.4)
    ax3.legend(facecolor=P["bg"], edgecolor=P["a4"], labelcolor=P["text"])

    fig.suptitle(
        f"05 -- Monte Carlo Quantum Trajectories vs Exact Analytical | {BACKEND.upper()} | {TS}",
        fontsize=14, fontweight="bold", color=P["text"], y=0.98,
    )
    path = f"examples/plots/05_noise_sim_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"\n  Plot saved -> {path}")

    json_path = f"results/noise_sim_{TS}.json"
    with open(json_path, "w") as f:
        json.dump({"noise_vals": noise_vals.tolist(),
                   "exact_amp": exact_amp.tolist(),
                   "exact_phase": exact_phase.tolist(),
                   "exact_depol": exact_depol.tolist(),
                   "backend": BACKEND}, f, indent=2)
    print(f"  JSON saved -> {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 6 — Barren Plateau Study
#  Mirrors: examples/06_barren_plateaus.py
# ─────────────────────────────────────────────────────────────────────────────

def _build_pqc(n, depth):
    """Return a function params -> state (tensor)."""
    def pqc(params):
        s = zero_state(n)
        idx = 0
        for _ in range(depth):
            for q in range(n):
                s = apply_1q(s, RY(params[idx]), q, n); idx += 1
                s = apply_1q(s, RZ(params[idx]), q, n); idx += 1
            for q in range(n - 1):
                s = apply_cnot(s, q, q + 1, n)
        return s
    return pqc, n * depth * 2   # (fn, num_params)

def _grad_variances(n, depth, num_trials=100, seed=0):
    """Compute gradient variances for a PQC of (n qubits, depth layers)."""
    pqc_fn, num_p = _build_pqc(n, depth)

    def loss(params):
        s = pqc_fn(params)
        return pauli_z_expect(s, 0, n)

    grad_fn = jax.jit(jax.grad(loss))
    key     = jax.random.PRNGKey(seed)
    all_grads = []
    for _ in range(num_trials):
        key, sk = jax.random.split(key)
        p       = jax.random.uniform(sk, shape=(num_p,), minval=0.0, maxval=2 * jnp.pi)
        g       = np.array(grad_fn(p))
        all_grads.append(g)
    return np.var(np.array(all_grads), axis=0)

def _loss_landscape_2d(n=4, depth=2, resolution=60):
    pqc_fn, num_p = _build_pqc(n, depth)
    key    = jax.random.PRNGKey(77)
    p0     = jax.random.uniform(key, shape=(num_p,), minval=0.0, maxval=2 * jnp.pi)

    def loss_2d(t0, t1):
        params = p0.at[0].set(t0).at[1].set(t1)
        return pauli_z_expect(pqc_fn(params), 0, n)

    loss_vv = jax.jit(jax.vmap(jax.vmap(loss_2d, in_axes=(None, 0)), in_axes=(0, None)))
    theta   = jnp.linspace(0, 2 * jnp.pi, resolution)
    Z       = np.array(loss_vv(theta, theta))
    return np.array(theta), Z

def run_barren_plateaus():
    banner("Experiment 6 — Barren Plateau Study  (06_barren_plateaus.py)")
    print(f"\n  Backend: {BACKEND.upper()}")

    qubit_range = list(range(2, 11))
    depth_range = list(range(1, 11))
    NUM_TRIALS  = 100   # reduced from 150 to keep it reasonably fast on TPU

    # Study 1: Width scaling
    print(f"\n  [Study 1] Gradient variance vs. width (depth=4, trials={NUM_TRIALS})")
    print(f"  {'Qubits':<8}  {'Mean Var':>14}")
    width_results = []
    for n in qubit_range:
        var       = _grad_variances(n, depth=4, num_trials=NUM_TRIALS)
        mean_var  = float(np.mean(var))
        width_results.append({"n": n, "mean_var": mean_var,
                               "max_var": float(np.max(var)),
                               "min_var": float(np.min(var))})
        print(f"  {n:<8d}  {mean_var:>14.6e}")

    # Study 2: Depth scaling
    print(f"\n  [Study 2] Gradient variance vs. depth (n=4, trials={NUM_TRIALS})")
    print(f"  {'Depth':<8}  {'Mean Var':>14}")
    depth_results = []
    for d in depth_range:
        var      = _grad_variances(4, depth=d, num_trials=NUM_TRIALS)
        mean_var = float(np.mean(var))
        depth_results.append({"depth": d, "mean_var": mean_var,
                               "max_var": float(np.max(var)),
                               "min_var": float(np.min(var))})
        print(f"  {d:<8d}  {mean_var:>14.6e}")

    # Study 3: 2D landscape
    print("\n  [Study 3] Computing 2D loss landscape (4 qubits, 2 layers)...", end="", flush=True)
    theta, Z = _loss_landscape_2d(n=4, depth=2, resolution=60)
    print(" done.")

    # Exponential fit for width
    ns      = np.array([r["n"] for r in width_results])
    wvs     = np.array([r["mean_var"] for r in width_results])
    log_wvs = np.log(wvs + 1e-20)
    w_fit   = np.polyfit(ns, log_wvs, 1)
    print(f"\n  Width exp. decay: Var ~ exp({w_fit[0]:.4f} * n)  ({np.exp(w_fit[0]):.4f}x per qubit)")

    # Save JSON
    json_path = f"results/barren_plateau_{TS}.json"
    with open(json_path, "w") as f:
        json.dump({
            "width_study":  {"qubit_range": qubit_range, "results": width_results,
                             "exp_decay_slope": float(w_fit[0])},
            "depth_study":  {"depth_range": depth_range, "results": depth_results},
            "landscape":    {"theta": theta.tolist(), "Z": Z.tolist()},
            "backend": BACKEND,
        }, f, indent=2)
    print(f"  JSON saved -> {json_path}")

    # Plot
    fig = plt.figure(figsize=(18, 12), facecolor=P["bg"])
    gp  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)

    # (0,0) Width scaling
    ax0 = fig.add_subplot(gp[0, 0])
    ax0.semilogy(ns, wvs, "o-", color=P["a1"], lw=2.5, ms=8, label="Empirical Var")
    ns_fit = np.linspace(min(ns), max(ns), 200)
    ax0.semilogy(ns_fit, np.exp(np.poly1d(w_fit)(ns_fit)), "--", color=P["a3"], lw=2,
                 label=f"Exp fit ({np.exp(w_fit[0]):.3f}x/qubit)")
    ax0.set_xlabel("Qubits (n)"); ax0.set_ylabel("Var(dE/dt) [log]")
    ax0.set_title("Barren Plateau: Width Scaling")
    ax0.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

    # (0,1) Depth scaling
    ax1 = fig.add_subplot(gp[0, 1])
    ds  = np.array([r["depth"]    for r in depth_results])
    dvs = np.array([r["mean_var"] for r in depth_results])
    ax1.semilogy(ds, dvs, "s-", color=P["a4"], lw=2.5, ms=8)
    log_dvs  = np.log(dvs + 1e-20)
    d_fit    = np.polyfit(ds, log_dvs, 1)
    ds_fit   = np.linspace(min(ds), max(ds), 200)
    ax1.semilogy(ds_fit, np.exp(np.poly1d(d_fit)(ds_fit)), "--", color=P["a3"], lw=2,
                 label=f"Exp fit ({np.exp(d_fit[0]):.3f}x/layer)")
    ax1.set_xlabel("Circuit Depth (p)"); ax1.set_ylabel("Var(dE/dt) [log]")
    ax1.set_title("Barren Plateau: Depth Scaling")
    ax1.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

    # (0,2) Heatmap
    ax2 = fig.add_subplot(gp[0, 2])
    data_m = np.outer(wvs, np.ones(len(ds)))
    im     = ax2.imshow(np.log10(data_m + 1e-20), aspect="auto",
                        extent=[min(ds)-0.5, max(ds)+0.5, min(ns)-0.5, max(ns)+0.5],
                        origin="lower", cmap="plasma")
    cbar   = fig.colorbar(im, ax=ax2)
    cbar.set_label("log10 Var", color=P["text"]); cbar.ax.tick_params(colors=P["text"])
    ax2.set_xlabel("Depth"); ax2.set_ylabel("Qubits")
    ax2.set_title("Gradient Variance Heatmap (log10)")
    ax2.tick_params(colors=P["text"]); ax2.set_facecolor(P["panel"])

    # (1, 0:2) 2D landscape contour
    ax3 = fig.add_subplot(gp[1, :2])
    TH, PH = np.meshgrid(theta, theta)
    contour = ax3.contourf(TH, PH, Z.T, levels=60, cmap="viridis")
    cbar3   = fig.colorbar(contour, ax=ax3)
    cbar3.set_label("E[Z0]", color=P["text"]); cbar3.ax.tick_params(colors=P["text"])
    ax3.contour(TH, PH, Z.T, levels=15, colors="white", alpha=0.2, lw=0.5)
    ax3.set_xlabel("theta_0 (rad)"); ax3.set_ylabel("theta_1 (rad)")
    ax3.set_title("2D Loss Landscape -- PQC (4 qubits, 2 layers)")
    ax3.tick_params(colors=P["text"]); ax3.set_facecolor(P["panel"])
    ax3.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax3.set_xticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
    ax3.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax3.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])

    # (1,2) Gradient norm distribution
    ax4 = fig.add_subplot(gp[1, 2])
    for n_q, col in [(2, P["a2"]), (5, P["a1"]), (8, P["a3"])]:
        pqc_fn, num_p = _build_pqc(n_q, 4)
        key = jax.random.PRNGKey(1234 + n_q)
        grad_norms = []
        for _ in range(100):
            key, sk = jax.random.split(key)
            p   = jax.random.uniform(sk, shape=(num_p,), minval=0.0, maxval=2*jnp.pi)
            g   = jax.grad(lambda pp: pauli_z_expect(pqc_fn(pp), 0, n_q))(p)
            grad_norms.append(float(jnp.linalg.norm(g)))
        ax4.hist(grad_norms, bins=25, color=col, alpha=0.7, density=True,
                 label=f"n={n_q} (mean={np.mean(grad_norms):.4f})")
    ax4.set_xlabel("|grad E|"); ax4.set_ylabel("Density")
    ax4.set_title("Gradient Norm Distribution (n=2,5,8)")
    ax4.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

    theme(fig, [ax0, ax1, ax4])
    fig.suptitle(
        f"Barren Plateau Phenomenon in PQCs -- JAX Quantum Research | {BACKEND.upper()} | {TS}\n"
        "Ref: McClean et al. (2018) Nat. Comm. 9, 4812",
        color=P["text"], fontsize=13, fontweight="bold", y=0.97,
    )
    path = f"examples/plots/06_barren_plateau_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"])
    plt.close()
    print(f"  Plot saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    w = 78
    print("=" * w)
    print("  JAX Quantum Research — 6 Examples Runner  (TPU Self-Contained)".center(w))
    print("=" * w)
    print(f"  Backend  : {BACKEND.upper()}")
    print(f"  Devices  : {DEVICES}")
    print(f"  Timestamp: {TS}")
    print("=" * w)

    t0 = time.perf_counter()

    run_state_prep()
    run_vqc()
    run_vqe()
    run_qaoa()
    run_noise_sim()
    run_barren_plateaus()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * w)
    print(f"  ALL 6 EXPERIMENTS COMPLETE — total time: {elapsed:.1f}s".center(w))
    print("=" * w)
    print(f"\n  Results in : results/")
    print(f"  Plots in   : examples/plots/")

if __name__ == "__main__":
    main()
