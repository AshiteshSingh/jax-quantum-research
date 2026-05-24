#!/usr/bin/env python3
"""
================================================================================
  JAX Quantum Research Suite — TPU v5litepod-16 Edition
  Single self-contained file. Zero imports beyond JAX + matplotlib.
  Runs all 5 experiments and saves plots + JSON results directly on the VM.

  Experiments:
    1. GHZ State Preparation     (Adam optimiser, fidelity convergence)
    2. VQC XOR Classifier        (jax.vmap, decision boundary)
    3. VQE H₂ Ground State       (Jordan-Wigner, chemical accuracy)
    4. QAOA MaxCut               (6-node weighted graph, depths p=1..5)
    5. Qubit Scaling Benchmark   (TPU HBM safeguard, 6-panel plot)
================================================================================
"""
import os, sys, time, math, csv, json, warnings
from datetime import datetime
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ── Headless Matplotlib (must be set before pyplot import) ───────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp

warnings.filterwarnings("ignore", category=jnp.ComplexWarning)
warnings.filterwarnings("ignore", message="Casting complex values")
warnings.filterwarnings("ignore", message="Glyph")          # suppress emoji/unicode font warnings
warnings.filterwarnings("ignore", category=UserWarning)     # suppress all matplotlib UserWarnings

# ─────────────────────────────────────────────────────────────────────────────
# Global timestamp, output directories
# ─────────────────────────────────────────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("results",       exist_ok=True)
os.makedirs("examples/plots", exist_ok=True)

BACKEND = jax.default_backend()
DEVICES = jax.devices()
NUM_DEV  = len(DEVICES)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
P = {
    "bg":"#0d1117","panel":"#161b22","border":"#30363d","text":"#e6edf3",
    "sub":"#8b949e","a1":"#58a6ff","a2":"#3fb950","a3":"#f78166",
    "a4":"#d2a8ff","a5":"#ffa657","grid":"#21262d",
}

def theme(fig, axes):
    fig.patch.set_facecolor(P["bg"])
    for ax in (axes if hasattr(axes,"__iter__") else [axes]):
        ax.set_facecolor(P["panel"])
        ax.tick_params(colors=P["text"], labelsize=10)
        ax.xaxis.label.set_color(P["text"])
        ax.yaxis.label.set_color(P["text"])
        ax.title.set_color(P["text"])
        for sp in ax.spines.values(): sp.set_edgecolor(P["border"])
        ax.grid(True, color=P["grid"], ls="--", alpha=0.6, lw=0.7)

def banner(title):
    w = 78
    print("\n" + "═"*w)
    print(f" {title.center(w-2)} ")
    print("═"*w)

def fmt_bytes(b):
    for u in ("B","KB","MB","GB","TB"):
        if b < 1024: return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"

# ─────────────────────────────────────────────────────────────────────────────
# Pure-JAX Quantum Simulator Primitives
# (No jax_qsim dependency — everything inline)
# ─────────────────────────────────────────────────────────────────────────────

def zero_state(n):
    """Return |0⟩^⊗n as a complex64 tensor of shape (2,)*n."""
    s = jnp.zeros((2,)*n, dtype=jnp.complex64)
    return s.at[(0,)*n].set(1.0)

def apply_1q(state, gate, t, n):
    """Apply a 2×2 gate to qubit t of an n-qubit state tensor."""
    gate = gate.astype(jnp.complex64)
    out  = jnp.tensordot(gate, state, axes=((1,),(t,)))
    axes = list(range(1, n)); axes.insert(t, 0)
    return jnp.transpose(out, axes)

_CNOT = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                   dtype=jnp.complex64).reshape(2,2,2,2)

def apply_cnot(state, c, t, n):
    out  = jnp.tensordot(_CNOT, state, axes=((2,3),(c,t)))
    dest = [None]*n
    dest[c] = 0; dest[t] = 1
    k = 2
    for i in range(n):
        if dest[i] is None: dest[i] = k; k += 1
    return jnp.transpose(out, dest)

# Gate constructors
def H():   return jnp.array([[1,1],[1,-1]], dtype=jnp.complex64)/jnp.sqrt(2.)
def X():   return jnp.array([[0,1],[1,0]], dtype=jnp.complex64)
def RX(θ): c=jnp.cos(θ/2); s=-1j*jnp.sin(θ/2); return jnp.array([[c,s],[s,c]])
def RY(θ): c=jnp.cos(θ/2); s=jnp.sin(θ/2);     return jnp.array([[c,-s],[s,c]])
def RZ(θ): e=jnp.exp(-1j*θ/2); return jnp.array([[e,0],[0,jnp.conj(e)]])

def pauli_z_expectation(state, qubit, n):
    """⟨Z_qubit⟩ — marginalise all other qubits."""
    probs   = jnp.abs(state)**2
    axes    = tuple(i for i in range(n) if i != qubit)
    marginal= jnp.sum(probs, axis=axes)
    return jnp.real(marginal[0] - marginal[1])

def pauli_zz_expectation(state, q0, q1, n):
    """⟨Z_q0 Z_q1⟩"""
    probs   = jnp.abs(state)**2
    axes    = tuple(i for i in range(n) if i not in (q0,q1))
    marginal= jnp.sum(probs, axis=axes)   # shape (2,2)
    return jnp.real(marginal[0,0] - marginal[0,1] - marginal[1,0] + marginal[1,1])

def pauli_x_expectation(state, qubit, n):
    """⟨X_qubit⟩ — apply H then measure Z."""
    s2 = apply_1q(state, H(), qubit, n)
    return pauli_z_expectation(s2, qubit, n)

def state_fidelity(state, target_flat, n):
    """F = |⟨target|ψ⟩|² where target_flat is a 2^n complex vector."""
    flat    = state.reshape(-1)
    overlap = jnp.vdot(target_flat.astype(jnp.complex64), flat)
    return jnp.real(jnp.abs(overlap)**2)

def adam(p, g, m, v, t, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    t  = t+1
    m  = b1*m  + (1-b1)*g
    v  = b2*v  + (1-b2)*g**2
    mh = m/(1-b1**t)
    vh = v/(1-b2**t)
    return p - lr*mh/(jnp.sqrt(vh)+eps), m, v, t

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — GHZ State Preparation
# ═════════════════════════════════════════════════════════════════════════════

def run_state_prep():
    banner("EXPERIMENT 1 — GHZ State Preparation (3 Qubits)")
    N = 3
    # Target: (|000⟩ + |111⟩)/√2
    target = jnp.zeros(2**N, dtype=jnp.complex64)
    target = target.at[0].set(1/jnp.sqrt(2.))
    target = target.at[7].set(1/jnp.sqrt(2.))

    # 9-parameter hardware-efficient ansatz
    def circuit(params):
        s = zero_state(N)
        # Layer 1
        s = apply_1q(s, RX(params[0]), 0, N)
        s = apply_1q(s, RY(params[1]), 1, N)
        s = apply_1q(s, RZ(params[2]), 2, N)
        s = apply_cnot(s,0,1,N); s = apply_cnot(s,1,2,N)
        # Layer 2
        s = apply_1q(s, RX(params[3]), 0, N)
        s = apply_1q(s, RY(params[4]), 1, N)
        s = apply_1q(s, RZ(params[5]), 2, N)
        s = apply_cnot(s,0,1,N); s = apply_cnot(s,1,2,N)
        # Layer 3
        s = apply_1q(s, RX(params[6]), 0, N)
        s = apply_1q(s, RY(params[7]), 1, N)
        s = apply_1q(s, RZ(params[8]), 2, N)
        return s

    def loss(params):
        return 1.0 - state_fidelity(circuit(params), target, N)

    @jax.jit
    def step(params, m, v, t):
        val, g = jax.value_and_grad(loss)(params)
        params, m, v, t = adam(params, g, m, v, t, lr=0.05)
        return params, m, v, t, val

    key    = jax.random.PRNGKey(42)
    params = jax.random.normal(key, (9,)) * 0.1
    m = jnp.zeros(9); v = jnp.zeros(9); t = 0
    hist = []
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Fidelity':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
    for ep in range(1, 201):
        params, m, v, t, lv = step(params, m, v, t)
        hist.append(float(lv))
        if ep == 1 or ep % 20 == 0:
            print(f"  {ep:>6}  {lv:>10.6f}  {1-lv:>10.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10,5), facecolor=P["bg"])
    fids = [1-l for l in hist]
    ax.plot(hist,  color=P["a3"], lw=2.5, label="Loss (1−Fidelity)")
    ax.plot(fids,  color=P["a2"], lw=2.5, label="Fidelity")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.set_title("⚛  GHZ State Preparation — Fidelity Convergence")
    ax.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"])
    theme(fig, ax)
    path = f"examples/plots/01_state_prep_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  Plot saved → {path}")
    json.dump({"experiment":"GHZ_state_prep","loss_history":hist},
              open(f"results/state_prep_{TS}.json","w"), indent=2)
    print(f"  📄 JSON saved → results/state_prep_{TS}.json")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Variational Quantum Classifier (XOR)
# ═════════════════════════════════════════════════════════════════════════════

def run_vqc():
    banner("EXPERIMENT 2 — VQC XOR Classifier (2 Qubits, jax.vmap)")
    N = 2
    key = jax.random.PRNGKey(24)
    key, k1, k2 = jax.random.split(key, 3)
    X = jax.random.uniform(k1, (200,2), minval=-1.5, maxval=1.5)
    Y = jnp.where(X[:,0]*X[:,1] < 0, 1.0, 0.0)

    # 8-param circuit: 2 inputs (angle-encoded) + 6 variational
    def circuit_single(full_params):
        s = zero_state(N)
        s = apply_1q(s, RX(full_params[0]), 0, N)
        s = apply_1q(s, RX(full_params[1]), 1, N)
        s = apply_1q(s, RY(full_params[2]), 0, N)
        s = apply_1q(s, RY(full_params[3]), 1, N)
        s = apply_cnot(s,0,1,N)
        s = apply_1q(s, RY(full_params[4]), 0, N)
        s = apply_1q(s, RY(full_params[5]), 1, N)
        s = apply_cnot(s,0,1,N)
        s = apply_1q(s, RY(full_params[6]), 0, N)
        s = apply_1q(s, RY(full_params[7]), 1, N)
        return pauli_z_expectation(s, 1, N)

    def predict(params, x):
        return circuit_single(jnp.hstack([x, params]))

    predict_batch = jax.vmap(predict, in_axes=(None, 0))

    def loss(params, Xb, Yb):
        preds = predict_batch(params, Xb)
        return jnp.mean((preds - (Yb*2-1))**2)

    @jax.jit
    def step(params, m, v, t, Xb, Yb):
        val, g = jax.value_and_grad(loss)(params, Xb, Yb)
        params, m, v, t = adam(params, g, m, v, t, lr=0.03)
        return params, m, v, t, val

    params = jax.random.normal(k2, (6,)) * 0.1
    m = jnp.zeros(6); v = jnp.zeros(6); t = 0
    hist = []
    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Accuracy':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
    for ep in range(1, 151):
        params, m, v, t, lv = step(params, m, v, t, X, Y)
        hist.append(float(lv))
        if ep == 1 or ep % 15 == 0:
            preds  = predict_batch(params, X)
            acc    = float(jnp.mean(jnp.where(preds>0,1.,0.) == Y))
            print(f"  {ep:>6}  {lv:>10.6f}  {acc:>10.2%}")

    # Decision boundary grid
    gx = jnp.linspace(-1.8,1.8,40)
    gy = jnp.linspace(-1.8,1.8,40)
    xx,yy = jnp.meshgrid(gx, gy)
    grid  = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    zz    = predict_batch(params, grid).reshape(40,40)

    fig, axes = plt.subplots(1,2, figsize=(14,6), facecolor=P["bg"])
    # Left: decision boundary
    ax = axes[0]
    cf = ax.contourf(np.array(xx), np.array(yy), np.array(zz), levels=50,
                     cmap="coolwarm", alpha=0.85)
    plt.colorbar(cf, ax=ax).ax.tick_params(colors=P["text"])
    ax.scatter(*np.array(X[Y==0]).T, c=P["a1"], s=20, label="Class 0", alpha=0.8)
    ax.scatter(*np.array(X[Y==1]).T, c=P["a3"], s=20, label="Class 1", alpha=0.8)
    ax.set_title("🎯  VQC Decision Boundary (XOR)")
    ax.set_xlabel("x₀"); ax.set_ylabel("x₁")
    ax.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"])
    theme(fig, ax)
    # Right: loss curve
    ax2 = axes[1]
    ax2.plot(hist, color=P["a5"], lw=2.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss")
    ax2.set_title("📉  VQC Training Loss")
    theme(fig, ax2)
    fig.suptitle(f"Variational Quantum Classifier — {BACKEND.upper()} │ {TS}",
                 color=P["text"], fontsize=13, fontweight="bold")
    path = f"examples/plots/02_vqc_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  Plot saved → {path}")
    json.dump({"experiment":"VQC_XOR","loss_history":hist},
              open(f"results/vqc_{TS}.json","w"), indent=2)
    print(f"  📄 JSON saved → results/vqc_{TS}.json")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — VQE: H₂ Ground State Energy
# ═════════════════════════════════════════════════════════════════════════════

# H₂ Hamiltonian (Jordan-Wigner, STO-3G, R=0.735 Å)
H2_TERMS = [
    (-0.81054, {}),
    ( 0.17120, {0:"Z"}), (-0.22278, {1:"Z"}),
    (-0.22278, {2:"Z"}), ( 0.17120, {3:"Z"}),
    ( 0.12091, {0:"Z",1:"Z"}), ( 0.16862, {0:"Z",2:"Z"}),
    ( 0.17434, {1:"Z",2:"Z"}), ( 0.04532, {0:"Z",3:"Z"}),
    ( 0.16862, {1:"Z",3:"Z"}), ( 0.12091, {2:"Z",3:"Z"}),
    ( 0.04532, {0:"X",1:"X",2:"Y",3:"Y"}),
    (-0.04532, {0:"Y",1:"X",2:"X",3:"Y"}),
    (-0.04532, {0:"X",1:"Y",2:"Y",3:"X"}),
    ( 0.04532, {0:"Y",1:"Y",2:"X",3:"X"}),
]
FCI_ENERGY = -1.1372

def apply_pauli_string(state, pauli_dict, n):
    """Apply a Pauli string to state and return the resulting state."""
    _H = H(); _X = X()
    for q, op in pauli_dict.items():
        if   op == "X": state = apply_1q(state, _X, q, n)
        elif op == "Y":
            # Y = iXZ: apply Z first then X with phase
            Zg = jnp.diag(jnp.array([1.+0j, -1.+0j]))
            state = apply_1q(state, Zg, q, n)
            state = apply_1q(state, _X, q, n)
            state = state * 1j
        elif op == "Z":
            Zg = jnp.diag(jnp.array([1.+0j, -1.+0j]))
            state = apply_1q(state, Zg, q, n)
    return state

def h2_energy(state, n=4):
    """Compute ⟨ψ|H_H₂|ψ⟩ by looping over Pauli terms."""
    energy = 0.0
    for coeff, pdict in H2_TERMS:
        if not pdict:
            energy = energy + coeff
        else:
            bra   = jnp.conj(state)
            ket   = apply_pauli_string(state, pdict, n)
            exp_v = jnp.real(jnp.sum(bra * ket))
            energy = energy + coeff * exp_v
    return energy

def build_hea(params, n=4, layers=3):
    """Hardware-efficient ansatz for VQE."""
    s = zero_state(n)
    # Hartree-Fock reference |0011⟩
    s = apply_1q(s, X(), 2, n); s = apply_1q(s, X(), 3, n)
    pi = 0
    for _ in range(layers):
        for q in range(n):
            s = apply_1q(s, RY(params[pi]), q, n); pi+=1
            s = apply_1q(s, RZ(params[pi]), q, n); pi+=1
        for q in range(n): s = apply_cnot(s, q, (q+1)%n, n)
    for q in range(n):
        s = apply_1q(s, RY(params[pi]), q, n); pi+=1
        s = apply_1q(s, RZ(params[pi]), q, n); pi+=1
    return s

def run_vqe():
    banner("EXPERIMENT 3 — VQE: H₂ Ground State Energy (4 Qubits, JW mapping)")
    N_LAYERS = 3
    N_PARAMS = N_LAYERS*4*2 + 4*2  # layers*(4 qubits * 2 gates) + final layer
    print(f"  Parameters : {N_PARAMS}")
    print(f"  FCI target : {FCI_ENERGY} Hartree")

    def energy_fn(params):
        state = build_hea(params, n=4, layers=N_LAYERS)
        return h2_energy(state, n=4)

    vg = jax.jit(jax.value_and_grad(energy_fn))

    key    = jax.random.PRNGKey(42)
    params = jax.random.normal(key, (N_PARAMS,)) * 0.05
    m = jnp.zeros(N_PARAMS); v = jnp.zeros(N_PARAMS); t = 0

    hist = []
    print(f"\n  {'Epoch':>6}  {'Energy(Ha)':>14}  {'|∇E|':>12}  {'Error(mHa)':>12}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*12}  {'─'*12}")
    t0 = time.perf_counter()
    for ep in range(1, 401):
        e, g = vg(params)
        params, m, v, t = adam(params, g, m, v, t, lr=5e-3)
        ev  = float(e); gn = float(jnp.linalg.norm(g))
        err = abs(ev - FCI_ENERGY)*1000
        hist.append({"epoch":ep, "energy":ev, "grad_norm":gn, "error_mha":err,
                     "elapsed_s": time.perf_counter()-t0})
        if ep == 1 or ep % 40 == 0 or ep == 400:
            mark = " ✓" if err < 1.6 else ""
            print(f"  {ep:>6}  {ev:>14.8f}  {gn:>12.6f}  {err:>12.4f}{mark}")

    final_e = hist[-1]["energy"]
    final_err = abs(final_e - FCI_ENERGY)*1000
    print(f"\n  ╔{'═'*42}╗")
    print(f"  ║  VQE energy    : {final_e:+.8f} Ha        ║")
    print(f"  ║  FCI reference : {FCI_ENERGY:+.8f} Ha        ║")
    print(f"  ║  Error         : {final_err:.4f} mHartree       ║")
    print(f"  ║  Chem. accuracy: {'✓ YES (<1.6 mHa)' if final_err < 1.6 else f'✗ NO ({final_err:.2f} mHa)'}        ║")
    print(f"  ╚{'═'*42}╝")

    PES = [(0.40,-0.8527),(0.50,-1.0284),(0.60,-1.0994),(0.70,-1.1279),
           (0.735,-1.1372),(0.80,-1.1378),(0.90,-1.1311),(1.00,-1.1186),
           (1.20,-1.0882),(1.50,-1.0374),(2.00,-0.9877),(2.50,-0.9694)]

    fig = plt.figure(figsize=(16,11), facecolor=P["bg"])
    gs  = gridspec.GridSpec(2,2,figure=fig,hspace=0.45,wspace=0.35,
                            left=0.08,right=0.97,top=0.91,bottom=0.07)
    eps    = [h["epoch"]     for h in hist]
    energs = [h["energy"]    for h in hist]
    gnorms = [h["grad_norm"] for h in hist]

    ax0 = fig.add_subplot(gs[0,0])
    ax0.plot(eps, energs, color=P["a1"], lw=2)
    ax0.axhline(FCI_ENERGY, color=P["a3"], ls="--", lw=1.5, label=f"FCI {FCI_ENERGY} Ha")
    ax0.axhspan(FCI_ENERGY-1.6e-3, FCI_ENERGY+1.6e-3,
                color=P["a2"], alpha=0.12, label="Chem. accuracy band")
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("Energy (Ha)")
    ax0.set_title("⚛  VQE Energy Convergence — H₂")
    ax0.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax0)

    ax1 = fig.add_subplot(gs[0,1])
    ax1.semilogy(eps, gnorms, color=P["a4"], lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("|∇E| [log]")
    ax1.set_title("📉  Gradient Norm Decay")
    theme(fig, ax1)

    ax2 = fig.add_subplot(gs[1,0])
    delta = [abs(hist[i]["energy"]-hist[i-1]["energy"]) for i in range(1,len(hist))]
    ax2.semilogy(eps[1:], delta, color=P["a5"], lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("|ΔE| [log]")
    ax2.set_title("🔍  Energy Change per Step")
    theme(fig, ax2)

    ax3 = fig.add_subplot(gs[1,1])
    r_pes,e_pes = zip(*PES)
    ax3.plot(r_pes, e_pes, "o-", color=P["a2"], lw=2, ms=6, label="FCI/STO-3G")
    ax3.axvline(0.735, color=P["a3"], ls=":", lw=1.5, label="Eq. R=0.735Å")
    ax3.scatter([0.735],[FCI_ENERGY],color=P["a3"],s=100,zorder=5)
    ax3.scatter([0.735],[final_e],color=P["a1"],s=120,zorder=6,marker="*",
                label=f"VQE ({final_e:.5f} Ha)")
    ax3.set_xlabel("Bond Length (Å)"); ax3.set_ylabel("Energy (Ha)")
    ax3.set_title("📊  H₂ Potential Energy Surface")
    ax3.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax3)

    fig.suptitle(f"VQE — H₂ Ground State │ JAX Quantum Simulator │ {BACKEND.upper()} │ {TS}",
                 color=P["text"], fontsize=13, fontweight="bold", y=0.97)
    path = f"examples/plots/vqe_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  VQE plot saved → {path}")
    json.dump({"fci_energy":FCI_ENERGY,"history":hist},
              open(f"results/vqe_{TS}.json","w"), indent=2)
    print(f"  📄 VQE JSON saved → results/vqe_{TS}.json")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — QAOA MaxCut (6-node weighted graph)
# ═════════════════════════════════════════════════════════════════════════════

EDGES   = [(0,1,1.5),(1,2,2.0),(2,3,1.0),(3,4,1.5),(4,5,2.0),
           (5,0,1.0),(0,3,0.5),(1,4,0.5),(2,5,0.5)]
N_NODES = 6
CLASS_CUT = 9.0

def qaoa_cost_exact(params, n, p, edges):
    """Run QAOA circuit and return negative cut expectation."""
    s = zero_state(n)
    Hg = H()
    for q in range(n): s = apply_1q(s, Hg, q, n)
    pi = 0
    for layer in range(p):
        gamma = params[pi]; beta = params[pi+1]; pi += 2
        for (u,v,w) in edges:
            s = apply_cnot(s,u,v,n)
            s = apply_1q(s, RZ(w*gamma), v, n)
            s = apply_cnot(s,u,v,n)
        for q in range(n): s = apply_1q(s, RX(beta), q, n)
    # Evaluate MaxCut Hamiltonian H_C = -½Σ w(1-ZiZj)
    cut = 0.0
    for (u,v,w) in edges:
        zz = pauli_zz_expectation(s, u, v, n)
        cut = cut + w/2*(1.0 - zz)
    return -cut   # negative because we minimise

def run_qaoa():
    banner("EXPERIMENT 4 — QAOA MaxCut (6-node weighted graph, p=1..5)")
    # Classical brute-force baseline
    best_cut, best_mask = 0, 0
    for mask in range(1<<N_NODES):
        cut = sum(w for u,v,w in EDGES if bool(mask>>u&1)!=bool(mask>>v&1))
        if cut > best_cut: best_cut,best_mask = cut,mask
    print(f"  Classical MaxCut: {best_cut:.2f}  (exhaustive)")
    print(f"  Best partition  : {['A' if best_mask>>q&1 else 'B' for q in range(N_NODES)]}\n")

    all_res = []
    print(f"  {'p':>3}  {'E[cut]':>8}  {'Approx ratio':>14}  {'Time(s)':>9}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*14}  {'─'*9}")

    COLORS_RES = [P["a1"],P["a2"],P["a3"],P["a4"],P["a5"]]
    fig = plt.figure(figsize=(16,10), facecolor=P["bg"])
    gsp = gridspec.GridSpec(2,2,figure=fig,hspace=0.42,wspace=0.35,
                            left=0.08,right=0.97,top=0.91,bottom=0.07)
    ax_conv = fig.add_subplot(gsp[0,0])
    ax_ar   = fig.add_subplot(gsp[0,1])
    ax_cuts = fig.add_subplot(gsp[1,0])
    ax_graph= fig.add_subplot(gsp[1,1])

    for p in range(1, 6):
        key    = jax.random.PRNGKey(42+p)
        params = jax.random.uniform(key, (p*2,), minval=0., maxval=2*jnp.pi)
        m = jnp.zeros(p*2); v = jnp.zeros(p*2); t = 0

        def make_cost(p_=p):
            def cost_fn(params): return qaoa_cost_exact(params, N_NODES, p_, EDGES)
            return jax.jit(jax.value_and_grad(cost_fn))
        vg = make_cost()

        hist_cut = []
        t0 = time.perf_counter()
        for _ in range(200):
            neg_cut, g = vg(params)
            params, m, v, t = adam(params, g, m, v, t, lr=0.05)
            hist_cut.append(float(-neg_cut))
        dt = time.perf_counter()-t0

        best_exp = max(hist_cut)
        ar = best_exp / CLASS_CUT
        all_res.append({"p":p,"history":hist_cut,"best_exp":best_exp,"approx_ratio":ar})
        print(f"  {p:>3}  {best_exp:>8.4f}  {ar:>14.4f}  {dt:>9.2f}s")
        ax_conv.plot(hist_cut, color=COLORS_RES[p-1], lw=1.8, label=f"p={p}", alpha=0.9)

    # Convergence
    ax_conv.axhline(CLASS_CUT,color=P["a3"],ls="--",lw=1.5,label=f"Classical {CLASS_CUT}")
    ax_conv.set_xlabel("Epoch"); ax_conv.set_ylabel("Cut value")
    ax_conv.set_title("📈  QAOA Convergence per Depth p")
    ax_conv.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax_conv)

    # Approx ratio bar
    ps  = [r["p"] for r in all_res]
    ars = [r["approx_ratio"] for r in all_res]
    bars= ax_ar.bar(ps, ars, color=P["a1"], alpha=0.85, edgecolor=P["border"])
    for bar,ar in zip(bars,ars):
        ax_ar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                   f"{ar:.3f}", ha="center", va="bottom", color=P["text"], fontsize=10)
    ax_ar.axhline(1.0, color=P["a2"], ls="--", lw=1.5, label="Optimal")
    ax_ar.set_ylim(0.5,1.05)
    ax_ar.set_xlabel("Circuit depth p"); ax_ar.set_ylabel("Approximation ratio")
    ax_ar.set_title("🎯  Approximation Ratio vs QAOA Depth")
    ax_ar.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax_ar)

    # Best cut bar
    bests = [r["best_exp"] for r in all_res]
    ax_cuts.bar(ps, bests, color=P["a2"], alpha=0.85, edgecolor=P["border"], label="Best E[cut]")
    ax_cuts.axhline(CLASS_CUT,color=P["a3"],ls="--",lw=1.5,label=f"Classical {CLASS_CUT}")
    ax_cuts.set_xlabel("Depth p"); ax_cuts.set_ylabel("Cut value")
    ax_cuts.set_title("🔬  Best Cut per Depth")
    ax_cuts.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax_cuts)

    # Graph visualization
    angles = np.linspace(0,2*np.pi,N_NODES,endpoint=False)
    xp,yp  = np.cos(angles), np.sin(angles)
    for u,v,w in EDGES:
        ax_graph.plot([xp[u],xp[v]],[yp[u],yp[v]],color=P["sub"],lw=1+w,alpha=0.7)
        ax_graph.text((xp[u]+xp[v])/2,(yp[u]+yp[v])/2,f"{w}",
                      color=P["a5"],fontsize=9,ha="center")
    ax_graph.scatter(xp,yp,s=400,color=P["a1"],zorder=5,edgecolors=P["border"],lw=1.5)
    for i,(x,y) in enumerate(zip(xp,yp)):
        ax_graph.text(x,y,str(i),ha="center",va="center",
                      color=P["bg"],fontsize=11,fontweight="bold")
    ax_graph.set_xlim(-1.4,1.4); ax_graph.set_ylim(-1.4,1.4)
    ax_graph.set_aspect("equal"); ax_graph.axis("off")
    ax_graph.set_facecolor(P["panel"])
    ax_graph.set_title(f"🕸  MaxCut Graph ({N_NODES} nodes, {len(EDGES)} edges)")

    fig.suptitle(f"QAOA MaxCut │ JAX │ {BACKEND.upper()} │ {TS}",
                 color=P["text"],fontsize=13,fontweight="bold",y=0.97)
    path = f"examples/plots/qaoa_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  QAOA plot saved → {path}")
    json.dump({"classical_maxcut":CLASS_CUT,"graph_edges":EDGES,"results":all_res},
              open(f"results/qaoa_{TS}.json","w"), indent=2)
    print(f"  📄 QAOA JSON saved → results/qaoa_{TS}.json")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — TPU Qubit Scaling Benchmark
# ═════════════════════════════════════════════════════════════════════════════

def get_hbm_mib():
    try:
        s = jax.devices()[0].memory_stats()
        if s and "bytes_in_use" in s:
            return s["bytes_in_use"]/1024/1024
    except: pass
    return 0.0

def run_tpu_benchmark():
    banner(f"EXPERIMENT 5 — TPU Qubit Scaling Benchmark ({NUM_DEV} devices, {BACKEND.upper()})")

    MEM_PER_DEV_GB = 16.0               # v5litepod-16: 16 GB HBM per chip
    TOTAL_HBM_GB   = NUM_DEV * MEM_PER_DEV_GB   # 16 chips × 16 GB = 256 GB
    OS_RESERVE_GB  = 10.0               # flat OS/runtime headroom (user request)
    usable_gb      = TOTAL_HBM_GB - OS_RESERVE_GB  # 246 GB available

    # Math recap:
    #   n=34 → 2^34 × 8 bytes = 128 GB state vector  ✓ fits in 246 GB
    #   n=35 → 2^35 × 8 bytes = 256 GB               ✗ exceeds 246 GB → auto-stop
    # Gradient computation adds ~1-2× state-vector overhead (XLA buffer reuse)
    # so n=34 is the practical maximum on this hardware.

    print(f"  Total HBM           : {TOTAL_HBM_GB:.0f} GB  ({NUM_DEV} chips × {MEM_PER_DEV_GB:.0f} GB)")
    print(f"  OS/runtime reserve  : {OS_RESERVE_GB:.0f} GB  (flat, per user request)")
    print(f"  Usable for compute  : {usable_gb:.0f} GB")
    print(f"  Max safe qubit count: 34  (2^34×8 = 128 GB state vector)\n")

    def bench_circuit(params, n):
        s = zero_state(n)
        Hg = H()
        for i in range(n): s = apply_1q(s, Hg, i, n)
        for i in range(n):
            s = apply_1q(s, RX(params[i]),       i, n)
            s = apply_1q(s, RY(params[i+n]),     i, n)
            s = apply_1q(s, RZ(params[i+2*n]),   i, n)
        for i in range(n-1): s = apply_cnot(s, i, i+1, n)
        s = apply_cnot(s, n-1, 0, n)
        probs   = jnp.abs(s)**2
        marginal= jnp.sum(probs, axis=tuple(range(1,n)))
        return jnp.real(marginal[0] - marginal[1])

    HDR = ("Qubits","State Size","Gates","Uncompiled(s)","JIT Compile(s)","JIT Exec(s)","HBM Used","Throughput")
    cw  = (7,11,7,14,15,12,11,18)
    sep = "─"*(sum(cw)+len(cw)*3-1)
    def frow(*v): return " │ ".join(str(x).ljust(w) for x,w in zip(v,cw))
    print("  "+sep); print("  "+frow(*HDR)); print("  "+sep)

    results = []
    for n in range(10, 37):   # n=35 will exceed 246 GB and auto-stop; 36 is safe ceiling
        sb = (2**n)*8
        sg = sb/1024**3
        ng = 1 + n*4 + n
        np_ = 3*n
        if sg > usable_gb:
            print("  "+sep)
            print(f"  │ n={n:2d} │ {fmt_bytes(sb)} > {usable_gb:.0f} GB safety cap — STOPPING")
            break

        def make_fn(n_=n):
            @jax.jit
            def fn(p): return bench_circuit(p, n_)
            return jax.jit(jax.grad(fn))
        gfn = make_fn()

        params = jnp.ones((np_,), dtype=jnp.float32)*0.5

        # Eager (skip above 20 qubits to avoid hanging)
        if n <= 20:
            try:
                t0 = time.perf_counter()
                with jax.disable_jit():
                    eg = jax.grad(bench_circuit)(params, n)
                    eg.block_until_ready()
                t_eager = time.perf_counter()-t0
            except: t_eager = float("nan")
        else:
            t_eager = float("nan")

        hbm0 = get_hbm_mib()
        t0 = time.perf_counter()
        try:
            g = gfn(params); g.block_until_ready()
        except Exception as e:
            print(f"  │ n={n:2d} │ FAILED: {e}"); break
        t_jit_c = time.perf_counter()-t0
        hbm1 = get_hbm_mib()
        hbm_delta = max(0., hbm1-hbm0)

        jt = []
        for _ in range(5):
            t0 = time.perf_counter()
            g = gfn(params); g.block_until_ready()
            jt.append(time.perf_counter()-t0)
        tjm = float(np.mean(jt)); tjs = float(np.std(jt))
        sp  = t_eager/tjm if (tjm>0 and not math.isnan(t_eager)) else 0.
        gps = ng/tjm if tjm>0 else 0.

        r = {"n_qubits":n,"state_size_bytes":sb,"state_size_str":fmt_bytes(sb),
             "num_gates":ng,"t_uncompiled_s":t_eager if not math.isnan(t_eager) else 0.,
             "t_jit_compile_s":t_jit_c,"t_jit_mean_s":tjm,"t_jit_std_s":tjs,
             "speedup_x":sp,"gates_per_second":gps,"hbm_used_mib":hbm_delta,
             "hbm_total_gb":TOTAL_HBM_GB}
        results.append(r)

        print("  "+frow(
            n, fmt_bytes(sb), ng,
            f"{t_eager:.5f}" if not math.isnan(t_eager) else "Skipped",
            f"{t_jit_c:.5f}", f"{tjm:.5f}",
            f"{hbm_delta:.1f} MiB" if hbm_delta>0 else "N/A",
            f"{gps/1e6:.2f}M/s" if gps>=1e6 else f"{gps:.1f}/s"
        ))

    print("  "+sep)
    if not results: print("  No results collected."); return
    print(f"\n  Peak qubits benchmarked: {results[-1]['n_qubits']} "
          f"({results[-1]['state_size_str']})\n")

    # Save CSV + JSON
    csv_path = f"results/tpu_benchmark_{TS}.csv"
    with open(csv_path,"w",newline="") as f:
        csv.DictWriter(f, fieldnames=results[0].keys()).writeheader()
        csv.DictWriter(f, fieldnames=results[0].keys()).writerows(results)
    print(f"  📄 CSV  → {csv_path}")
    meta = {"timestamp":TS,"backend":BACKEND,"devices":NUM_DEV,
            "usable_gb":usable_gb,"results":results}
    json.dump(meta, open(f"results/tpu_benchmark_{TS}.json","w"), indent=2)
    print(f"  📄 JSON → results/tpu_benchmark_{TS}.json")

    # 6-panel benchmark plot
    ns  = [r["n_qubits"]       for r in results]
    tuc = [r["t_uncompiled_s"] for r in results]
    tjc = [r["t_jit_compile_s"]for r in results]
    tjm = [r["t_jit_mean_s"]   for r in results]
    tjs = [r["t_jit_std_s"]    for r in results]
    sps = [r["speedup_x"]      for r in results]
    smb = [r["state_size_bytes"]/(1<<20) for r in results]
    hbm = [r["hbm_used_mib"]   for r in results]
    gps = [r["gates_per_second"]for r in results]

    fig = plt.figure(figsize=(18,14), facecolor=P["bg"])
    gsp = gridspec.GridSpec(3,2,figure=fig,hspace=0.48,wspace=0.35,
                            left=0.07,right=0.97,top=0.92,bottom=0.06)

    ax0 = fig.add_subplot(gsp[0,0])
    vi  = [i for i,t in enumerate(tuc) if t>0]
    if vi: ax0.semilogy([ns[i] for i in vi],[tuc[i] for i in vi],
                         "o--",color=P["a3"],lw=2,label="Uncompiled",ms=6)
    ax0.semilogy(ns,tjc,"s--",color=P["a5"],lw=2,label="JIT compile+exec",ms=6)
    ax0.fill_between(ns,[m-s for m,s in zip(tjm,tjs)],[m+s for m,s in zip(tjm,tjs)],
                     color=P["a1"],alpha=0.2)
    ax0.semilogy(ns,tjm,"d-",color=P["a1"],lw=2.5,label="JIT exec (mean±σ)",ms=7)
    ax0.set_xlabel("Qubits"); ax0.set_ylabel("Time (s) [log]")
    ax0.set_title("⏱  Execution Time Scaling")
    ax0.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    ax0.set_xticks(ns); theme(fig,ax0)

    ax1 = fig.add_subplot(gsp[0,1])
    vsps= [s for s in sps if s>0]; vns=[ns[i] for i,s in enumerate(sps) if s>0]
    if vsps:
        bars=ax1.bar(vns,vsps,color=P["a2"],alpha=0.85,edgecolor=P["border"])
        for b,s in zip(bars,vsps):
            ax1.text(b.get_x()+b.get_width()/2,b.get_height()+.05,
                     f"{s:.1f}×",ha="center",va="bottom",color=P["text"],fontsize=8)
    ax1.set_xlabel("Qubits"); ax1.set_ylabel("Speedup (Uncompiled/JIT)")
    ax1.set_title("🚀  JIT Speedup over Eager")
    ax1.set_xticks(ns); theme(fig,ax1)

    ax2 = fig.add_subplot(gsp[1,0])
    ax2.semilogy(ns,smb,"o-",color=P["a4"],lw=2.5,ms=7)
    ax2.axhline(TOTAL_HBM_GB*1024, color=P["a3"],ls="--",lw=1.5,
                label=f"Total HBM ({TOTAL_HBM_GB:.0f} GB = {NUM_DEV} chips x 16 GB)")
    ax2.axhline(usable_gb*1024,color=P["a5"],ls=":",lw=1.5,
                label=f"Usable cap ({usable_gb:.0f} GB, -10 GB OS reserve)")
    ax2.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    ax2.set_xlabel("Qubits"); ax2.set_ylabel("State-Vector (MiB) [log]")
    ax2.set_title("💾  Memory Footprint (2ⁿ×8 bytes)")
    ax2.set_xticks(ns); theme(fig,ax2)

    ax3 = fig.add_subplot(gsp[1,1])
    ax3.plot(ns,[g/1e6 for g in gps],"s-",color=P["a5"],lw=2.5,ms=7)
    ax3.set_xlabel("Qubits"); ax3.set_ylabel("Throughput (Mgates/s)")
    ax3.set_title("⚡  Gate Throughput on TPU (JIT)")
    ax3.set_xticks(ns); theme(fig,ax3)

    ax4 = fig.add_subplot(gsp[2,0])
    ax4.bar(ns,[v if v>0 else 0 for v in hbm],color=P["a1"],alpha=0.8,
            edgecolor=P["border"])
    ax4.set_xlabel("Qubits"); ax4.set_ylabel("HBM Delta (MiB)")
    ax4.set_title(f"🎮  HBM Allocation Delta ({NUM_DEV}-chip mesh)")
    ax4.set_xticks(ns); theme(fig,ax4)

    ax5 = fig.add_subplot(gsp[2,1])
    lt  = np.log2(np.array(tjm)+1e-12)
    cf  = np.polyfit(ns,lt,1)
    nf  = np.linspace(min(ns),max(ns),200)
    ax5.scatter(ns,tjm,color=P["a1"],s=55,zorder=5,label="JIT exec data")
    ax5.plot(nf,2**np.poly1d(cf)(nf),"-",color=P["a3"],lw=2.5,
             label=f"Exp fit: 2^({cf[0]:.3f}n)")
    ax5.set_yscale("log"); ax5.set_xlabel("Qubits"); ax5.set_ylabel("Time (s) [log]")
    ax5.set_title(f"📈  Exponential Scaling (slope {cf[0]:.3f})")
    ax5.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    ax5.set_xticks(ns); theme(fig,ax5)

    fig.suptitle(
        f"JAX TPU Quantum Scaling Benchmark  |  {BACKEND.upper()}  |  {NUM_DEV} devices  |  {TS}\n"
        f"v5litepod-16  |  {TOTAL_HBM_GB:.0f} GB total  |  {usable_gb:.0f} GB usable  |  max n=34 qubits",
        color=P["text"],fontsize=13,fontweight="bold",y=0.98)
    path = f"examples/plots/tpu_benchmark_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"  🖼  Benchmark plot saved → {path}")

# ═════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner(f"JAX QUANTUM RESEARCH SUITE — TPU Edition  │  {TS}")
    print(f"  Backend  : {BACKEND.upper()}")
    print(f"  Devices  : {NUM_DEV}")
    for i,d in enumerate(DEVICES): print(f"    [{i:2d}] {d}")

    t_total = time.perf_counter()

    run_state_prep()
    run_vqc()
    run_vqe()
    run_qaoa()
    run_tpu_benchmark()

    banner(f"ALL EXPERIMENTS COMPLETE — total time: {time.perf_counter()-t_total:.1f}s")
    print(f"  📁 Results  → results/")
    print(f"  🖼  Plots    → examples/plots/")
    print(f"  🕐 Timestamp: {TS}\n")
