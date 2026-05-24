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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.sharding import PositionalSharding

warnings.filterwarnings("ignore", category=jnp.ComplexWarning)
warnings.filterwarnings("ignore", message="Casting complex values")
warnings.filterwarnings("ignore", message="Glyph")          
warnings.filterwarnings("ignore", category=UserWarning)     


TS = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("tpu/results",       exist_ok=True)
os.makedirs("tpu/plots",         exist_ok=True)

BACKEND = jax.default_backend()
DEVICES = jax.devices()
NUM_DEV  = len(DEVICES)


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

# 

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
    marginal= jnp.sum(probs, axis=axes)    
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
    path = f"tpu/plots/01_state_prep_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  Plot saved → {path}")
    json.dump({"experiment":"GHZ_state_prep","loss_history":hist},
              open(f"tpu/results/state_prep_{TS}.json","w"), indent=2)
    print(f"  📄 JSON saved → tpu/results/state_prep_{TS}.json")

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
    path = f"tpu/plots/02_vqc_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  Plot saved → {path}")
    json.dump({"experiment":"VQC_XOR","loss_history":hist},
              open(f"tpu/results/vqc_{TS}.json","w"), indent=2)
    print(f"  📄 JSON saved → tpu/results/vqc_{TS}.json")

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
    path = f"tpu/plots/vqe_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  VQE plot saved → {path}")
    json.dump({"fci_energy":FCI_ENERGY,"history":hist},
              open(f"tpu/results/vqe_{TS}.json","w"), indent=2)
    print(f"  📄 VQE JSON saved → tpu/results/vqe_{TS}.json")

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
    path = f"tpu/plots/qaoa_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  QAOA plot saved → {path}")
    json.dump({"classical_maxcut":CLASS_CUT,"graph_edges":EDGES,"results":all_res},
              open(f"tpu/results/qaoa_{TS}.json","w"), indent=2)
    print(f"  📄 QAOA JSON saved → tpu/results/qaoa_{TS}.json")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — Quantum Noise Simulation (Monte Carlo Trajectories)
# ═════════════════════════════════════════════════════════════════════════════

def amplitude_damping_kraus(gamma):
    """Amplitude damping Kraus operators."""
    K0 = jnp.array([[1,0],[0,jnp.sqrt(1-gamma)]], dtype=jnp.complex64)
    K1 = jnp.array([[0,jnp.sqrt(gamma)],[0,0]], dtype=jnp.complex64)
    return [K0, K1]

def phase_damping_kraus(gamma):
    """Phase damping Kraus operators."""
    K0 = jnp.array([[1,0],[0,jnp.sqrt(1-gamma)]], dtype=jnp.complex64)
    K1 = jnp.array([[0,0],[0,jnp.sqrt(gamma)]], dtype=jnp.complex64)
    return [K0, K1]

def depolarizing_kraus(p):
    """Depolarizing channel Kraus operators."""
    s = jnp.sqrt(p/3)
    K0 = jnp.sqrt(1-p) * jnp.eye(2, dtype=jnp.complex64)
    K1 = s * jnp.array([[0,1],[1,0]], dtype=jnp.complex64)       # X
    K2 = s * jnp.array([[0,-1j],[1j,0]], dtype=jnp.complex64)    # Y
    K3 = s * jnp.array([[1,0],[0,-1]], dtype=jnp.complex64)      # Z
    return [K0, K1, K2, K3]

def apply_channel_1q(state, kraus_ops, key):
    """Apply a single-qubit noise channel via stochastic trajectory.
    state: shape (2,) single qubit state.
    Returns the post-channel state (renormalised).
    """
    # Compute probability of each Kraus outcome
    probs = jnp.array([jnp.real(jnp.vdot(K @ state, K @ state)) for K in kraus_ops])
    probs = probs / jnp.sum(probs)
    # Stochastic selection
    idx = jax.random.choice(key, len(kraus_ops), p=probs)
    # Stack Kraus ops and index
    K_stack = jnp.stack(kraus_ops)
    new_state = K_stack[idx] @ state
    norm = jnp.sqrt(jnp.real(jnp.vdot(new_state, new_state)) + 1e-12)
    return new_state / norm

def run_noise_sim():
    banner("EXPERIMENT 5 — Quantum Noise Simulation (Monte Carlo Trajectories)")

    # Initial states
    state_1    = jnp.array([0.0, 1.0], dtype=jnp.complex64)  # |1⟩
    state_plus = jnp.array([1.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.)  # |+⟩

    noise_vals = jnp.linspace(0.0, 0.99, 25)
    trajectory_counts = [10, 100, 500]
    base_key = jax.random.PRNGKey(101)

    def simulate_amp_traj(key, gamma):
        kraus = amplitude_damping_kraus(gamma)
        s = apply_channel_1q(state_1, kraus, key)
        # ⟨Z⟩ = |α|² - |β|²
        return jnp.real(jnp.abs(s[0])**2 - jnp.abs(s[1])**2)

    def simulate_phase_traj(key, gamma):
        kraus = phase_damping_kraus(gamma)
        s = apply_channel_1q(state_plus, kraus, key)
        # ⟨X⟩ via H then ⟨Z⟩
        Hg = jnp.array([[1,1],[1,-1]], dtype=jnp.complex64)/jnp.sqrt(2.)
        sh = Hg @ s
        return jnp.real(jnp.abs(sh[0])**2 - jnp.abs(sh[1])**2)

    def simulate_depol_traj(key, p):
        kraus = depolarizing_kraus(p)
        s = apply_channel_1q(state_plus, kraus, key)
        Hg = jnp.array([[1,1],[1,-1]], dtype=jnp.complex64)/jnp.sqrt(2.)
        sh = Hg @ s
        return jnp.real(jnp.abs(sh[0])**2 - jnp.abs(sh[1])**2)

    print("  Running amplitude damping, phase damping, depolarizing simulations...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=P["bg"])
    traj_colors = {10: P["a3"], 100: P["a1"], 500: P["a2"]}

    # --- Amplitude Damping ---
    ax1 = axes[0]
    exact_amp_pop = 1.0 - np.array(noise_vals)  # population of |1⟩ = 1 - gamma
    ax1.plot(noise_vals, exact_amp_pop, color=P["a5"], lw=3, label="Exact Analytical", zorder=5)

    for nt in trajectory_counts:
        subkeys = jax.random.split(base_key, nt)
        avg_pops = []
        for gv in noise_vals:
            z_vals = jax.vmap(simulate_amp_traj, in_axes=(0, None))(subkeys, gv)
            pop_1 = (1.0 - z_vals) / 2.0  # convert ⟨Z⟩ to |1⟩ population
            avg_pops.append(float(jnp.mean(pop_1)))
        ax1.scatter(noise_vals, avg_pops, color=traj_colors[nt], s=30, alpha=0.8, label=f"{nt} Trajectories")

    ax1.set_title("|1⟩ Relaxation (Amplitude Damping)")
    ax1.set_xlabel("Damping Rate (gamma)"); ax1.set_ylabel("Population |1⟩")
    ax1.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=8)
    theme(fig, ax1)

    # --- Phase Damping ---
    ax2 = axes[1]
    exact_phase = np.sqrt(np.maximum(1.0 - np.array(noise_vals), 0))
    ax2.plot(noise_vals, exact_phase, color=P["a5"], lw=3, label="Exact Analytical", zorder=5)

    for nt in trajectory_counts:
        subkeys = jax.random.split(base_key, nt)
        avg_x = []
        for gv in noise_vals:
            x_vals = jax.vmap(simulate_phase_traj, in_axes=(0, None))(subkeys, gv)
            avg_x.append(float(jnp.mean(x_vals)))
        ax2.scatter(noise_vals, avg_x, color=traj_colors[nt], s=30, alpha=0.8, label=f"{nt} Trajectories")

    ax2.set_title("Dephasing of |+⟩ (Phase Damping)")
    ax2.set_xlabel("Dephasing Rate (gamma)"); ax2.set_ylabel("⟨X⟩")
    ax2.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=8)
    theme(fig, ax2)

    # --- Depolarizing ---
    ax3 = axes[2]
    exact_depol = 1.0 - (4.0/3.0) * np.array(noise_vals)
    ax3.plot(noise_vals, exact_depol, color=P["a5"], lw=3, label="Exact Analytical", zorder=5)

    for nt in trajectory_counts:
        subkeys = jax.random.split(base_key, nt)
        avg_x = []
        for pv in noise_vals:
            x_vals = jax.vmap(simulate_depol_traj, in_axes=(0, None))(subkeys, pv)
            avg_x.append(float(jnp.mean(x_vals)))
        ax3.scatter(noise_vals, avg_x, color=traj_colors[nt], s=30, alpha=0.8, label=f"{nt} Trajectories")

    ax3.set_title("Depolarizing Noise on |+⟩")
    ax3.set_xlabel("Depol. Probability (p)"); ax3.set_ylabel("⟨X⟩")
    ax3.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=8)
    theme(fig, ax3)

    fig.suptitle(f"Monte Carlo Quantum Trajectories vs Exact Solutions │ {BACKEND.upper()} │ {TS}",
                 color=P["text"], fontsize=13, fontweight="bold", y=0.98)
    path = f"tpu/plots/05_noise_sim_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  Noise simulation plot saved → {path}")
    json.dump({"experiment":"noise_simulation","noise_vals":noise_vals.tolist(),
               "trajectory_counts":trajectory_counts},
              open(f"tpu/results/noise_sim_{TS}.json","w"), indent=2)
    print(f"  📄 JSON saved → tpu/results/noise_sim_{TS}.json")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — Noisy NISQ Circuit Simulation & Scaling
# ═════════════════════════════════════════════════════════════════════════════

def run_nisq_benchmark():
    banner("EXPERIMENT 6 — Noisy NISQ Circuit Simulation & Scaling Benchmark")

    def nisq_circuit_noisy(params, n, depth, noise_p, key):
        """Run a single noisy NISQ trajectory: RX/RY layers + CNOT + depolarizing."""
        s = zero_state(n)
        pi = 0
        for d_ in range(depth):
            for q in range(n):
                s = apply_1q(s, RX(params[pi]), q, n); pi += 1
                s = apply_1q(s, RY(params[pi]), q, n); pi += 1
                # Single-qubit depolarizing noise (stochastic)
                key, sk = jax.random.split(key)
                # Simplified inline noise: with prob noise_p, apply random Pauli
                r = jax.random.uniform(sk)
                key, sk2 = jax.random.split(key)
                pauli_idx = jax.random.randint(sk2, (), 0, 3)
                # Apply noise conditionally
                Xg = jnp.array([[0,1],[1,0]], dtype=jnp.complex64)
                Yg = jnp.array([[0,-1j],[1j,0]], dtype=jnp.complex64)
                Zg = jnp.array([[1,0],[0,-1]], dtype=jnp.complex64)
                Ig = jnp.eye(2, dtype=jnp.complex64)
                pauli_stack = jnp.stack([Xg, Yg, Zg])
                gate = jnp.where(r < noise_p, pauli_stack[pauli_idx], Ig)
                s = apply_1q(s, gate, q, n)
            # CNOT entanglement
            for q in range(0, n-1, 2):
                s = apply_cnot(s, q, q+1, n)
        return s

    NUM_Q = 8  # moderate for TPU (tensor ops scale as 2^n)
    DEPTH = 4
    NUM_TRAJS = 50
    noise_rates = jnp.linspace(0.0, 0.05, 8)
    n_params = NUM_Q * DEPTH * 2

    key = jax.random.PRNGKey(42)
    key, pk, tk = jax.random.split(key, 3)
    params = jax.random.uniform(pk, (n_params,), minval=0., maxval=2*jnp.pi)

    print(f"  Qubits: {NUM_Q}, Depth: {DEPTH}, Trajectories: {NUM_TRAJS}")
    print(f"  Noise rates: {len(noise_rates)} steps from 0 to 0.05")
    print(f"  Computing noisy trajectories...", flush=True)

    # Get ideal state (noise_p=0)
    ideal = nisq_circuit_noisy(params, NUM_Q, DEPTH, 0.0, tk)
    ideal_flat = ideal.reshape(-1)

    # Compute fidelities for each noise rate
    mean_fids = []
    for ni, p_noise in enumerate(noise_rates):
        traj_keys = jax.random.split(tk, NUM_TRAJS)
        fids = []
        for j in range(NUM_TRAJS):
            noisy = nisq_circuit_noisy(params, NUM_Q, DEPTH, float(p_noise), traj_keys[j])
            noisy_flat = noisy.reshape(-1)
            fid = float(jnp.abs(jnp.vdot(ideal_flat, noisy_flat))**2)
            fids.append(fid)
        mf = float(np.mean(fids))
        mean_fids.append(mf)
        print(f"    noise={float(p_noise):.4f}  mean_fidelity={mf:.4f}")

    # Qubit scaling benchmark
    print(f"\n  Running qubit scaling benchmark (noisy circuits)...")
    scaling_qubits = [4, 5, 6, 7, 8, 9, 10]
    bench_times = []
    for nq in scaling_qubits:
        np_ = nq * DEPTH * 2
        p_ = jax.random.uniform(pk, (np_,), minval=0., maxval=2*jnp.pi)
        # Warm up
        _ = nisq_circuit_noisy(p_, nq, DEPTH, 0.01, tk)
        t0 = time.perf_counter()
        for j in range(10):
            _ = nisq_circuit_noisy(p_, nq, DEPTH, 0.01, jax.random.fold_in(tk, j))
        dt = (time.perf_counter() - t0) / 10 * 1000
        bench_times.append(dt)
        print(f"    Qubits: {nq:2d}  |  {dt:.2f} ms/trajectory")

    # Theoretical fidelity bound: (1-p)^(num_noisy_gates)
    num_noisy_gates = DEPTH * (NUM_Q + (NUM_Q//2)*2)
    theoretical_fid = [(1.0 - float(p))**num_noisy_gates for p in noise_rates]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=P["bg"])

    ax1.plot(noise_rates, mean_fids, "o-", color=P["a2"], lw=3, ms=7, label="Mean Trajectory Fidelity")
    ax1.plot(noise_rates, theoretical_fid, "--", color=P["a3"], lw=2.5, label=f"Theoretical (1-p)^{num_noisy_gates}")
    ax1.set_title(f"Fidelity Decay vs Noise Rate ({NUM_Q} Qubits)")
    ax1.set_xlabel("Depolarizing Noise Rate (p)"); ax1.set_ylabel("State Fidelity")
    ax1.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax1)

    ax2.bar(scaling_qubits, bench_times, color=P["a4"], alpha=0.85, edgecolor=P["border"])
    ax2.plot(scaling_qubits, bench_times, "D-", color=P["a5"], lw=2, ms=6)
    ax2.set_title(f"TPU Scaling: Noisy NISQ Circuit ({DEPTH} layers)")
    ax2.set_xlabel("Number of Qubits"); ax2.set_ylabel("Time per Trajectory (ms)")
    theme(fig, ax2)

    fig.suptitle(f"Noisy NISQ Circuit Simulation │ {BACKEND.upper()} │ {TS}",
                 color=P["text"], fontsize=13, fontweight="bold", y=0.98)
    path = f"tpu/plots/06_nisq_benchmark_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  NISQ benchmark plot saved → {path}")
    json.dump({"experiment":"nisq_benchmark","num_qubits":NUM_Q,"depth":DEPTH,
               "mean_fidelities":mean_fids,"scaling_qubits":scaling_qubits,
               "scaling_times_ms":bench_times},
              open(f"tpu/results/nisq_benchmark_{TS}.json","w"), indent=2)
    print(f"  📄 JSON saved → tpu/results/nisq_benchmark_{TS}.json")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7 — Barren Plateau Study
# ═════════════════════════════════════════════════════════════════════════════

def build_pqc_state(params, n, depth):
    """Build a parameterised quantum circuit (RY/RZ + CNOT chain) and return state."""
    s = zero_state(n)
    pi = 0
    for _ in range(depth):
        for q in range(n):
            s = apply_1q(s, RY(params[pi]), q, n); pi += 1
            s = apply_1q(s, RZ(params[pi]), q, n); pi += 1
        for q in range(n-1):
            s = apply_cnot(s, q, q+1, n)
    return s

def pqc_z0_expectation(params, n, depth):
    """⟨Z_0⟩ for a random PQC."""
    s = build_pqc_state(params, n, depth)
    return pauli_z_expectation(s, 0, n)

def compute_grad_variances(n, depth, num_trials=150, seed=0):
    """Sample random params, compute gradient, return variance of each component."""
    n_params = n * depth * 2

    def loss(p):
        return pqc_z0_expectation(p, n, depth)

    grad_fn = jax.jit(jax.grad(loss))
    key = jax.random.PRNGKey(seed)
    all_grads = []
    for _ in range(num_trials):
        key, sk = jax.random.split(key)
        p = jax.random.uniform(sk, (n_params,), minval=0., maxval=2*jnp.pi)
        g = np.array(grad_fn(p))
        all_grads.append(g)
    all_grads = np.array(all_grads)
    return np.var(all_grads, axis=0)

def run_barren_plateau():
    banner("EXPERIMENT 7 — Barren Plateau Study (Vanishing Gradients in PQCs)")
    print("  Reference: McClean et al. (2018) Nature Comm. 9, 4812\n")

    qubit_range = list(range(2, 9))     # 2-8 qubits (tensor ops)
    depth_range = list(range(1, 9))     # 1-8 layers
    NUM_TRIALS = 100

    # Study 1: Width scaling
    print(f"  [Study 1] Gradient variance vs width  (depth=4, trials={NUM_TRIALS})")
    print(f"  {'Qubits':<8}  {'Mean Var':>15}  {'Max Var':>15}  {'Min Var':>15}")
    print(f"  {'─'*8}  {'─'*15}  {'─'*15}  {'─'*15}")
    width_results = []
    for n in qubit_range:
        var = compute_grad_variances(n, depth=4, num_trials=NUM_TRIALS)
        mv = float(np.mean(var))
        width_results.append({"n": n, "mean_var": mv,
                              "max_var": float(np.max(var)),
                              "min_var": float(np.min(var))})
        print(f"  {n:<8d}  {mv:>15.6e}  {np.max(var):>15.6e}  {np.min(var):>15.6e}")

    # Study 2: Depth scaling
    print(f"\n  [Study 2] Gradient variance vs depth  (n=4, trials={NUM_TRIALS})")
    print(f"  {'Depth':<8}  {'Mean Var':>15}  {'Max Var':>15}  {'Min Var':>15}")
    print(f"  {'─'*8}  {'─'*15}  {'─'*15}  {'─'*15}")
    depth_results = []
    for d in depth_range:
        var = compute_grad_variances(4, depth=d, num_trials=NUM_TRIALS)
        mv = float(np.mean(var))
        depth_results.append({"depth": d, "mean_var": mv,
                              "max_var": float(np.max(var)),
                              "min_var": float(np.min(var))})
        print(f"  {d:<8d}  {mv:>15.6e}  {np.max(var):>15.6e}  {np.min(var):>15.6e}")

    # Study 3: 2D loss landscape
    print("\n  [Study 3] Computing 2D loss landscape (4 qubits, 2 layers)...", end="", flush=True)
    n_ls, d_ls, res = 4, 2, 50
    n_params_ls = n_ls * d_ls * 2
    key_ls = jax.random.PRNGKey(77)
    params0 = jax.random.uniform(key_ls, (n_params_ls,), minval=0., maxval=2*jnp.pi)

    theta = np.linspace(0, 2*np.pi, res)
    Z_landscape = np.zeros((res, res))
    for i, t0 in enumerate(theta):
        for j, t1 in enumerate(theta):
            p_ = params0.at[0].set(t0).at[1].set(t1)
            Z_landscape[i, j] = float(pqc_z0_expectation(p_, n_ls, d_ls))
    print(" done.")

    # Exponential fit
    ns_arr = np.array([r["n"] for r in width_results])
    wvs_arr = np.array([r["mean_var"] for r in width_results])
    log_wvs = np.log(wvs_arr + 1e-20)
    width_fit = np.polyfit(ns_arr, log_wvs, 1)
    print(f"\n  Width decay fit: Var ~ exp({width_fit[0]:.4f} * n)")
    print(f"  => {np.exp(width_fit[0]):.4f}x per qubit added")

    # Study 4: Gradient norm distributions (computed during plotting)

    # ── PLOT ──
    fig = plt.figure(figsize=(18, 12), facecolor=P["bg"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)

    # (0,0) Width scaling
    ax0 = fig.add_subplot(gs[0, 0])
    ns_w = [r["n"] for r in width_results]
    wvs = [r["mean_var"] for r in width_results]
    ax0.semilogy(ns_w, wvs, "o-", color=P["a1"], lw=2.5, ms=8, label="Empirical Var")
    ns_fit_x = np.linspace(min(ns_w), max(ns_w), 200)
    ax0.semilogy(ns_fit_x, np.exp(np.poly1d(width_fit)(ns_fit_x)),
                 "--", color=P["a3"], lw=2, label=f"Exp fit (x{np.exp(width_fit[0]):.3f}/qubit)")
    ax0.set_xlabel("Number of Qubits"); ax0.set_ylabel("Var(dE/dθ) [log]")
    ax0.set_title("Barren Plateau: Width Scaling")
    ax0.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax0)

    # (0,1) Depth scaling
    ax1 = fig.add_subplot(gs[0, 1])
    ds = [r["depth"] for r in depth_results]
    dvs = [r["mean_var"] for r in depth_results]
    ax1.semilogy(ds, dvs, "s-", color=P["a4"], lw=2.5, ms=8)
    log_dvs = np.log(np.array(dvs) + 1e-20)
    depth_fit = np.polyfit(ds, log_dvs, 1)
    ds_fit_x = np.linspace(min(ds), max(ds), 200)
    ax1.semilogy(ds_fit_x, np.exp(np.poly1d(depth_fit)(ds_fit_x)),
                 "--", color=P["a3"], lw=2, label=f"Exp fit (x{np.exp(depth_fit[0]):.3f}/layer)")
    ax1.set_xlabel("Circuit Depth"); ax1.set_ylabel("Var(dE/dθ) [log]")
    ax1.set_title("Barren Plateau: Depth Scaling")
    ax1.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax1)

    # (0,2) Heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    data_mat = np.outer(np.array(wvs), np.ones(len(ds)))
    im = ax2.imshow(np.log10(data_mat + 1e-20), aspect="auto",
                    extent=[min(ds)-0.5, max(ds)+0.5, min(ns_w)-0.5, max(ns_w)+0.5],
                    origin="lower", cmap="plasma")
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("log10 Var", color=P["text"])
    cbar.ax.tick_params(colors=P["text"])
    ax2.set_xlabel("Circuit Depth"); ax2.set_ylabel("Qubits")
    ax2.set_title("Gradient Variance Heatmap")
    ax2.tick_params(colors=P["text"])
    ax2.set_facecolor(P["panel"])

    # (1,0-1) 2D Loss landscape
    ax3 = fig.add_subplot(gs[1, :2])
    TH, PH = np.meshgrid(theta, theta)
    contour = ax3.contourf(TH, PH, Z_landscape.T, levels=60, cmap="viridis")
    cbar3 = fig.colorbar(contour, ax=ax3)
    cbar3.set_label("E[Z0]", color=P["text"])
    cbar3.ax.tick_params(colors=P["text"])
    ax3.contour(TH, PH, Z_landscape.T, levels=15, colors="white", alpha=0.2, linewidths=0.5)
    ax3.set_xlabel("θ₀ (rad)"); ax3.set_ylabel("θ₁ (rad)")
    ax3.set_title("2D Loss Landscape — PQC (4 qubits, 2 layers)\nFlat regions = barren plateau")
    ax3.tick_params(colors=P["text"])
    ax3.set_facecolor(P["panel"])

    # (1,2) Gradient norm distribution
    ax4 = fig.add_subplot(gs[1, 2])
    for n_q, col in [(2, P["a2"]), (4, P["a1"]), (7, P["a3"])]:
        key_h = jax.random.PRNGKey(1234 + n_q)
        grad_norms = []
        for _ in range(100):
            key_h, sk = jax.random.split(key_h)
            p_ = jax.random.uniform(sk, (n_q*4*2,), minval=0., maxval=2*jnp.pi)
            g = jax.grad(lambda pp: pqc_z0_expectation(pp, n_q, 4))(p_)
            grad_norms.append(float(jnp.linalg.norm(g)))
        ax4.hist(grad_norms, bins=20, color=col, alpha=0.7,
                 label=f"n={n_q} (mean={np.mean(grad_norms):.4f})", density=True)
    ax4.set_xlabel("|∇E|"); ax4.set_ylabel("Density")
    ax4.set_title("Gradient Norm Distribution\n(depth=4, varying width)")
    ax4.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    theme(fig, ax4)

    fig.suptitle(f"Barren Plateau Phenomenon in PQCs │ {BACKEND.upper()} │ {TS}\n"
                 f"McClean et al. (2018) Nat. Comm. 9, 4812",
                 color=P["text"], fontsize=13, fontweight="bold", y=0.97)

    path = f"tpu/plots/07_barren_plateau_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"\n  🖼  Barren plateau plot saved → {path}")

    json.dump({"experiment":"barren_plateau",
               "width_study":width_results, "depth_study":depth_results,
               "width_exp_decay_slope": float(width_fit[0]),
               "depth_exp_decay_slope": float(depth_fit[0])},
              open(f"tpu/results/barren_plateau_{TS}.json","w"), indent=2)
    print(f"  📄 JSON saved → tpu/results/barren_plateau_{TS}.json")

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8 — TPU Qubit Scaling Benchmark
# ═════════════════════════════════════════════════════════════════════════════

def get_hbm_mib():
    try:
        s = jax.devices()[0].memory_stats()
        if s and "bytes_in_use" in s:
            # Multiply by NUM_DEV to reflect total mesh memory since data is sharded
            return (s["bytes_in_use"]/1024/1024) * NUM_DEV
    except: pass
    return 0.0

def run_tpu_benchmark():
    banner(f"EXPERIMENT 5 — TPU Qubit Scaling Benchmark ({NUM_DEV} devices, {BACKEND.upper()})")

    MEM_PER_DEV_GB = 16.0               # v5e-16: 16 GB HBM per chip
    TOTAL_HBM_GB   = NUM_DEV * MEM_PER_DEV_GB   # 16 chips × 16 GB = 256 GB
    OS_RESERVE_GB  = 10.0               # flat OS/runtime headroom (user request)
    usable_gb      = TOTAL_HBM_GB - OS_RESERVE_GB  # 246 GB available

    print(f"  Total HBM           : {TOTAL_HBM_GB:.0f} GB  ({NUM_DEV} chips × {MEM_PER_DEV_GB:.0f} GB)")
    print(f"  OS/runtime reserve  : {OS_RESERVE_GB:.0f} GB  (flat, per user request)")
    print(f"  Usable for compute  : {usable_gb:.0f} GB")
    print(f"  Max safe qubit count: 34  (2^34×8 = 128 GB state vector)\n")

    HDR = ("Qubits","State Size","FWD Compile(s)","FWD Exec(s)","GRAD Compile(s)","GRAD Exec(s)","HBM Used","Throughput")
    cw  = (7,11,14,12,15,13,11,18)
    sep = "─"*(sum(cw)+len(cw)*3-1)
    def frow(*v): return " │ ".join(str(x).ljust(w) for x,w in zip(v,cw))
    print("  "+sep); print("  "+frow(*HDR)); print("  "+sep)

    results = []

    # ── Sharding layout across all TPU chips ──
    sharding = PositionalSharding(jax.devices()).reshape(NUM_DEV)

    # ── Small fixed param count: benchmark tests STATE VECTOR scaling, not params.
    #    lax.fori_loop stores N intermediate carries for gradient backprop,
    #    each of size 2^n. With 108 params that's 108 × 2^n × 4 bytes → OOM at n=29.
    #    With 6 params: 6 × 2^34 × 4 = 384 GB — tight but feasible with remat.
    MAX_PARAMS = 6

    # ── Compilation timeout (seconds)
    COMPILE_TIMEOUT = 300.0

    for n in range(10, 37):
        sb = (2**n) * 8  # complex64 = 8 bytes
        sg = sb / 1024**3
        dim = 2**n

        if sg > usable_gb:
            print("  "+sep)
            print(f"  | n={n:2d} | {fmt_bytes(sb)} > {usable_gb:.0f} GB cap -- STOPPING")
            break

        n_params = min(3 * n, MAX_PARAMS)

        def make_fwd(dim_=dim, max_p_=MAX_PARAMS):
            @jax.jit
            def fwd(params):
                # 1. Allocate & shard
                state = jnp.ones(dim_, dtype=jnp.complex64) / jnp.sqrt(dim_ + 0.0)
                state = lax.with_sharding_constraint(state, sharding)

                # 2. Index array
                idx = jnp.arange(dim_, dtype=jnp.float32) * (2 * jnp.pi / dim_)
                idx = lax.with_sharding_constraint(idx, sharding)

                # 3. Phase: fori_loop with only 6 iterations (tiny gradient overhead)
                def phase_body(k, carry):
                    phase, p, ix = carry
                    phase = phase + p[k] * jnp.sin((k + 1.0) * ix)
                    return (phase, p, ix)

                phase_init = jnp.zeros(dim_, dtype=jnp.float32)
                phase_init = lax.with_sharding_constraint(phase_init, sharding)
                phase_angles, _, _ = lax.fori_loop(
                    0, max_p_, phase_body, (phase_init, params, idx))

                state = state * jnp.exp(1j * phase_angles)
                state = jnp.roll(state, shift=dim_ // 2)

                # 4. Amplitude: fori_loop with only 6 iterations
                def amp_body(k, carry):
                    amp, p, ix = carry
                    amp = amp + 0.1 * p[k] * jnp.cos((k + 1.0) * ix)
                    return (amp, p, ix)

                amp_init = jnp.ones(dim_, dtype=jnp.float32)
                amp_init = lax.with_sharding_constraint(amp_init, sharding)
                amplitudes, _, _ = lax.fori_loop(
                    0, max_p_, amp_body, (amp_init, params, idx))

                state = state * amplitudes
                state = state / jnp.sqrt(jnp.sum(jnp.abs(state)**2) + 1e-12)

                probs = jnp.abs(state)**2
                half = dim_ // 2
                p0 = jnp.sum(probs[:half])
                p1 = jnp.sum(probs[half:])
                return jnp.real(p0 - p1)
            return fwd
        fwd_fn = make_fwd()

        def make_grad(dim_=dim, max_p_=MAX_PARAMS):
            # jax.remat checkpoints the forward: during backprop, intermediates
            # are RECOMPUTED instead of stored, saving ~50% HBM at cost of 2x compute.
            fwd = jax.remat(make_fwd(dim_, max_p_))
            return jax.jit(jax.grad(fwd))
        grad_fn = make_grad()

        params_full = jnp.ones((MAX_PARAMS,), dtype=jnp.float32) * 0.5

        # ── Forward: compile + execute ──
        hbm0 = get_hbm_mib()
        t0 = time.perf_counter()
        try:
            val = fwd_fn(params_full); val.block_until_ready()
        except Exception as e:
            print(f"  | n={n:2d} | FWD FAILED: {e}"); break
        t_fwd_compile = time.perf_counter() - t0
        hbm1 = get_hbm_mib()
        hbm_delta = max(0., hbm1 - hbm0)

        if t_fwd_compile > COMPILE_TIMEOUT:
            print(f"  | n={n:2d} | FWD compile took {t_fwd_compile:.0f}s > {COMPILE_TIMEOUT:.0f}s -- STOPPING")
            break

        # Forward exec (cached)
        fwd_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            val = fwd_fn(params_full); val.block_until_ready()
            fwd_times.append(time.perf_counter() - t0)
        t_fwd = float(np.mean(fwd_times))

        # ── Gradient: compile + execute (may OOM at large n — that's OK) ──
        t_grad_compile = -1.0
        t_grad = -1.0
        grad_ok = True
        t0 = time.perf_counter()
        try:
            g = grad_fn(params_full); g.block_until_ready()
            t_grad_compile = time.perf_counter() - t0
        except Exception as e:
            t_grad_compile = time.perf_counter() - t0
            print(f"  | n={n:2d} | GRAD OOM (expected at large n): {type(e).__name__}")
            grad_ok = False

        if grad_ok and t_grad_compile > COMPILE_TIMEOUT:
            print(f"  | n={n:2d} | GRAD compile took {t_grad_compile:.0f}s — slow but continuing FWD-only")
            grad_ok = False

        if grad_ok:
            grad_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                g = grad_fn(params_full); g.block_until_ready()
                grad_times.append(time.perf_counter() - t0)
            t_grad = float(np.mean(grad_times))

        ops_per_fwd = dim * 6
        throughput  = ops_per_fwd / t_fwd if t_fwd > 0 else 0

        r = {"n_qubits":n, "state_size_bytes":sb, "state_size_str":fmt_bytes(sb),
             "num_ops": ops_per_fwd,
             "t_fwd_compile_s":t_fwd_compile, "t_fwd_exec_s":t_fwd,
             "t_grad_compile_s":t_grad_compile if grad_ok else -1,
             "t_grad_exec_s":t_grad if grad_ok else -1,
             "grad_ok": grad_ok,
             "hbm_used_mib":hbm_delta, "hbm_total_gb":TOTAL_HBM_GB,
             "throughput_ops_s": throughput}
        results.append(r)

        grad_compile_str = f"{t_grad_compile:.3f}" if grad_ok else "OOM"
        grad_exec_str    = f"{t_grad:.5f}" if grad_ok else "OOM"

        print("  "+frow(
            n, fmt_bytes(sb),
            f"{t_fwd_compile:.3f}", f"{t_fwd:.5f}",
            grad_compile_str, grad_exec_str,
            f"{hbm_delta:.1f} MiB" if hbm_delta>0 else "N/A",
            f"{throughput/1e9:.2f}G/s" if throughput>=1e9 else
            f"{throughput/1e6:.2f}M/s" if throughput>=1e6 else f"{throughput:.1f}/s"
        ))
        sys.stdout.flush()

    print("  "+sep)
    if not results: print("  No results collected."); return
    print(f"\n  Peak qubits benchmarked: {results[-1]['n_qubits']} "
          f"({results[-1]['state_size_str']})\n")

    # Save CSV + JSON
    csv_path = f"tpu/results/tpu_benchmark_{TS}.csv"
    with open(csv_path,"w",newline="") as f:
        csv.DictWriter(f, fieldnames=results[0].keys()).writeheader()
        csv.DictWriter(f, fieldnames=results[0].keys()).writerows(results)
    print(f"  📄 CSV  → {csv_path}")
    meta = {"timestamp":TS,"backend":BACKEND,"devices":NUM_DEV,
            "usable_gb":usable_gb,"results":results}
    json.dump(meta, open(f"tpu/results/tpu_benchmark_{TS}.json","w"), indent=2)
    print(f"  📄 JSON → tpu/results/tpu_benchmark_{TS}.json")

    # 6-panel benchmark plot
    ns   = [r["n_qubits"]         for r in results]
    fwdc = [r["t_fwd_compile_s"]  for r in results]
    fwde = [r["t_fwd_exec_s"]     for r in results]
    # Gradient data: only where gradient succeeded
    grad_results = [r for r in results if r.get("grad_ok", True)]
    ns_g = [r["n_qubits"]         for r in grad_results]
    grdc = [r["t_grad_compile_s"] for r in grad_results]
    grde = [r["t_grad_exec_s"]    for r in grad_results]
    smb  = [r["state_size_bytes"]/(1<<20) for r in results]
    hbm  = [r["hbm_used_mib"]     for r in results]
    tput = [r["throughput_ops_s"]  for r in results]

    fig = plt.figure(figsize=(18,14), facecolor=P["bg"])
    gsp = gridspec.GridSpec(3,2,figure=fig,hspace=0.48,wspace=0.35,
                            left=0.07,right=0.97,top=0.92,bottom=0.06)

    # (0,0) Forward + Gradient execution time
    ax0 = fig.add_subplot(gsp[0,0])
    ax0.semilogy(ns, fwde, "o-", color=P["a1"], lw=2.5, ms=7, label="Forward exec")
    if ns_g:
        ax0.semilogy(ns_g, grde, "s-", color=P["a3"], lw=2.5, ms=7, label="Gradient exec")
    ax0.set_xlabel("Qubits"); ax0.set_ylabel("Time (s) [log]")
    ax0.set_title("Execution Time Scaling (FWD + GRAD)")
    ax0.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    ax0.set_xticks(ns); theme(fig,ax0)

    # (0,1) Compile vs Execute time
    ax1 = fig.add_subplot(gsp[0,1])
    ax1.semilogy(ns, fwdc, "o--", color=P["a5"], lw=2, ms=6, label="FWD compile")
    ax1.semilogy(ns, fwde, "o-",  color=P["a1"], lw=2, ms=6, label="FWD exec")
    if ns_g:
        ax1.semilogy(ns_g, grdc, "s--", color=P["a4"], lw=2, ms=6, label="GRAD compile")
        ax1.semilogy(ns_g, grde, "s-",  color=P["a3"], lw=2, ms=6, label="GRAD exec")
    ax1.set_xlabel("Qubits"); ax1.set_ylabel("Time (s) [log]")
    ax1.set_title("Compile vs Execute Time Breakdown")
    ax1.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=8)
    ax1.set_xticks(ns); theme(fig,ax1)

    # (1,0) State-vector memory footprint
    ax2 = fig.add_subplot(gsp[1,0])
    ax2.semilogy(ns, smb, "o-", color=P["a4"], lw=2.5, ms=7)
    ax2.axhline(TOTAL_HBM_GB*1024, color=P["a3"], ls="--", lw=1.5,
                label=f"Total HBM ({TOTAL_HBM_GB:.0f} GB = {NUM_DEV} x 16 GB)")
    ax2.axhline(usable_gb*1024, color=P["a5"], ls=":", lw=1.5,
                label=f"Usable cap ({usable_gb:.0f} GB)")
    ax2.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    ax2.set_xlabel("Qubits"); ax2.set_ylabel("State-Vector (MiB) [log]")
    ax2.set_title("Memory Footprint (2^n x 8 bytes)")
    ax2.set_xticks(ns); theme(fig,ax2)

    # (1,1) Throughput
    ax3 = fig.add_subplot(gsp[1,1])
    ax3.plot(ns, [t/1e9 for t in tput], "s-", color=P["a5"], lw=2.5, ms=7)
    ax3.set_xlabel("Qubits"); ax3.set_ylabel("Throughput (Gops/s)")
    ax3.set_title("Quantum State-Vector Throughput (JIT)")
    ax3.set_xticks(ns); theme(fig,ax3)

    # (2,0) HBM allocation delta
    ax4 = fig.add_subplot(gsp[2,0])
    ax4.bar(ns, [v if v>0 else 0 for v in hbm], color=P["a1"], alpha=0.8,
            edgecolor=P["border"])
    ax4.set_xlabel("Qubits"); ax4.set_ylabel("HBM Delta (MiB)")
    ax4.set_title(f"HBM Allocation Delta ({NUM_DEV}-chip cluster)")
    ax4.set_xticks(ns); theme(fig,ax4)

    # (2,1) Exponential scaling law
    ax5 = fig.add_subplot(gsp[2,1])
    if len(ns_g) >= 2:
        lt  = np.log2(np.array(grde) + 1e-12)
        cf  = np.polyfit(ns_g, lt, 1)
        nf  = np.linspace(min(ns_g), max(ns_g), 200)
        ax5.scatter(ns_g, grde, color=P["a1"], s=55, zorder=5, label="GRAD exec data")
        ax5.plot(nf, 2**np.poly1d(cf)(nf), "-", color=P["a3"], lw=2.5,
                 label=f"Exp fit: 2^({cf[0]:.3f}n)")
    slope_str = f"slope = {cf[0]:.3f}" if len(ns_g) >= 2 else "N/A (gradient OOM)"
    ax5.set_yscale("log"); ax5.set_xlabel("Qubits"); ax5.set_ylabel("Time (s) [log]")
    ax5.set_title(f"Exponential Scaling Law ({slope_str})")
    ax5.legend(facecolor=P["panel"],edgecolor=P["border"],labelcolor=P["text"],fontsize=9)
    ax5.set_xticks(ns_g if ns_g else ns); theme(fig,ax5)

    TPU_INFO = (f"Google Cloud TPU v5e-16  |  {NUM_DEV} chips × {MEM_PER_DEV_GB:.0f} GB HBM2e  |  "
                f"{TOTAL_HBM_GB:.0f} GB total  |  {usable_gb:.0f} GB usable")
    fig.suptitle(
        f"JAX TPU Quantum Scaling Benchmark  |  {BACKEND.upper()}  |  {TS}\n"
        f"{TPU_INFO}  |  peak n={results[-1]['n_qubits']} qubits ({results[-1]['state_size_str']})",
        color=P["text"], fontsize=12, fontweight="bold", y=0.98)

    # Add TPU info watermark to each panel
    for ax_i in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax_i.annotate(f"TPU v5e-16 · {NUM_DEV} chips", xy=(0.98, 0.02),
                      xycoords="axes fraction", ha="right", va="bottom",
                      fontsize=7, color=P["sub"], alpha=0.6)

    path = f"tpu/plots/tpu_benchmark_{TS}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=P["bg"]); plt.close()
    print(f"  Benchmark plot saved -> {path}")

# ═════════════════════════════════════════════════════════════════════════════
# Tee class — duplicates stdout to both console and a log file
# ═════════════════════════════════════════════════════════════════════════════

class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath, mode="w"):
        self._file = open(filepath, mode, encoding="utf-8", errors="replace")
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

# ═════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Redirect all output to both console AND a txt log file ──
    LOG_PATH = f"tpu/results/run_output_{TS}.txt"
    tee = Tee(LOG_PATH)
    sys.stdout = tee

    banner(f"JAX QUANTUM RESEARCH SUITE — TPU v5e-16 Edition  │  {TS}")
    print(f"  Backend       : {BACKEND.upper()}")
    print(f"  Devices       : {NUM_DEV}")
    print(f"  TPU type      : Google Cloud TPU v5e-16")
    print(f"  HBM per chip  : 16 GB HBM2e")
    print(f"  Total HBM     : {NUM_DEV * 16} GB")
    for i,d in enumerate(DEVICES): print(f"    [{i:2d}] {d}")

    t_total = time.perf_counter()

    run_state_prep()
    run_vqc()
    run_vqe()
    run_qaoa()
    run_noise_sim()
    run_nisq_benchmark()
    run_barren_plateau()
    run_tpu_benchmark()

    banner(f"ALL EXPERIMENTS COMPLETE — total time: {time.perf_counter()-t_total:.1f}s")
    print(f"  📁 Results   → tpu/results/")
    print(f"  🖼  Plots     → tpu/plots/")
    print(f"  📝 Full log  → {LOG_PATH}")
    print(f"  🕐 Timestamp : {TS}\n")

    tee.close()
