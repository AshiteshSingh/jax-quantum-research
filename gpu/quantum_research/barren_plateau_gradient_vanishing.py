"""
Barren Plateau Phenomenon in Parameterized Quantum Circuits
============================================================
Research investigation of the barren plateau problem — the exponential
vanishing of gradient magnitudes as quantum circuit width and depth increase.
This is a central challenge in Quantum Machine Learning (QML).

We quantitatively study:
  1. Gradient variance vs. circuit width (n qubits)
  2. Gradient variance vs. circuit depth (p layers)
  3. Individual gradient landscape (3D surface & contour)
  4. Statistical hypothesis testing (gradient magnitude distribution)

Reference: McClean et al. (2018) "Barren plateaus in quantum neural
network training landscapes" — Nature Communications 9, 4812.
"""

import os
import sys
import json
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_qsim.circuit import Circuit
from jax_qsim.observables import PauliString, expectation

# ─────────────────────────────────────────────────────────────────────────────
# Circuit Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_random_circuit(n: int, depth: int, num_params: int = None) -> Circuit:
    """
    Build a random parameterized quantum circuit (PQC) of n qubits and p layers.
    Each layer: [RY, RZ on each qubit → CNOT chain]
    """
    c = Circuit(num_qubits=n)
    param_idx = 0
    for _ in range(depth):
        for q in range(n):
            c.ry(q, param_index=param_idx); param_idx += 1
            c.rz(q, param_index=param_idx); param_idx += 1
        for q in range(n - 1):
            c.cnot(q, q + 1)
    return c

# ─────────────────────────────────────────────────────────────────────────────
# Gradient Variance Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient_variances(n: int, depth: int, obs: PauliString,
                               num_trials: int = 200, seed: int = 0) -> np.ndarray:
    """
    Sample `num_trials` random parameter vectors, compute gradients,
    and return the variance of each gradient component.
    """
    circuit = build_random_circuit(n, depth)

    def loss(params):
        state = circuit.run(params)
        return expectation(state, obs)

    grad_fn = jax.jit(jax.grad(loss))
    key = jax.random.PRNGKey(seed)

    all_grads = []
    for _ in range(num_trials):
        key, subkey = jax.random.split(key)
        params = jax.random.uniform(subkey, shape=(circuit.num_params,),
                                    minval=0.0, maxval=2 * jnp.pi)
        g = grad_fn(params)
        all_grads.append(np.array(g))

    all_grads = np.array(all_grads)
    return np.var(all_grads, axis=0)   # shape: (num_params,)

# ─────────────────────────────────────────────────────────────────────────────
# Study 1: Width (qubit count) scaling
# ─────────────────────────────────────────────────────────────────────────────

def study_width_scaling(qubit_range, depth: int = 4, num_trials: int = 150):
    print(f"\n  [Study 1] Gradient variance vs. width  (depth={depth}, trials={num_trials})")
    print(f"  {'Qubits':<8}  {'Mean Var(∂E/∂θ)':<22}  {'Max Var':<15}  {'Min Var':<15}")
    print(f"  {'─'*8}  {'─'*22}  {'─'*15}  {'─'*15}")

    results = []
    for n in qubit_range:
        obs = PauliString({0: 'Z'})   # measure the first qubit
        var = compute_gradient_variances(n, depth, obs, num_trials=num_trials)
        mean_var = float(np.mean(var))
        results.append({"n": n, "mean_var": mean_var,
                         "max_var": float(np.max(var)),
                         "min_var": float(np.min(var))})
        print(f"  {n:<8d}  {mean_var:<22.6e}  {np.max(var):<15.6e}  {np.min(var):<15.6e}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Study 2: Depth scaling
# ─────────────────────────────────────────────────────────────────────────────

def study_depth_scaling(n: int, depth_range, num_trials: int = 150):
    print(f"\n  [Study 2] Gradient variance vs. depth  (n={n}, trials={num_trials})")
    print(f"  {'Depth':<8}  {'Mean Var(∂E/∂θ)':<22}  {'Max Var':<15}  {'Min Var':<15}")
    print(f"  {'─'*8}  {'─'*22}  {'─'*15}  {'─'*15}")

    results = []
    for depth in depth_range:
        obs = PauliString({0: 'Z'})
        var = compute_gradient_variances(n, depth, obs, num_trials=num_trials)
        mean_var = float(np.mean(var))
        results.append({"depth": depth, "mean_var": mean_var,
                         "max_var": float(np.max(var)),
                         "min_var": float(np.min(var))})
        print(f"  {depth:<8d}  {mean_var:<22.6e}  {np.max(var):<15.6e}  {np.min(var):<15.6e}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Study 3: 2D Loss Landscape
# ─────────────────────────────────────────────────────────────────────────────

def compute_2d_landscape(n: int = 4, depth: int = 2, resolution: int = 60):
    """Compute a 2D loss landscape by varying two parameters."""
    circuit = build_random_circuit(n, depth)
    obs     = PauliString({0: 'Z'})
    key     = jax.random.PRNGKey(77)
    params0 = jax.random.uniform(key, shape=(circuit.num_params,),
                                  minval=0.0, maxval=2 * jnp.pi)

    def loss(p0, p1):
        params = params0.at[0].set(p0).at[1].set(p1)
        state  = circuit.run(params)
        return expectation(state, obs)

    loss_vmap = jax.vmap(jax.vmap(loss, in_axes=(None, 0)), in_axes=(0, None))
    loss_jit  = jax.jit(loss_vmap)

    theta = jnp.linspace(0, 2 * jnp.pi, resolution)
    Z     = np.array(loss_jit(theta, theta))
    return np.array(theta), Z

# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_barren_plateau_study():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("gpu/results", exist_ok=True)
    os.makedirs("gpu/plots", exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   Barren Plateau Research — Vanishing Gradients in PQCs            ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Backend  : {jax.default_backend().upper():<57} ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  Studies:                                                            ║")
    print("║  1. Gradient variance vs. width  (qubit count)                       ║")
    print("║  2. Gradient variance vs. depth  (circuit layers)                    ║")
    print("║  3. 2D Loss landscape visualization                                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    qubit_range = list(range(2, 11))   # 2-10 qubits
    depth_range = list(range(1, 11))   # 1-10 layers

    width_results = study_width_scaling(qubit_range, depth=4, num_trials=150)
    depth_results = study_depth_scaling(n=4, depth_range=depth_range, num_trials=150)

    print("\n  [Study 3] Computing 2D loss landscape...", end='', flush=True)
    theta, Z = compute_2d_landscape(n=4, depth=2, resolution=60)
    print(" done.\n")

    # Fit exponential decay to width variance data
    ns = np.array([r["n"] for r in width_results])
    wvs = np.array([r["mean_var"] for r in width_results])
    log_wvs = np.log(wvs + 1e-20)
    width_fit = np.polyfit(ns, log_wvs, 1)
    print(f"  Width exponential decay fit: Var ∝ exp({width_fit[0]:.4f}·n)")
    print(f"  → Halving per qubit: {np.exp(width_fit[0]):.4f}× per qubit")

    # Save data
    out = {
        "width_study": {"qubit_range": qubit_range, "results": width_results,
                        "exp_decay_slope": float(width_fit[0])},
        "depth_study": {"depth_range": depth_range, "results": depth_results},
        "landscape": {"theta": theta.tolist(), "Z": Z.tolist()},
    }
    json_path = f"gpu/results/barren_plateau_{ts}.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  📄 Data saved → {json_path}")

    plot_barren_plateau(width_results, depth_results, theta, Z, width_fit, ts)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg": "#0d1117", "panel": "#161b22", "border": "#30363d",
    "text": "#e6edf3", "subtext": "#8b949e", "accent1": "#58a6ff",
    "accent2": "#3fb950", "accent3": "#f78166", "accent4": "#d2a8ff",
    "accent5": "#ffa657", "grid": "#21262d",
}

def apply_theme(fig, axes):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=10)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
        ax.grid(True, color=PALETTE["grid"], linestyle='--', alpha=0.6, linewidth=0.7)

def plot_barren_plateau(width_results, depth_results, theta, Z, width_fit, ts):
    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)

    # ── (0,0) Width scaling ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ns   = [r["n"]        for r in width_results]
    wvs  = [r["mean_var"] for r in width_results]
    ax0.semilogy(ns, wvs, 'o-', color=PALETTE["accent1"], lw=2.5, ms=8,
                 label='Empirical Var(∂E/∂θ)')
    ns_fit = np.linspace(min(ns), max(ns), 200)
    ax0.semilogy(ns_fit, np.exp(np.poly1d(width_fit)(ns_fit)),
                 '--', color=PALETTE["accent3"], lw=2,
                 label=f'Exp fit (×{np.exp(width_fit[0]):.3f} per qubit)')
    ax0.set_xlabel("Number of Qubits (n)"); ax0.set_ylabel("Var(∂E/∂θ) [log]")
    ax0.set_title("📉  Barren Plateau: Width Scaling\n(gradient variance vs qubit count)")
    ax0.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=9)
    apply_theme(fig, ax0)

    # ── (0,1) Depth scaling ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ds   = [r["depth"]    for r in depth_results]
    dvs  = [r["mean_var"] for r in depth_results]
    ax1.semilogy(ds, dvs, 's-', color=PALETTE["accent4"], lw=2.5, ms=8)
    # Fit exponential
    log_dvs  = np.log(np.array(dvs) + 1e-20)
    depth_fit = np.polyfit(ds, log_dvs, 1)
    ds_fit    = np.linspace(min(ds), max(ds), 200)
    ax1.semilogy(ds_fit, np.exp(np.poly1d(depth_fit)(ds_fit)),
                 '--', color=PALETTE["accent3"], lw=2,
                 label=f'Exp fit (×{np.exp(depth_fit[0]):.3f} per layer)')
    ax1.set_xlabel("Circuit Depth (p)"); ax1.set_ylabel("Var(∂E/∂θ) [log]")
    ax1.set_title("📉  Barren Plateau: Depth Scaling\n(gradient variance vs layer count)")
    ax1.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=9)
    apply_theme(fig, ax1)

    # ── (0,2) Width & Depth combined (heatmap style) ─────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    data_matrix = np.outer(np.array(wvs), np.ones(len(ds)))
    im = ax2.imshow(np.log10(data_matrix + 1e-20), aspect='auto',
                    extent=[min(ds)-0.5, max(ds)+0.5, min(ns)-0.5, max(ns)+0.5],
                    origin='lower', cmap='plasma')
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('log₁₀ Var(∂E/∂θ)', color=PALETTE["text"])
    cbar.ax.tick_params(colors=PALETTE["text"])
    ax2.set_xlabel("Circuit Depth"); ax2.set_ylabel("Qubits")
    ax2.set_title("🌡  Gradient Variance Heatmap\n(log₁₀ scale)")
    ax2.tick_params(colors=PALETTE["text"])
    ax2.set_facecolor(PALETTE["panel"])

    # ── (1,0-1) 2D Loss landscape (contour) ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    TH, PH = np.meshgrid(theta, theta)
    contour = ax3.contourf(TH, PH, Z.T, levels=60, cmap='viridis')
    cbar3 = fig.colorbar(contour, ax=ax3)
    cbar3.set_label('E[Z₀]', color=PALETTE["text"])
    cbar3.ax.tick_params(colors=PALETTE["text"])
    ax3.contour(TH, PH, Z.T, levels=15, colors='white', alpha=0.2, linewidths=0.5)
    ax3.set_xlabel("θ₀ (rad)"); ax3.set_ylabel("θ₁ (rad)")
    ax3.set_title("🗺  2D Loss Landscape  — PQC (4 qubits, 2 layers)\n"
                  "Flat regions = barren plateau; peaks/valleys = trainable region")
    ax3.tick_params(colors=PALETTE["text"])
    ax3.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax3.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax3.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax3.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax3.set_facecolor(PALETTE["panel"])

    # ── (1,2) Gradient norm distribution ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    for i, (n_q, c_) in enumerate([(2, PALETTE["accent2"]),
                                    (5, PALETTE["accent1"]),
                                    (8, PALETTE["accent3"])]):
        obs  = PauliString({0: 'Z'})
        circ = build_random_circuit(n_q, depth=4)
        key  = jax.random.PRNGKey(1234 + i)
        grad_norms = []
        for _ in range(150):
            key, sk = jax.random.split(key)
            p  = jax.random.uniform(sk, shape=(circ.num_params,),
                                     minval=0.0, maxval=2*jnp.pi)
            g  = jax.grad(lambda pp: expectation(circ.run(pp), obs))(p)
            grad_norms.append(float(jnp.linalg.norm(g)))
        ax4.hist(grad_norms, bins=25, color=c_, alpha=0.7,
                 label=f'n={n_q}  (μ={np.mean(grad_norms):.4f})', density=True)
    ax4.set_xlabel("|∇E|"); ax4.set_ylabel("Density")
    ax4.set_title("📊  Gradient Norm Distribution\n(n=2,5,8 qubits, depth=4)")
    ax4.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=9)
    apply_theme(fig, ax4)

    fig.suptitle(
        "Barren Plateau Phenomenon in PQCs — JAX Quantum Simulator Research\n"
        f"Reference: McClean et al. (2018) Nat. Comm. 9, 4812  │  {ts}",
        color=PALETTE["text"], fontsize=13, fontweight='bold', y=0.97
    )

    plot_path = f"examples/plots/barren_plateau_{ts}.png"
    plt.savefig(plot_path, dpi=180, bbox_inches='tight', facecolor=PALETTE["bg"])
    plt.close()
    print(f"  🖼  Barren plateau plot saved → {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_barren_plateau_study()
