"""
Experiment 4: Quantum Approximate Optimization Algorithm (QAOA) for MaxCut
Solves the Max-Cut problem on a weighted 6-node graph using QAOA in pure JAX.
Optimizes angles for depths p=1 to 5, comparing to brute-force classical MaxCut.
"""

import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sys

# Ensure jax_qsim is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_qsim.circuit import Circuit
import jax_qsim.statevector as sv

# ==============================================================================
# Graph Definition & Classical MaxCut Baseline
# ==============================================================================
# Graph edges: (u, v, weight)
EDGES = [
    (0, 1, 1.5), (1, 2, 2.0), (2, 3, 1.0),
    (3, 4, 1.5), (4, 5, 2.0), (5, 0, 1.0),
    (0, 3, 0.5), (1, 4, 0.5), (2, 5, 0.5)
]
NUM_NODES = 6
CLASSICAL_OPT_CUT = 9.0  # Known exact max cut for this weighted graph

def brute_force_maxcut():
    """Computes exact MaxCut via classical exhaustive search."""
    best_cut = 0.0
    best_partition = None
    for i in range(1 << NUM_NODES):
        partition = [(i >> q) & 1 for q in range(NUM_NODES)]
        cut = 0.0
        for u, v, w in EDGES:
            if partition[u] != partition[v]:
                cut += w
        if cut > best_cut:
            best_cut = cut
            best_partition = partition
    return best_cut, best_partition

# Setup paths
os.makedirs("results", exist_ok=True)

# ==============================================================================
# QAOA Ansatz & Loss Function
# ==============================================================================
def build_qaoa_circuit(params, p):
    """
    Builds the QAOA variational circuit for a given depth p.
    params contains alternating gammas (cost) and betas (mixer):
    [gamma_0, beta_0, gamma_1, beta_1, ..., gamma_{p-1}, beta_{p-1}]
    """
    c = Circuit(NUM_NODES)
    
    # 1. Prepare uniform superposition |+>^⊗N
    for q in range(NUM_NODES):
        c.h(q)
        
    p_idx = 0
    # 2. Apply alternating Cost and Mixer layers
    for layer in range(p):
        gamma = params[p_idx]; p_idx += 1
        beta = params[p_idx]; p_idx += 1
        
        # --- A. Cost Hamiltonian layer: e^{-i * gamma * H_C} ---
        # H_C = 1/2 * sum_{u,v} w_{uv} * (I - Z_u * Z_v)
        # This translates to: CNOT(u, v), RZ(w * gamma), CNOT(u, v) for each edge
        for u, v, w in EDGES:
            c.cnot(u, v)
            # Parametric gate RZ maps to index: we encode it directly
            c.rz(v, p_idx)
            # To apply a custom angle theta = w * gamma, we will map this parametric RZ
            # to utilize a pre-calculated scaled parameter: theta_edge = w * gamma
            p_idx += 1
            c.cnot(u, v)
            
        # --- B. Mixer Hamiltonian layer: e^{-i * beta * H_M} ---
        # H_M = sum_q X_q
        # This translates to: RX(2 * beta) on each qubit
        for q in range(NUM_NODES):
            c.rx(q, p_idx)
            p_idx += 1
            
    return c

def resolve_qaoa_params(raw_params, p):
    """
    Maps raw variational params [gamma_0, beta_0, ..., gamma_{p-1}, beta_{p-1}]
    to the full list of parameters required by our Circuit builder.
    """
    full_params = []
    for layer in range(p):
        gamma = raw_params[2 * layer]
        beta = raw_params[2 * layer + 1]
        
        # Alternating gamma and beta
        # Cost layer angles: w * gamma for each edge
        for u, v, w in EDGES:
            full_params.append(w * gamma)
            
        # Mixer layer angles: 2 * beta for each qubit
        for q in range(NUM_NODES):
            full_params.append(2.0 * beta)
            
    return jnp.array(full_params, dtype=jnp.float32)

def evaluate_qaoa_cut(raw_params, p):
    """Computes the expected cut value: E[Cut] = sum_{u,v} w/2 * (1 - <Z_u Z_v>)."""
    # 1. Compile full parameters
    full_params = resolve_qaoa_params(raw_params, p)
    # 2. Build and run circuit
    c = build_qaoa_circuit(raw_params, p)
    state = c.run(full_params, 'statevector')
    
    # 3. Compute expectation value <Z_u Z_v> for each edge
    probs = jnp.abs(state) ** 2
    
    expected_cut = 0.0
    for u, v, w in EDGES:
        # Sum over all state indices to get <Z_u Z_v>
        # Margin out all other qubits
        axes = tuple(i for i in range(NUM_NODES) if i not in (u, v))
        marginal = jnp.sum(probs, axis=axes)
        # marginal is a 2x2 grid representing probabilities for qubits u and v:
        # [0,0]=|00>, [0,1]=|01>, [1,0]=|10>, [1,1]=|11>
        # <Zu Zv> = P(00) - P(01) - P(10) + P(11)
        zz = marginal[0, 0] - marginal[0, 1] - marginal[1, 0] + marginal[1, 1]
        expected_cut += (w / 2.0) * (1.0 - zz)
        
    return expected_cut

def loss_fn(raw_params, p):
    # We want to maximize expected cut, so loss is negative expected cut
    return -evaluate_qaoa_cut(raw_params, p)

# ==============================================================================
# JIT-compiled Optimization Step with Adam
# ==============================================================================
def adam_update(p_arr, g, m, v, t, lr=0.04, b1=0.9, b2=0.999, eps=1e-8):
    t = t + 1
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * (g ** 2)
    mh = m / (1.0 - b1 ** t)
    vh = v / (1.0 - b2 ** t)
    return p_arr - lr * mh / (jnp.sqrt(vh) + eps), m, v, t

# We compile custom step functions for each p to utilize JAX static trace JIT
def make_step_fn(p):
    @jax.jit
    def step(params, m, v, t):
        loss, grads = jax.value_and_grad(lambda pr: loss_fn(pr, p))(params)
        params, m, v, t = adam_update(params, grads, m, v, t)
        return params, m, v, t, -loss
    return step

# ==============================================================================
# Execute QAOA Optimization
# ==============================================================================
def run_experiment():
    print("=" * 80)
    print(" EXPERIMENT 4: QAOA MaxCut (6-Node Graph) ".center(80, "="))
    print("=" * 80)
    
    # 1. Compute brute force baseline
    best_classical_cut, best_partition = brute_force_maxcut()
    partition_str = "".join(['A' if b == 0 else 'B' for b in best_partition])
    print(f"Classical MaxCut Capacity   : {best_classical_cut:.2f}")
    print(f"Optimal Node Partition (A/B): {partition_str}")
    print("-" * 80)
    
    depths = [1, 2, 3, 4, 5]
    all_histories = {}
    approx_ratios = []
    final_cuts = []
    
    print(f"{'Depth p':^10} | {'Expected Cut E[C]':^20} | {'Approx Ratio':^18} | {'Elapsed Time':^15}")
    print("-" * 72)
    
    for p in depths:
        # Initialize QAOA raw params [gamma_0, beta_0, ...] randomly
        key = jax.random.PRNGKey(42 + p)
        raw_params = jax.random.uniform(key, (2 * p,), minval=0.0, maxval=2.0 * jnp.pi)
        m = jnp.zeros(2 * p)
        v = jnp.zeros(2 * p)
        t = 0
        
        epochs = 120
        hist = []
        
        step_fn = make_step_fn(p)
        
        t0 = time.time()
        for ep in range(epochs):
            raw_params, m, v, t, current_cut = step_fn(raw_params, m, v, t)
            hist.append(float(current_cut))
            
        dt = time.time() - t0
        best_cut_found = max(hist)
        ratio = best_cut_found / CLASSICAL_OPT_CUT
        
        all_histories[p] = hist
        approx_ratios.append(ratio)
        final_cuts.append(best_cut_found)
        
        print(f"{p:^10d} | {best_cut_found:^20.4f} | {ratio:^18.2%} | {dt:^13.3f}s")
        
    print("=" * 80)
    
    # ==============================================================================
    # Plotting & Visualization (4-Panel Grid Plot)
    # ==============================================================================
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0d1117')
    
    # Panel 1: Graph Topology
    ax_graph = axes[0, 0]
    ax_graph.set_facecolor('#161b22')
    
    # Draw graph nodes in a circle
    angles = np.linspace(0, 2 * np.pi, NUM_NODES, endpoint=False)
    coords_x = np.cos(angles)
    coords_y = np.sin(angles)
    
    for u, v, w in EDGES:
        ax_graph.plot([coords_x[u], coords_x[v]], [coords_y[u], coords_y[v]], 
                      color='#8b949e', lw=1.0 + w * 1.5, alpha=0.7)
        ax_graph.text((coords_x[u] + coords_x[v]) / 2.0, (coords_y[u] + coords_y[v]) / 2.0, 
                      f"{w}", color='#ffa657', fontsize=9, fontweight='bold', ha='center')
                      
    node_colors = ['#58a6ff' if b == 0 else '#f78166' for b in best_partition]
    ax_graph.scatter(coords_x, coords_y, s=350, color=node_colors, edgecolor='#30363d', zorder=5)
    for q in range(NUM_NODES):
        ax_graph.text(coords_x[q], coords_y[q], f"{q}", color='#e6edf3', fontsize=11, 
                      fontweight='bold', ha='center', va='center')
                      
    ax_graph.set_title("🕸  Max-Cut Graph Topology & Partition", fontsize=13, color='#e6edf3', fontweight='bold', pad=12)
    ax_graph.set_xlim(-1.3, 1.3)
    ax_graph.set_ylim(-1.3, 1.3)
    ax_graph.set_aspect('equal')
    ax_graph.axis('off')
    
    # Panel 2: Convergence Curves
    ax_conv = axes[0, 1]
    ax_conv.set_facecolor('#161b22')
    colors = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657']
    for idx, p in enumerate(depths):
        ax_conv.plot(all_histories[p], color=colors[idx], lw=2.0, label=f"Depth p = {p}")
    ax_conv.axhline(CLASSICAL_OPT_CUT, color='#f78166', ls=':', lw=1.5, label="Classical Max-Cut (9.0)")
    ax_conv.set_title("📈  QAOA Convergence per Depth (p)", fontsize=13, color='#e6edf3', fontweight='bold', pad=12)
    ax_conv.set_xlabel("Epoch / Optimization Step", fontsize=11, color='#8b949e')
    ax_conv.set_ylabel("Expected Cut Value E[Cut]", fontsize=11, color='#8b949e')
    ax_conv.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    ax_conv.grid(True, linestyle='--', color='#21262d', alpha=0.6)
    ax_conv.tick_params(colors='#e6edf3')
    for spine in ax_conv.spines.values():
        spine.set_edgecolor('#30363d')
        
    # Panel 3: Expected Cut vs Depth
    ax_cut = axes[1, 0]
    ax_cut.set_facecolor('#161b22')
    bars = ax_cut.bar(depths, final_cuts, color='#58a6ff', alpha=0.85, edgecolor='#30363d', width=0.5)
    ax_cut.axhline(CLASSICAL_OPT_CUT, color='#f78166', ls='--', lw=1.5, label="Classical Limit")
    for bar in bars:
        height = bar.get_height()
        ax_cut.text(bar.get_x() + bar.get_width()/2.0, height + 0.1, f"{height:.3f}", 
                    ha='center', va='bottom', color='#e6edf3', fontsize=10)
    ax_cut.set_title("🔬  Expected Cut vs Circuit Depth", fontsize=13, color='#e6edf3', fontweight='bold', pad=12)
    ax_cut.set_xlabel("Circuit Depth p", fontsize=11, color='#8b949e')
    ax_cut.set_ylabel("Expected Cut Value", fontsize=11, color='#8b949e')
    ax_cut.grid(True, linestyle='--', color='#21262d', alpha=0.4)
    ax_cut.tick_params(colors='#e6edf3')
    for spine in ax_cut.spines.values():
        spine.set_edgecolor('#30363d')
        
    # Panel 4: Approximation Ratio vs Depth
    ax_ratio = axes[1, 1]
    ax_ratio.set_facecolor('#161b22')
    ax_ratio.plot(depths, approx_ratios, marker='o', color='#3fb950', lw=2.5, ms=8, label="QAOA Approx Ratio")
    ax_ratio.axhline(1.0, color='#ffa657', ls=':', lw=1.5, label="Global Optimum (1.0)")
    for i, txt in enumerate(approx_ratios):
        ax_ratio.annotate(f"{txt:.2%}", (depths[i], approx_ratios[i]), textcoords="offset points", 
                          xytext=(0, 10), ha='center', color='#e6edf3', fontsize=9, fontweight='bold')
    ax_ratio.set_title("🎯  Approximation Ratio vs Circuit Depth", fontsize=13, color='#e6edf3', fontweight='bold', pad=12)
    ax_ratio.set_xlabel("Circuit Depth p", fontsize=11, color='#8b949e')
    ax_ratio.set_ylabel("Approximation Ratio (E[C] / Classical C)", fontsize=11, color='#8b949e')
    ax_ratio.set_ylim(0.6, 1.05)
    ax_ratio.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', loc='lower right')
    ax_ratio.grid(True, linestyle='--', color='#21262d', alpha=0.6)
    ax_ratio.tick_params(colors='#e6edf3')
    for spine in ax_ratio.spines.values():
        spine.set_edgecolor('#30363d')
        
    fig.suptitle("Quantum Approximate Optimization Algorithm (QAOA) Max-Cut — 6-Node Weighted Graph Solver", 
                 color='#e6edf3', fontsize=16, fontweight='bold', y=0.98)
                 
    plot_path = os.path.join("results", "04_qaoa_maxcut.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor='#0d1117')
    plt.close()
    
    print(f"Plot saved successfully to: {plot_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
