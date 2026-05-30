"""
Generate publication-quality figures from real N=10 benchmark data.
Uses only matplotlib — no external dependencies beyond what is already installed.

Run from anywhere:
    python benchmarks/generate_n10_figures.py

Figures are saved to benchmarks/results/ alongside the JSON data.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

# ── portable paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_PATH = os.path.join(RESULTS_DIR, "n10_benchmark_20260530_214024.json")
OUT_DIR   = RESULTS_DIR
os.makedirs(OUT_DIR, exist_ok=True)


with open(DATA_PATH, encoding='utf-8') as f:
    data = json.load(f)

scaling = data["bench_C_scaling"]
grad    = data["bench_D_gradient"]

qubits  = sorted([int(k) for k in scaling.keys()])
means   = [scaling[str(q)]["mean_s"] * 1000 for q in qubits]   # ms
stds    = [scaling[str(q)]["std_s"]  * 1000 for q in qubits]

# ── style ───────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
ACCENT1  = "#58a6ff"   # blue
ACCENT2  = "#f78166"   # orange-red
ACCENT3  = "#3fb950"   # green
GRID_CLR = "#30363d"
TEXT_CLR = "#e6edf3"
LABEL_CLR = "#8b949e"

def style_ax(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=LABEL_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.7)

# ===========================================================================
# FIGURE 1 — Scaling sweep: execution time vs qubit count
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
style_ax(ax)

ax.semilogy(qubits, means, 'o-', color=ACCENT1, lw=2, ms=7,
            markerfacecolor=ACCENT1, markeredgecolor='white', markeredgewidth=0.8,
            label='jax_qsim (N=10 runs, post-JIT)')
ax.fill_between(qubits,
                [max(1e-4, m - s) for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15, color=ACCENT1)

# Annotate key points
for q, m in [(10, means[qubits.index(10)]),
             (15, means[qubits.index(15)]),
             (20, means[qubits.index(20)])]:
    ax.annotate(f"{q}q: {m:.2f} ms", xy=(q, m),
                xytext=(q + 0.4, m * 2.5),
                color=ACCENT1, fontsize=8,
                arrowprops=dict(arrowstyle='->', color=ACCENT1, lw=0.8))

# Exponential growth reference line
qs = np.linspace(4, 20, 200)
base = means[0] / (2**qubits[0])
ref  = base * (2 ** qs)
ax.semilogy(qs, ref, '--', color=ACCENT2, lw=1.2, alpha=0.6,
            label=r'$O(2^n)$ reference')

ax.set_xlabel("Number of Qubits  (n)", fontsize=11)
ax.set_ylabel("Execution Time (ms, log scale)", fontsize=11)
ax.set_title("Scaling of jax_qsim Statevector Simulation\n"
             "Hardware-Efficient Ansatz · 3 Layers · N=10 Timed Runs · CPU Baseline",
             fontsize=11, pad=12)
ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
ax.set_xticks(qubits)

fig.text(0.99, 0.02,
         "Source: n10_benchmark_20260530_214024.json  |  JAX 0.10.1  |  N=10 post-warmup runs",
         ha='right', fontsize=7, color=LABEL_CLR)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig_scaling_n10.png")
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved: " + out)


# ===========================================================================
# FIGURE 2 — Bar chart: raw runs for n=10, 15, 20 qubits
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=DARK_BG)
fig.suptitle("Per-Run Timing Distributions  (N=10)  —  jax_qsim HEA Circuit",
             color=TEXT_CLR, fontsize=12, y=1.01)

configs = [(10, ACCENT1), (15, ACCENT3), (20, ACCENT2)]
for ax, (q, color) in zip(axes, configs):
    style_ax(ax)
    runs_ms = [r * 1000 for r in scaling[str(q)]["runs_s"]]
    m_ms    = scaling[str(q)]["mean_s"] * 1000
    s_ms    = scaling[str(q)]["std_s"]  * 1000
    bars = ax.bar(range(1, 11), runs_ms, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axhline(m_ms, color='white', lw=1.5, ls='--', label=f"mean={m_ms:.3f} ms")
    ax.axhspan(m_ms - s_ms, m_ms + s_ms, alpha=0.12, color='white', label=f"±1σ={s_ms:.3f} ms")
    ax.set_xlabel("Run index", fontsize=9)
    ax.set_ylabel("Time (ms)", fontsize=9)
    ax.set_title(f"{q}-Qubit Circuit", color=TEXT_CLR, fontsize=10)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=8)
    ax.set_xticks(range(1, 11))

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig_per_run_n10.png")
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved: " + out)


# ===========================================================================
# FIGURE 3 — Gradient comparison: jax.grad vs PSR
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=DARK_BG)
fig.suptitle("Gradient Method Comparison  (15-Qubit VQC, 120 Parameters, N=10 Runs)",
             color=TEXT_CLR, fontsize=12)

# Left: violin / scatter of runs
ax = axes[0]
style_ax(ax)
grad_runs_ms = [r * 1000 for r in grad["jax_grad_reverse"]["runs_s"]]
psr_runs_ms  = [r * 1000 for r in grad["psr_emulation"]["runs_s"]]

ax.scatter([1]*10, grad_runs_ms, color=ACCENT1, s=60, alpha=0.85, zorder=5, label='jax.grad runs')
ax.scatter([2]*10, psr_runs_ms,  color=ACCENT2, s=60, alpha=0.85, zorder=5, label='PSR runs')

for vals, x, color in [(grad_runs_ms, 1, ACCENT1), (psr_runs_ms, 2, ACCENT2)]:
    m = np.mean(vals)
    s = np.std(vals)
    ax.errorbar(x, m, yerr=s, fmt='D', color='white', ms=9, capsize=6, lw=2, zorder=6)

ax.set_xticks([1, 2])
ax.set_xticklabels(['jax.grad\n(reverse-mode AD)', 'Parameter-Shift Rule\n(emulated, 120 params)'],
                   color=TEXT_CLR)
ax.set_ylabel("Gradient Step Time (ms)", fontsize=10)
ax.set_title("Individual Run Times", color=TEXT_CLR, fontsize=10)
ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)

# Right: mean+std bar + speedup annotation
ax = axes[1]
style_ax(ax)
methods = ['jax.grad\n(reverse-mode)', 'PSR\n(emulated)']
means_d  = [grad["jax_grad_reverse"]["mean_s"] * 1000,
            grad["psr_emulation"]["mean_s"]     * 1000]
stds_d   = [grad["jax_grad_reverse"]["std_s"] * 1000,
            grad["psr_emulation"]["std_s"]     * 1000]
colors   = [ACCENT1, ACCENT2]

bars = ax.bar(methods, means_d, color=colors, alpha=0.85, edgecolor='white', linewidth=0.7)
ax.errorbar(methods, means_d, yerr=stds_d, fmt='none', color='white', capsize=8, lw=2)

speedup = grad["psr_emulation"]["speedup_vs_grad"]
ax.annotate(
    f"PSR is {speedup:.1f}x slower\nthan jax.grad",
    xy=(1, means_d[1]), xytext=(0.5, means_d[1] * 0.65),
    color=ACCENT2, fontsize=10, ha='center',
    arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=1.2)
)
for bar, m in zip(bars, means_d):
    ax.text(bar.get_x() + bar.get_width()/2, m + 50,
            f"{m:.1f} ms", ha='center', color=TEXT_CLR, fontsize=9, fontweight='bold')

ax.set_ylabel("Mean Gradient Step Time (ms)", fontsize=10)
ax.set_title("Mean ± 1σ  (N=10 runs)", color=TEXT_CLR, fontsize=10)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig_gradient_n10.png")
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved: " + out)


# ===========================================================================
# FIGURE 4 — Summary table figure (key numbers at a glance)
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis('off')

table_data = [
    ["Benchmark", "Qubits", "Params", "Mean Time", "Std Dev", "N Runs"],
    ["Scaling sweep", "4",  "32",  f"{means[0]:.4f} ms",   f"{stds[0]:.4f} ms",  "10"],
    ["Scaling sweep", "8",  "64",  f"{means[qubits.index(8)]:.4f} ms",  f"{stds[qubits.index(8)]:.4f} ms",  "10"],
    ["Scaling sweep", "10", "80",  f"{means[qubits.index(10)]:.4f} ms", f"{stds[qubits.index(10)]:.4f} ms", "10"],
    ["Scaling sweep", "12", "96",  f"{means[qubits.index(12)]:.3f} ms", f"{stds[qubits.index(12)]:.3f} ms", "10"],
    ["Scaling sweep", "15", "120", f"{means[qubits.index(15)]:.2f} ms", f"{stds[qubits.index(15)]:.2f} ms", "10"],
    ["Scaling sweep", "18", "144", f"{means[qubits.index(18)]:.2f} ms", f"{stds[qubits.index(18)]:.2f} ms", "10"],
    ["Scaling sweep", "20", "120", f"{means[qubits.index(20)]:.2f} ms", f"{stds[qubits.index(20)]:.2f} ms", "10"],
    ["jax.grad (AD)", "15", "120", f"{grad['jax_grad_reverse']['mean_s']*1000:.2f} ms",
                                    f"{grad['jax_grad_reverse']['std_s']*1000:.2f} ms", "10"],
    ["PSR (emulated)", "15", "120", f"{grad['psr_emulation']['mean_s']*1000:.1f} ms",
                                     f"{grad['psr_emulation']['std_s']*1000:.1f} ms", "10"],
    ["20q proxy exec", "20", "80", f"{data['bench_A_proxy_20q']['jax_qsim_20q_proxy']['mean_s']*1000:.1f} ms",
                                    f"{data['bench_A_proxy_20q']['jax_qsim_20q_proxy']['std_s']*1000:.1f} ms", "10"],
]

col_widths = [0.25, 0.10, 0.10, 0.18, 0.18, 0.10]
row_height = 0.085
x0 = 0.02

for row_i, row in enumerate(table_data):
    y = 1.0 - row_i * row_height - row_height
    bg = CARD_BG if row_i > 0 else "#1f2937"
    if row_i % 2 == 0 and row_i > 0:
        bg = "#0d1117"
    rect = FancyBboxPatch((x0 - 0.01, y - 0.005), 0.98, row_height - 0.005,
                          boxstyle="round,pad=0.005", facecolor=bg, edgecolor=GRID_CLR, lw=0.5,
                          transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    x = x0
    for col_i, (cell, cw) in enumerate(zip(row, col_widths)):
        color = TEXT_CLR if row_i > 0 else ACCENT1
        weight = 'bold' if row_i == 0 else 'normal'
        if row_i > 0 and col_i == 3:
            color = ACCENT3
        if row_i > 0 and row_i >= 8 and col_i == 0:
            color = ACCENT2
        ax.text(x, y + row_height * 0.35, cell,
                transform=ax.transAxes, fontsize=8.5,
                color=color, weight=weight, va='center')
        x += cw

ax.set_title("Real N=10 Benchmark Results  —  jax_qsim · JAX 0.10.1 · CPU Backend\n"
             "Timestamp: 2026-05-30 21:40:24  |  All times post-warmup (2 warmup runs discarded)",
             color=TEXT_CLR, fontsize=11, pad=15, loc='left')

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig_summary_table_n10.png")
fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved: " + out)

print("\nAll 4 figures generated successfully!")
