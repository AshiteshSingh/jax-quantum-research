"""
Plot scaling benchmark from N=10 JSON data.
Reproduces Figure B from the research paper.

Usage:
    python benchmarks/plot_scaling.py
    python benchmarks/plot_scaling.py --json path/to/benchmark.json
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "results", "n10_benchmark_20260530_214024.json")


def plot_scaling(json_path: str, out_path: str = None):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    scaling = data["bench_C_scaling"]
    qubits = sorted([int(k) for k in scaling.keys()])
    means  = [scaling[str(q)]["mean_s"] * 1000 for q in qubits]
    stds   = [scaling[str(q)]["std_s"]  * 1000 for q in qubits]

    DARK_BG  = "#0d1117"
    CARD_BG  = "#161b22"
    ACCENT1  = "#58a6ff"
    ACCENT2  = "#f78166"
    GRID_CLR = "#30363d"
    TEXT_CLR = "#e6edf3"
    LABEL_CLR = "#8b949e"

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=LABEL_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.7)

    ax.semilogy(qubits, means, 'o-', color=ACCENT1, lw=2, ms=7,
                markerfacecolor=ACCENT1, markeredgecolor='white', markeredgewidth=0.8,
                label='jax_qsim (N=10 runs, post-JIT)')
    ax.fill_between(qubits,
                    [max(1e-4, m - s) for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color=ACCENT1)

    qs = np.linspace(min(qubits), max(qubits), 200)
    base = means[0] / (2 ** qubits[0])
    ax.semilogy(qs, base * (2 ** qs), '--', color=ACCENT2, lw=1.2, alpha=0.6,
                label=r'$O(2^n)$ reference')

    ax.set_xlabel("Qubits (n)", fontsize=11)
    ax.set_ylabel("Execution Time (ms, log scale)", fontsize=11)
    ax.set_title("jax_qsim Scaling — 3-layer HEA · N=10 · CPU Baseline", fontsize=11)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
    ax.set_xticks(qubits)

    if out_path is None:
        out_path = os.path.join(SCRIPT_DIR, "results", "plot_scaling.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling benchmark")
    parser.add_argument("--json", default=DEFAULT_JSON)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    plot_scaling(args.json, args.out)
