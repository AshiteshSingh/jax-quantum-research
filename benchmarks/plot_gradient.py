"""
Plot gradient comparison benchmark from N=10 JSON data.
Reproduces Figure A from the research paper.

Usage:
    python benchmarks/plot_gradient.py
    python benchmarks/plot_gradient.py --json path/to/benchmark.json
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "results", "n10_benchmark_20260530_212827.json")


def plot_gradient(json_path: str, out_path: str = None):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    grad = data["bench_D_gradient"]
    jaxgrad_runs = [r * 1000 for r in grad["jax_grad_reverse"]["runs_s"]]
    psr_runs     = [r * 1000 for r in grad["psr_emulation"]["runs_s"]]
    speedup      = grad["psr_emulation"]["speedup_vs_grad"]

    DARK_BG  = "#0d1117"
    CARD_BG  = "#161b22"
    ACCENT1  = "#58a6ff"
    ACCENT2  = "#f78166"
    GRID_CLR = "#30363d"
    TEXT_CLR = "#e6edf3"
    LABEL_CLR = "#8b949e"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=DARK_BG)
    fig.suptitle("Gradient Method Comparison — 15-Qubit VQC, 120 Params, N=10",
                 color=TEXT_CLR, fontsize=12)

    for ax in axes:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=LABEL_CLR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(True, color=GRID_CLR, linestyle='--', linewidth=0.5, alpha=0.7)

    # Left: scatter
    axes[0].scatter([1] * 10, jaxgrad_runs, color=ACCENT1, s=60, alpha=0.85, zorder=5,
                    label='jax.grad runs')
    axes[0].scatter([2] * 10, psr_runs, color=ACCENT2, s=60, alpha=0.85, zorder=5,
                    label='PSR runs')
    for vals, x in [(jaxgrad_runs, 1), (psr_runs, 2)]:
        axes[0].errorbar(x, np.mean(vals), yerr=np.std(vals),
                         fmt='D', color='white', ms=9, capsize=6, lw=2, zorder=6)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(['jax.grad\n(reverse AD)', 'PSR\n(120 params)'], color=TEXT_CLR)
    axes[0].set_ylabel("Gradient Step Time (ms)", fontsize=10)
    axes[0].set_title("Individual Run Times", color=TEXT_CLR, fontsize=10)
    axes[0].legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)

    # Right: bar
    means = [np.mean(jaxgrad_runs), np.mean(psr_runs)]
    stds  = [np.std(jaxgrad_runs), np.std(psr_runs)]
    labels = ['jax.grad\n(reverse AD)', 'PSR\n(emulated)']
    bars = axes[1].bar(labels, means, color=[ACCENT1, ACCENT2], alpha=0.85,
                       edgecolor='white', linewidth=0.7)
    axes[1].errorbar(labels, means, yerr=stds, fmt='none', color='white', capsize=8, lw=2)
    axes[1].annotate(f"PSR is {speedup:.1f}x slower",
                     xy=(1, means[1]), xytext=(0.5, means[1] * 0.7),
                     color=ACCENT2, fontsize=10, ha='center',
                     arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=1.2))
    axes[1].set_title(f"Mean +/- 1sigma (N=10)  |  Speedup={speedup:.1f}x",
                      color=TEXT_CLR, fontsize=10)

    if out_path is None:
        out_path = os.path.join(SCRIPT_DIR, "results", "plot_gradient.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot gradient comparison")
    parser.add_argument("--json", default=DEFAULT_JSON)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    plot_gradient(args.json, args.out)
