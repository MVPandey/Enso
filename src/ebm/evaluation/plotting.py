"""Plotting utilities for inference method comparison experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# -- Style constants ----------------------------------------------------------

COLORS = {
    'langevin': '#2171b5',
    'svgd': '#e6550d',
}
METHOD_LABELS = {
    'langevin': 'Langevin',
    'svgd': 'SVGD',
}
_FIGSIZE = (7, 4.5)


def _apply_style() -> None:
    """Apply consistent plot styling."""
    plt.rcParams.update(
        {
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.dpi': 150,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        }
    )


def plot_accuracy_vs_chains(
    results: list[dict],
    output_path: Path,
    metric: str = 'puzzle_accuracy',
) -> None:
    """
    Plot accuracy as a function of chain count for each method.

    Args:
        results: List of run dicts, each with keys 'method', 'n_chains',
            'puzzle_accuracy', 'cell_accuracy', 'constraint_satisfaction'.
        output_path: Where to save the figure.
        metric: Which accuracy metric to plot on the y-axis.

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    for method in ('langevin', 'svgd'):
        runs = sorted(
            [r for r in results if r['method'] == method],
            key=lambda r: r['n_chains'],
        )
        if not runs:
            continue
        chains = [r['n_chains'] for r in runs]
        values = [r[metric] * 100 for r in runs]
        ax.plot(
            chains,
            values,
            marker='o',
            linewidth=2,
            markersize=7,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        for x, y in zip(chains, values):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)

    ax.set_xlabel('Number of chains / particles')
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)')
    ax.set_title('Inference Method Comparison: Accuracy vs. Chain Count')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)


def plot_timing(results: list[dict], output_path: Path) -> None:
    """
    Plot wall-clock time per puzzle for each method and chain count.

    Args:
        results: List of run dicts with 'method', 'n_chains', 'time_s', 'n_puzzles'.
        output_path: Where to save the figure.

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    for method in ('langevin', 'svgd'):
        runs = sorted(
            [r for r in results if r['method'] == method],
            key=lambda r: r['n_chains'],
        )
        if not runs:
            continue
        chains = [r['n_chains'] for r in runs]
        ms_per_puzzle = [r['time_s'] / r['n_puzzles'] * 1000 for r in runs]
        ax.plot(
            chains,
            ms_per_puzzle,
            marker='s',
            linewidth=2,
            markersize=7,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel('Number of chains / particles')
    ax.set_ylabel('Time per puzzle (ms)')
    ax.set_title('Inference Throughput')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)


def plot_constraint_satisfaction(results: list[dict], output_path: Path) -> None:
    """
    Plot constraint satisfaction as grouped bars per chain count.

    Args:
        results: List of run dicts with 'method', 'n_chains', 'constraint_satisfaction'.
        output_path: Where to save the figure.

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    methods = ('langevin', 'svgd')
    all_chains = sorted({r['n_chains'] for r in results})
    x = np.arange(len(all_chains))
    width = 0.35

    for i, method in enumerate(methods):
        runs_by_chain = {r['n_chains']: r for r in results if r['method'] == method}
        values = [runs_by_chain.get(c, {}).get('constraint_satisfaction', 0) * 100 for c in all_chains]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=METHOD_LABELS[method], color=COLORS[method])
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{v:.1f}', ha='center', fontsize=8)

    ax.set_xlabel('Number of chains / particles')
    ax.set_ylabel('Constraint satisfaction (%)')
    ax.set_title('Sudoku Constraint Satisfaction')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in all_chains])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(output_path)
    plt.close(fig)


def plot_bandwidth_sweep(results: list[dict], output_path: Path) -> None:
    """
    Plot SVGD accuracy across bandwidth values (fixed chain count).

    Args:
        results: List of SVGD-only run dicts with 'kernel_bandwidth',
            'puzzle_accuracy', and 'n_chains'.
        output_path: Where to save the figure.

    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    chain_counts = sorted({r['n_chains'] for r in results})
    for nc in chain_counts:
        runs = sorted(
            [r for r in results if r['n_chains'] == nc],
            key=lambda r: r['kernel_bandwidth'],
        )
        if not runs:
            continue
        bw = [r['kernel_bandwidth'] for r in runs]
        acc = [r['puzzle_accuracy'] * 100 for r in runs]
        ax.plot(bw, acc, marker='o', linewidth=2, markersize=7, label=f'{nc} particles')

    ax.set_xlabel('Kernel bandwidth')
    ax.set_ylabel('Puzzle accuracy (%)')
    ax.set_title('SVGD: Bandwidth Sensitivity')
    ax.set_xscale('log')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
