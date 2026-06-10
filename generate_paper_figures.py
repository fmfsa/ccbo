"""
Generate all paper figures from experiment results.

Run after run_experiments.py for all three conditions:
    python run_experiments.py --benchmark CompleteGraph --condition main
    python run_experiments.py --benchmark CompleteGraph --condition wrong_edge
    python run_experiments.py --benchmark CompleteGraph --condition rccbo_teaser

Then:
    python generate_paper_figures.py
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from ccbo.visualize import plot_dag, plot_convergence_and_final, plot_wrong_edge_comparison
from ccbo.rccbo.visualize import plot_rccbo_with_partition_evolution

RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join('paper', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

BENCHMARK = 'CompleteGraph'


def _load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Fig 1: DAG
# ---------------------------------------------------------------------------

def generate_dag_figure():
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_dag(BENCHMARK, ax=ax)
    out = os.path.join(FIGURES_DIR, f'{BENCHMARK}_dag.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 2: Main convergence (from 'main' condition)
# ---------------------------------------------------------------------------

def _method_sort_key(method):
    """Order methods as BO, CBO, CCBO-* (sorted), RCCBO, then anything else."""
    if method == 'BO':
        return (0, method)
    if method == 'CBO':
        return (1, method)
    if method.startswith('CCBO'):
        return (2, method)
    if method == 'RCCBO':
        return (3, method)
    return (4, method)


def generate_comparison_figure(seeds=5):
    """Benchmark-agnostic convergence comparison from the 'main' pkl.

    Reads results/{BENCHMARK}_main_{seeds}seeds.pkl (seed -> method ->
    {'global_opt', ...}) and plots one line per method (BO, CBO, each CCBO
    coarsening, RCCBO) showing the mean +/- standard error across seeds.
    """
    data = _load(f'{BENCHMARK}_main_{seeds}seeds.pkl')
    if data is None:
        return

    # Union of method names across all seeds, in a stable display order.
    methods = sorted({m for s in data for m in data[s]}, key=_method_sort_key)

    # Distinct colour per method; BO drawn dashed to set the baseline apart.
    palette = plt.get_cmap('tab10').colors

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, method in enumerate(methods):
        curves = [data[s][method]['global_opt']
                  for s in data if method in data[s]]
        if not curves:
            continue
        # Pad to common length by repeating the last value.
        max_len = max(len(c) for c in curves)
        arr = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        mean = arr.mean(0)
        sem = arr.std(0) / np.sqrt(arr.shape[0])
        xs = np.arange(len(mean))
        ls = '--' if method == 'BO' else '-'
        color = palette[i % len(palette)]
        ax.plot(xs, mean, color=color, lw=2.2, ls=ls, label=method, zorder=5)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.set_xlabel('Optimization step')
    ax.set_ylabel(r'Best $Y$ found ($\downarrow$)')
    ax.set_title(f'{BENCHMARK}: BO vs CBO vs CCBO vs RCCBO')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, f'{BENCHMARK}_comparison.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 3: Wrong-edge robustness (from 'wrong_edge' condition)
# ---------------------------------------------------------------------------

def generate_wrong_edge_figure(seeds=5):
    data = _load(f'{BENCHMARK}_wrong_edge_{seeds}seeds.pkl')
    if data is None:
        return

    if BENCHMARK == 'LightTunnel':
        _generate_lt_wrong_edge_figure(data)
        return

    out = os.path.join(FIGURES_DIR, f'{BENCHMARK}_wrong_edge.pdf')
    plot_wrong_edge_comparison(data, output_path=out)
    print(f"Saved {out}")


def _generate_lt_wrong_edge_figure(results):
    """Generic wrong-edge figure for LightTunnel results.

    Groups results by (partition_label, misspec) and produces:
      - lt_wrong_edge_convergence.pdf: 4 curves, mean +/- SE across seeds
      - lt_wrong_edge_bar.pdf: final-Y bar chart for the 4 conditions
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        grouped[(r['partition_label'], r['misspec'])].append(r['global_opt'])

    # Order: fine/correct, fine/misspec, coarse/correct, coarse/misspec
    keys = sorted(grouped.keys(),
                  key=lambda k: ('coarse' not in k[0].lower() and 'rgb' not in k[0].lower(),
                                 k[1] == 'correct',
                                 k))
    # Above gives a deterministic order; we want a stable one:
    # ('finest','correct'), ('finest','WrongBG'),
    # ('{RGB},{P1P2}','correct'), ('{RGB},{P1P2}','WrongBG')
    canonical_order = [
        ('finest', 'correct'),
        ('finest', 'WrongBG'),
        ('{RGB},{P1P2}', 'correct'),
        ('{RGB},{P1P2}', 'WrongBG'),
    ]
    keys = [k for k in canonical_order if k in grouped]

    colors = {'finest': '#d62728', '{RGB},{P1P2}': '#1f77b4'}
    linestyles = {'correct': '-', 'WrongBG': '--'}

    # Convergence figure
    fig, ax = plt.subplots(figsize=(6, 4))
    for part, mis in keys:
        curves = grouped[(part, mis)]
        max_len = max(len(c) for c in curves)
        arr = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        mean = arr.mean(0)
        sem = arr.std(0) / np.sqrt(arr.shape[0])
        xs = np.arange(len(mean))
        label = f"{part} — {'correct DAG' if mis == 'correct' else 'WrongBG DAG'}"
        ax.plot(xs, mean, color=colors[part], ls=linestyles[mis], lw=2,
                label=label, zorder=5)
        ax.fill_between(xs, mean - sem, mean + sem, color=colors[part],
                        alpha=0.15)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel(r"Best $Y$ found  ($\downarrow$ if min, $\uparrow$ if max)")
    ax.set_title("LightTunnel wrong-edge robustness")
    ax.legend(fontsize=8.5, loc='best', framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_conv = os.path.join(FIGURES_DIR, 'lt_wrong_edge_convergence.pdf')
    fig.savefig(out_conv, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_conv}")

    # Bar chart of final Y
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    labels = []
    means, sems, bar_colors = [], [], []
    for part, mis in keys:
        curves = grouped[(part, mis)]
        finals = np.array([c[-1] for c in curves])
        means.append(finals.mean())
        sems.append(finals.std() / np.sqrt(len(finals)))
        labels.append(f"{part}\n{mis}")
        bar_colors.append(colors[part])
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=sems, color=bar_colors, alpha=0.85,
           capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Final best $Y$")
    ax.set_title("LightTunnel: misspecification gap")
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    out_bar = os.path.join(FIGURES_DIR, 'lt_wrong_edge_bar.pdf')
    fig.savefig(out_bar, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_bar}")


# ---------------------------------------------------------------------------
# Fig 4: RCCBO teaser (from 'rccbo_teaser' condition)
# ---------------------------------------------------------------------------

def generate_rccbo_figure(seeds=5):
    data = _load(f'{BENCHMARK}_rccbo_teaser_{seeds}seeds.pkl')
    if data is None:
        return

    # Pick the representative seed: the one whose RCCBO final Y is closest to median
    finals = [(s, data[s]['RCCBO']['global_opt'][-1]) for s in data if 'RCCBO' in data[s]]
    finals.sort(key=lambda x: x[1])
    median_seed = finals[len(finals) // 2][0]

    seed_results = {m: data[median_seed][m] for m in data[median_seed]}

    out = os.path.join(FIGURES_DIR, f'{BENCHMARK}_rccbo_convergence.pdf')
    plot_rccbo_with_partition_evolution(seed_results, rccbo_key='RCCBO',
                                        title=f'RCCBO convergence ({BENCHMARK}, '
                                              f'representative seed {median_seed})',
                                        save_path=out)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Multi-seed summary table (stdout)
# ---------------------------------------------------------------------------

def print_multi_seed_summary(seeds=5):
    data = _load(f'{BENCHMARK}_main_{seeds}seeds.pkl')
    if data is None:
        return

    print(f"\n{'='*70}\nMULTI-SEED SUMMARY — {BENCHMARK} ({seeds} seeds)\n{'='*70}")
    all_methods = sorted({m for s in data for m in data[s]})
    for method in all_methods:
        finals = [data[s][method]['global_opt'][-1]
                  for s in data if method in data[s]]
        if finals:
            mean, std = np.mean(finals), np.std(finals)
            print(f"  {method:35s}: ${mean:.2f} \\pm {std:.2f}$  (n={len(finals)})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='CompleteGraph',
                        choices=['CompleteGraph', 'SimplifiedCoralGraph',
                                 'LightTunnel'])
    parser.add_argument('--seeds', default=5, type=int)
    parser.add_argument('--only', default=None,
                        choices=[None, 'dag', 'comparison', 'wrong_edge', 'rccbo'],
                        help='Generate only this figure (default: all four)')
    args = parser.parse_args()
    BENCHMARK = args.benchmark

    if args.only in (None, 'dag'):
        try:
            generate_dag_figure()
        except (KeyError, ValueError, NotImplementedError) as e:
            print(f"  Skipping DAG figure for {BENCHMARK}: {e}")
    if args.only in (None, 'comparison'):
        generate_comparison_figure(seeds=args.seeds)
    if args.only in (None, 'wrong_edge'):
        generate_wrong_edge_figure(seeds=args.seeds)
    if args.only in (None, 'rccbo'):
        generate_rccbo_figure(seeds=args.seeds)
    print_multi_seed_summary(seeds=args.seeds)
    print(f"\nFigures saved to {FIGURES_DIR}")
