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

from ccbo.visualize import plot_dag
from ccbo.metrics import gap, pa_gap, reference_optimum
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
    """Order methods as BO, CBO, QCBO-* (sorted), then anything else."""
    if method == 'BO':
        return (0, method)
    if method == 'CBO':
        return (1, method)
    if method.startswith('QCBO'):
        return (2, method)
    return (3, method)


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
        sem = arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
        xs = np.arange(len(mean))
        ls = '--' if method == 'BO' else '-'
        color = palette[i % len(palette)]
        ax.plot(xs, mean, color=color, lw=2.2, ls=ls, label=method, zorder=5)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.set_xlabel('Optimization step')
    ax.set_ylabel(r'Best $Y$ found ($\downarrow$)')
    ax.set_title(f'{BENCHMARK}: BO vs CBO vs QCBO')
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
    """Benchmark-agnostic wrong-edge robustness figure.

    The ``wrong_edge`` pkl is a list of per-(partition, misspec, seed) result
    dicts. We draw one best-so-far curve per (partition, misspec): colour by
    partition, solid for the correct DAG, dashed for each misspecification.
    The companion numbers live in the LaTeX table (``emit_wrong_edge_table``).
    """
    data = _load(f'{BENCHMARK}_wrong_edge_{seeds}seeds.pkl')
    if data is None:
        return

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in data:
        grouped[(r['partition_label'], r['misspec'])].append(r['global_opt'])

    partitions = sorted({k[0] for k in grouped},
                        key=lambda p: (p != 'finest', p))   # finest first
    misspecs = sorted({k[1] for k in grouped},
                      key=lambda m: (m != 'correct', m))    # correct first
    palette = plt.get_cmap('tab10').colors
    part_color = {p: palette[i % len(palette)] for i, p in enumerate(partitions)}
    mis_ls = {m: ('-' if m == 'correct' else ls)
              for m, ls in zip(misspecs, ['-', '--', ':', '-.'])}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for part in partitions:
        for mis in misspecs:
            curves = grouped.get((part, mis))
            if not curves:
                continue
            max_len = max(len(c) for c in curves)
            arr = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
            mean = arr.mean(0)
            sem = (arr.std(0, ddof=1) / np.sqrt(arr.shape[0])
                   if arr.shape[0] > 1 else np.zeros_like(mean))
            xs = np.arange(len(mean))
            label = f"{part} — {mis}"
            ax.plot(xs, mean, color=part_color[part], ls=mis_ls[mis], lw=2,
                    label=label, zorder=5)
            ax.fill_between(xs, mean - sem, mean + sem,
                            color=part_color[part], alpha=0.15)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel(r"Best $Y$ found ($\downarrow$)")
    ax.set_title(f"{BENCHMARK}: robustness under misspecification")
    ax.legend(fontsize=8.5, loc='best', framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f'{BENCHMARK}_wrong_edge.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig: Coarsening sweep (from 'sweep' condition) — the V(pi) staircase
# ---------------------------------------------------------------------------

def generate_sweep_figure(seeds=10):
    """Final Y vs partition fineness across the valid coarsening lattice."""
    data = _load(f'{BENCHMARK}_sweep_{seeds}seeds.pkl')
    if data is None:
        return

    # label -> (num_parts, finals across seeds)
    entries = {}
    for s in data:
        for label, r in data[s].items():
            entries.setdefault(label, (r['num_parts'], []))[1].append(
                r['global_opt'][-1])

    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.RandomState(0)
    for label, (nparts, finals) in sorted(entries.items(),
                                          key=lambda kv: -kv[1][0]):
        finals = np.array(finals)
        jitter = rng.uniform(-0.08, 0.08)
        sem = finals.std(ddof=1) / np.sqrt(len(finals)) if len(finals) > 1 else 0.0
        ax.errorbar(nparts + jitter, finals.mean(),
                    yerr=sem,
                    fmt='o', capsize=3, ms=6)
        ax.annotate(label.replace('QCBO-', ''), (nparts + jitter, finals.mean()),
                    textcoords='offset points', xytext=(6, 4), fontsize=7)
    ax.set_xlabel('Partition fineness (number of clusters incl. $\\{Y\\}$)')
    ax.set_ylabel(r'Final best $Y$ ($\downarrow$)')
    ax.set_title(f'{BENCHMARK}: price of coarsening (Prop. 4)')
    ax.invert_xaxis()  # coarse -> fine left to right? keep finest on left
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f'{BENCHMARK}_sweep.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# LaTeX tables (stdout + paper/tables/*.tex)
# ---------------------------------------------------------------------------

TABLES_DIR = os.path.join('paper', 'tables')


def _sem(arr):
    arr = np.asarray(arr, dtype=float)
    return arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0


def emit_main_table(seeds=10):
    """LaTeX rows for the main-results table.

    Columns: Method & Final Y & GAP & PA-GAP (each mean ± s.e. over seeds).
    GAP / PA-GAP follow the survey's standardized definitions (ccbo.metrics)
    and are normalized against the reference optimum y* (known oracle value
    for CompleteGraph/ConfoundedCluster, else the best value observed).
    """
    data = _load(f'{BENCHMARK}_main_{seeds}seeds.pkl')
    if data is None:
        return
    os.makedirs(TABLES_DIR, exist_ok=True)
    methods = sorted({m for s in data for m in data[s]}, key=_method_sort_key)

    all_trajs = [data[s][m]['global_opt'] for s in data for m in data[s]]
    y_star = reference_optimum(BENCHMARK, all_trajs, task='min')
    print(f"[{BENCHMARK}] reference optimum y* = {y_star:.3f}")

    lines = []
    for m in methods:
        trajs = [data[s][m]['global_opt'] for s in data if m in data[s]]
        finals = np.array([t[-1] for t in trajs])
        gaps = np.array([gap(t, y_star, 'min') for t in trajs])
        pags = np.array([pa_gap(t, y_star, 'min') for t in trajs])
        label = m.replace('{', '\\{').replace('}', '\\}')
        lines.append(
            f"{label} & ${finals.mean():.3f} \\pm {_sem(finals):.3f}$ & "
            f"${gaps.mean():.3f} \\pm {_sem(gaps):.3f}$ & "
            f"${pags.mean():.3f} \\pm {_sem(pags):.3f}$ \\\\")
    out = os.path.join(TABLES_DIR, f'{BENCHMARK}_main.tex')
    with open(out, 'w') as f:
        # \bottomrule lives inside the included file: \input followed by
        # \bottomrule in the outer tabular triggers "Misplaced \noalign".
        f.write('\n'.join(lines) + '\n\\bottomrule\n')
    print(f"Saved {out}")
    print('\n'.join(lines))


def emit_wrong_edge_table(seeds=10):
    """LaTeX rows for the robustness table, with paired per-seed gaps."""
    data = _load(f'{BENCHMARK}_wrong_edge_{seeds}seeds.pkl')
    if data is None:
        return
    os.makedirs(TABLES_DIR, exist_ok=True)
    from collections import defaultdict
    grouped = defaultdict(dict)   # (partition, misspec) -> seed -> final
    num_es = {}
    for r in data:
        grouped[(r['partition_label'], r['misspec'])][r['seed']] = \
            r['global_opt'][-1]
        num_es[(r['partition_label'], r['misspec'])] = r['num_es']
    lines = []
    parts = sorted({k[0] for k in grouped})
    for part in parts:
        base = grouped.get((part, 'correct'), {})
        for (p, mis), per_seed in sorted(grouped.items()):
            if p != part:
                continue
            finals = np.array([per_seed[s] for s in sorted(per_seed)])
            sem = _sem(finals)
            if mis == 'correct' or not base:
                gap_str = '---'
            else:
                seeds_common = sorted(set(per_seed) & set(base))
                diffs = np.array([per_seed[s] - base[s] for s in seeds_common])
                gap_str = f"${diffs.mean():+.3f} \\pm {_sem(diffs):.3f}$"
            label_p = p.replace('{', '\\{').replace('}', '\\}')
            lines.append(
                f"{label_p} & {mis} & {num_es[(p, mis)]} & "
                f"${finals.mean():.3f} \\pm {sem:.3f}$ & {gap_str} \\\\")
    out = os.path.join(TABLES_DIR, f'{BENCHMARK}_wrong_edge.tex')
    with open(out, 'w') as f:
        # \bottomrule lives inside the included file (see emit_main_table).
        f.write('\n'.join(lines) + '\n\\bottomrule\n')
    print(f"Saved {out}")
    print('\n'.join(lines))


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
            mean = np.mean(finals)
            std = np.std(finals, ddof=1) if len(finals) > 1 else 0.0
            print(f"  {method:35s}: ${mean:.2f} \\pm {std:.2f}$  (n={len(finals)})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='CompleteGraph',
                        choices=['CompleteGraph', 'SimplifiedCoralGraph',
                                 'ConfoundedCluster', 'Tier1Graph'])
    parser.add_argument('--seeds', default=10, type=int)
    parser.add_argument('--only', default=None,
                        choices=[None, 'dag', 'comparison', 'wrong_edge',
                                 'sweep', 'tables'],
                        help='Generate only this output (default: all)')
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
        emit_wrong_edge_table(seeds=args.seeds)
    if args.only in (None, 'sweep'):
        generate_sweep_figure(seeds=args.seeds)
    if args.only == 'tables':
        emit_main_table(seeds=args.seeds)
        emit_wrong_edge_table(seeds=args.seeds)
    if args.only is None:
        emit_main_table(seeds=args.seeds)
    print_multi_seed_summary(seeds=args.seeds)
    print(f"\nFigures saved to {FIGURES_DIR}")
