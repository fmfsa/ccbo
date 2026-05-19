"""
Visualization for causal DAGs, coarsenings, and experiment results.

Plot style: seaborn colorblind palette, serif fonts, publication quality.
"""

import os
import pickle
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns
import networkx as nx

from .coarsening import (enumerate_valid_coarsenings_manip,
                          get_dag_edges_from_sem,
                          build_coarsened_admg, compute_POMIS,
                          project_out_hidden)

# ---- Global style ----
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.4)
plt.rcParams.update({
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'text.usetex': False,
})

# Colour constants — one per manipulative grouping
C_BO       = '#888888'   # grey
C_CBO      = '#0072B2'   # blue
C_SEP      = '#009E73'   # teal/green  — {B},{D},{E} all separate
C_BD_E     = '#56B4E9'   # sky blue    — {B,D},{E}
C_B_DE     = '#D55E00'   # vermilion   — {B},{D,E}
C_BE_D     = '#CC79A7'   # pink        — {B,E},{D}
C_BDE      = '#E69F00'   # orange      — {B,D,E} all merged

NODE_COLOR = '#8fbcd4'
EDGE_COLOR = '#1f4b73'
MANIP_COLOR = '#e07b54'
TARGET_COLOR = '#d4e88f'


# ===================================================================
# Classify results into the 5 manipulative groupings
# ===================================================================

def _get_manip_grouping(partition_str, manip_vars=None):
    """
    Parse a partition string and return the manipulative grouping as a
    frozenset of frozensets.

    E.g. "{A,B,C,D} | {E} | {F} | {Y}" with manip={B,D,E}
    → frozenset({frozenset({'B','D'}), frozenset({'E'})})
    """
    if manip_vars is None:
        manip_vars = {'B', 'D', 'E'}

    parts = partition_str.split(' | ')
    grouping = []
    for part in parts:
        part_vars = set(part.strip('{}').split(','))
        manip_in_part = frozenset(part_vars & manip_vars)
        if manip_in_part:
            grouping.append(manip_in_part)
    return frozenset(grouping)


# Canonical labels for each grouping (in order of expected performance)
GROUPING_META = {
    frozenset({frozenset({'B'}), frozenset({'D'}), frozenset({'E'})}):
        {'label': '{B},{D},{E}', 'short': 'all sep.', 'color': C_SEP, 'ls': '-', 'order': 0},
    frozenset({frozenset({'B', 'D'}), frozenset({'E'})}):
        {'label': '{B,D},{E}', 'short': 'B,D merged', 'color': C_BD_E, 'ls': '-', 'order': 1},
    frozenset({frozenset({'B'}), frozenset({'D', 'E'})}):
        {'label': '{B},{D,E}', 'short': 'D,E merged', 'color': C_B_DE, 'ls': '-', 'order': 2},
    frozenset({frozenset({'B', 'E'}), frozenset({'D'})}):
        {'label': '{B,E},{D}', 'short': 'B,E merged', 'color': C_BE_D, 'ls': '-', 'order': 3},
    frozenset({frozenset({'B', 'D', 'E'})}):
        {'label': '{B,D,E}', 'short': 'all merged', 'color': C_BDE, 'ls': '-', 'order': 4},
}


def _get_grouping_counts(graph_name, manip_vars=None):
    """
    Compute the total number of valid coarsenings for each manipulative grouping.

    Returns dict mapping grouping frozenset -> count (e.g. 53, 8, 29, 4, 4).
    """
    if manip_vars is None:
        manip_vars = {'B', 'D', 'E'}

    coarsenings = enumerate_valid_coarsenings_manip(graph_name)

    counts = {}
    for c in coarsenings:
        grouping = []
        for part in c['partition']:
            manip_in_part = frozenset(set(manip_vars) & part)
            if manip_in_part:
                grouping.append(manip_in_part)
        grouping = frozenset(grouping)
        counts[grouping] = counts.get(grouping, 0) + 1

    return counts


def _classify_results(results):
    """
    Classify results into method categories.

    Returns dict with keys:
        'BO': list of BO results
        'CBO': list of CBO results
        'groupings': OrderedDict mapping grouping frozenset -> list of results
    """
    from collections import OrderedDict

    bo = [r for r in results if r['method'] == 'BO']
    cbo = [r for r in results if r['method'].startswith('CBO-')]
    ccbo = [r for r in results if r['method'].startswith('CCBO')]

    groupings = {}
    for r in ccbo:
        partition_str = r.get('partition', '')
        g = _get_manip_grouping(partition_str)
        groupings.setdefault(g, []).append(r)

    # Sort by order
    sorted_groupings = OrderedDict()
    for g in sorted(groupings.keys(),
                    key=lambda g: GROUPING_META.get(g, {}).get('order', 99)):
        sorted_groupings[g] = groupings[g]

    return {'BO': bo, 'CBO': cbo, 'groupings': sorted_groupings}


# ===================================================================
# DAG visualisation
# ===================================================================

def _dag_layout(graph_name):
    """Hand-crafted positions for known graphs."""
    if graph_name == 'CompleteGraph':
        return {
            'A': (-1, 1), 'B': (1, 2),
            'C': (1, 1), 'D': (0, 0), 'E': (-1, 0), 'Y': (0, -1),
        }
    elif graph_name == 'ToyGraph':
        return {'X': (-1, 0), 'Z': (0, 0), 'Y': (1, 0)}
    return None


def plot_dag(graph_name, ax=None, title=None):
    """Draw the causal DAG with manipulative nodes highlighted."""
    dag_edges, nodes, hidden, manip_vars = get_dag_edges_from_sem(graph_name)

    G = nx.DiGraph()
    observable = sorted(set(nodes) - set(hidden or []))
    G.add_nodes_from(observable)
    for u, v in dag_edges:
        if u in observable and v in observable:
            G.add_edge(u, v)

    pos = _dag_layout(graph_name) or nx.spring_layout(G, seed=42)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    node_colors = []
    for n in G.nodes:
        if n == 'Y':
            node_colors.append(TARGET_COLOR)
        elif n in manip_vars:
            node_colors.append(MANIP_COLOR)
        else:
            node_colors.append(NODE_COLOR)

    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=node_colors, edgecolors=EDGE_COLOR,
        linewidths=1.5, font_size=14, font_weight='bold',
        node_size=800, arrows=True,
        arrowstyle='->', arrowsize=15,
        connectionstyle='arc3,rad=0.1',
    )
    ax.set_title(title or graph_name, fontsize=16, fontweight='bold')
    ax.axis('off')
    return ax


# ===================================================================
# Coarsening lattice (Hasse diagram)
# ===================================================================

def plot_coarsening_lattice(graph_name, output_path=None, max_show=30):
    """
    Visualise the lattice of valid coarsenings as a Hasse-style diagram.

    Nodes are coarsenings; y-position = number of partition elements.
    Colour by manipulative grouping.
    """
    _, nodes, hidden, manip_vars = get_dag_edges_from_sem(graph_name)
    coarsenings = enumerate_valid_coarsenings_manip(graph_name)

    if len(coarsenings) > max_show:
        coarsenings = coarsenings[:max_show]

    fig, ax = plt.subplots(figsize=(14, 8))

    by_level = {}
    for i, c in enumerate(coarsenings):
        level = c['num_parts']
        by_level.setdefault(level, []).append(i)

    pos = {}
    for level, indices in sorted(by_level.items()):
        n = len(indices)
        for j, idx in enumerate(indices):
            x = (j - (n - 1) / 2) * 1.5
            pos[idx] = (x, level)

    manip_set = set(manip_vars)
    for i, c in enumerate(coarsenings):
        manip_merged = any(
            len(manip_set & p) > 1 for p in c['partition'])
        is_identity = all(len(p) == 1 for p in c['partition'])

        if is_identity:
            color = C_CBO
        elif manip_merged:
            color = C_B_DE  # vermilion for merged
        else:
            color = C_SEP

        x, y = pos[i]
        label = '\n'.join(
            '{' + ','.join(sorted(p)) + '}'
            for p in c['partition'] if len(p) > 1
        )
        if is_identity:
            label = 'identity'

        ax.scatter(x, y, s=300, c=color, edgecolors='black',
                   linewidth=0.8, zorder=5)
        ax.annotate(label, (x, y), fontsize=5, ha='center', va='bottom',
                    xytext=(0, 12), textcoords='offset points')

    handles = [
        mpatches.Patch(color=C_CBO, label='Identity (= CBO)'),
        mpatches.Patch(color=C_SEP, label='Non-manip. merges only'),
        mpatches.Patch(color=C_B_DE, label='Merges manip. variables'),
    ]
    ax.legend(handles=handles, fontsize=10, loc='upper right')

    ax.set_ylabel('Number of partition elements', fontsize=13)
    ax.set_title(f'{graph_name}: Coarsening Lattice ({len(coarsenings)} coarsenings)',
                 fontsize=15, fontweight='bold')
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {output_path}")
    plt.close(fig)


# ===================================================================
# Main results figure: convergence + grouped bar chart
# ===================================================================

def plot_convergence_and_final(results, graph_name, output_path=None):
    """
    Single-panel convergence figure for the paper.
    One line per method/grouping with mean ± std shading.
    """
    cats = _classify_results(results)
    grouping_counts = _get_grouping_counts(graph_name)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # BO
    for r in cats['BO']:
        ax.plot(r['global_opt'], color=C_BO, lw=2.5, ls='--',
                label='BO', zorder=10)

    # CBO
    for r in cats['CBO']:
        ax.plot(r['global_opt'], color=C_CBO, lw=2.5, ls='-',
                label='CBO (full DAG)', zorder=10)
        break  # just one

    # CCBO groupings — mean ± std across coarsenings of each type
    for g, group in cats['groupings'].items():
        meta = GROUPING_META.get(g, {'label': '?', 'color': '#999999', 'ls': '-', 'order': 99})
        n_total = grouping_counts.get(g, len(group))
        traces = np.array([r['global_opt'] for r in group])
        mean_trace = traces.mean(axis=0)
        std_trace = traces.std(axis=0)
        ax.plot(mean_trace, color=meta['color'], lw=2, ls=meta['ls'],
                label=f'CCBO {meta["label"]} ($n$={n_total})', zorder=5)
        ax.fill_between(range(len(mean_trace)),
                        mean_trace - std_trace, mean_trace + std_trace,
                        color=meta['color'], alpha=0.15, zorder=4)

    ax.set_xlabel('Optimization step')
    ax.set_ylabel(r'Best $Y$ found ($\downarrow$)')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9,
              handlelength=2.5)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {output_path}")
    plt.close(fig)


# ===================================================================
# Standalone convergence plot
# ===================================================================

def plot_convergence(results, graph_name, output_path=None):
    """Standalone convergence plot with one line per method/grouping."""
    cats = _classify_results(results)
    grouping_counts = _get_grouping_counts(graph_name)
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in cats['BO']:
        ax.plot(r['global_opt'], color=C_BO, lw=2.5, ls='--',
                label='BO', zorder=10)

    for r in cats['CBO']:
        ax.plot(r['global_opt'], color=C_CBO, lw=2.5, ls='-',
                label='CBO (full DAG)', zorder=10)
        break

    for g, group in cats['groupings'].items():
        meta = GROUPING_META.get(g, {'label': '?', 'color': '#999999', 'ls': '-'})
        n_total = grouping_counts.get(g, len(group))
        traces = np.array([r['global_opt'] for r in group])
        mean_trace = traces.mean(axis=0)
        std_trace = traces.std(axis=0)
        ax.plot(mean_trace, color=meta['color'], lw=2, ls=meta['ls'],
                label=f'CCBO {meta["label"]} ($n$={n_total})', zorder=5)
        ax.fill_between(range(len(mean_trace)),
                        mean_trace - std_trace, mean_trace + std_trace,
                        color=meta['color'], alpha=0.15, zorder=4)

    ax.set_xlabel('Optimization step')
    ax.set_ylabel(r'Best $Y$ found ($\downarrow$)')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_title(f'{graph_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {output_path}")
    plt.close(fig)


# ===================================================================
# Performance vs partition granularity
# ===================================================================

def plot_performance_vs_num_parts(results, graph_name, output_path=None):
    """
    Scatter: final best Y vs number of partition elements, colored by
    manipulative grouping.
    """
    cats = _classify_results(results)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for g, group in cats['groupings'].items():
        meta = GROUPING_META.get(g, {'label': '?', 'color': '#999999'})
        num_parts = []
        final_y = []
        for r in group:
            partition_str = r.get('partition', '')
            n_parts = len(partition_str.split(' | '))
            num_parts.append(n_parts)
            final_y.append(r['global_opt'][-1])

        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(num_parts))
        ax.scatter(np.array(num_parts) + jitter, final_y,
                   color=meta['color'], s=60, alpha=0.8,
                   label=meta['label'],
                   edgecolors='white', linewidths=0.5, zorder=5)

    # Reference lines
    for r in cats['BO']:
        ax.axhline(r['global_opt'][-1], color=C_BO, ls='--', lw=1.5,
                   label='BO', zorder=3)
    for r in cats['CBO']:
        ax.axhline(r['global_opt'][-1], color=C_CBO, ls='-', lw=1.5,
                   label='CBO', zorder=3)
        break

    ax.set_xlabel('Number of partition elements')
    ax.set_ylabel(r'Final best $Y$ ($\downarrow$)')
    ax.set_title(f'{graph_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {output_path}")
    plt.close(fig)


# ===================================================================
# Summary table
# ===================================================================

def print_summary_table(results, graph_name):
    """Print a summary table of results grouped by manipulative grouping."""
    cats = _classify_results(results)

    print(f"\n{'='*70}")
    print(f"Summary: {graph_name}")
    print(f"{'='*70}")

    for r in cats['BO']:
        print(f"  {'BO':<35} final_y={r['global_opt'][-1]:.4f}")
    for r in cats['CBO']:
        print(f"  {'CBO (full DAG)':<35} final_y={r['global_opt'][-1]:.4f}")

    for g, group in cats['groupings'].items():
        meta = GROUPING_META.get(g, {'label': '?'})
        finals = [r['global_opt'][-1] for r in group]
        rep = finals[0]
        print(f"  CCBO {meta['label']:<28} n={len(group):<3} "
              f"final_y={rep:.4f}  "
              f"range=[{min(finals):.4f}, {max(finals):.4f}]")

    print()


# ===================================================================
# Wrong-edge robustness figure
# ===================================================================

def plot_wrong_edge_comparison(results, output_path=None):
    """
    Convergence-only figure for the wrong-edge robustness experiment.
    Solid = correct DAG, dashed = wrong DAG.

    Expects results with methods:
      'CCBO-correct-identity', 'CCBO-wrong-identity',
      'CCBO-correct-BD_E', 'CCBO-wrong-BD_E'
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    method_meta = {
        'CCBO-correct-identity': {
            'label': r'$\{B\},\{D\},\{E\}$ — correct DAG',
            'color': C_SEP, 'ls': '-', 'lw': 2.5},
        'CCBO-wrong-identity': {
            'label': r'$\{B\},\{D\},\{E\}$ — wrong DAG',
            'color': C_SEP, 'ls': '--', 'lw': 2.5},
        'CCBO-correct-BD_E': {
            'label': r'$\{B,D\},\{E\}$ — correct DAG',
            'color': C_BD_E, 'ls': '-', 'lw': 2.5},
        'CCBO-wrong-BD_E': {
            'label': r'$\{B,D\},\{E\}$ — wrong DAG',
            'color': C_BD_E, 'ls': '--', 'lw': 2.5},
    }

    method_order = [
        'CCBO-correct-identity', 'CCBO-wrong-identity',
        'CCBO-correct-BD_E', 'CCBO-wrong-BD_E',
    ]

    results_by_method = {r['method']: r for r in results}

    for method_name in method_order:
        if method_name not in results_by_method:
            continue
        r = results_by_method[method_name]
        meta = method_meta[method_name]
        ax.plot(r['global_opt'], color=meta['color'], lw=meta['lw'],
                ls=meta['ls'], label=meta['label'], zorder=5)

    ax.set_xlabel('Optimization step')
    ax.set_ylabel(r'Best $Y$ found ($\downarrow$)')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9,
              handlelength=2.5)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {output_path}")
    plt.close(fig)


# ===================================================================
# New-format results: {seed: {method: result_dict}}
# ===================================================================

def plot_main_results(pkl_path, output_dir=None, benchmark=None):
    """
    Plot convergence curves and final-Y bar chart from
    results pkl produced by run_experiments.py (new format:
    ``{seed: {method: result_dict}}``).

    Parameters
    ----------
    pkl_path : str
        Path to the pkl file.
    output_dir : str, optional
        Directory to save figures. Defaults to same dir as pkl.
    benchmark : str, optional
        Name for the title / file prefix. Inferred from pkl name if None.
    """
    import collections

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pkl_path))
    os.makedirs(output_dir, exist_ok=True)

    if benchmark is None:
        benchmark = os.path.basename(pkl_path).replace('_main_5seeds.pkl', '')

    # Gather per-method traces across seeds
    # raw = {seed_int: {method_str: {'global_opt': [...], ...}}}
    methods_traces = collections.defaultdict(list)
    for seed_data in raw.values():
        for method, rdict in seed_data.items():
            go = rdict.get('global_opt', [])
            if go:
                methods_traces[method].append(go)

    # Colour map — stable regardless of order
    palette = sns.color_palette("colorblind", 10)
    method_colors = {
        'BO':    palette[7],    # grey
        'CBO':   palette[0],    # blue
    }
    ccbo_colors = [palette[1], palette[2], palette[3], palette[4], palette[5], palette[6]]
    ccbo_idx = 0

    # Identify CCBO/RCCBO methods
    ccbo_methods = sorted([m for m in methods_traces if m not in ('BO', 'CBO')])
    for m in ccbo_methods:
        if m not in method_colors:
            method_colors[m] = ccbo_colors[ccbo_idx % len(ccbo_colors)]
            ccbo_idx += 1

    method_ls = {}
    for m in methods_traces:
        if m == 'BO':
            method_ls[m] = '--'
        elif m.startswith('RCCBO'):
            method_ls[m] = '-.'
        else:
            method_ls[m] = '-'

    # ---- Convergence curves ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    method_order = ['BO', 'CBO'] + ccbo_methods
    for m in method_order:
        if m not in methods_traces:
            continue
        traces = np.array(methods_traces[m])
        mean_t = traces.mean(axis=0)
        std_t = traces.std(axis=0)
        color = method_colors.get(m, '#666666')
        ls = method_ls.get(m, '-')
        ax.plot(mean_t, color=color, lw=2.2, ls=ls, label=m, zorder=5)
        ax.fill_between(range(len(mean_t)),
                        mean_t - std_t, mean_t + std_t,
                        color=color, alpha=0.15, zorder=4)

    ax.set_xlabel('Optimization step')
    ax.set_ylabel(r'Best $Y$ found ($\downarrow$)')
    ax.set_title(f'{benchmark} — convergence (mean ± std, {len(raw)} seeds)')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    conv_path = os.path.join(output_dir, f'{benchmark}_convergence.pdf')
    fig.savefig(conv_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {conv_path}")

    # ---- Bar chart: final Y mean ± std ----
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x_pos = np.arange(len(method_order))
    bar_means, bar_stds, bar_colors, bar_labels = [], [], [], []
    for m in method_order:
        if m not in methods_traces:
            continue
        finals = [t[-1] for t in methods_traces[m]]
        bar_means.append(np.mean(finals))
        bar_stds.append(np.std(finals))
        bar_colors.append(method_colors.get(m, '#666666'))
        bar_labels.append(m)

    x = np.arange(len(bar_labels))
    ax.bar(x, bar_means, yerr=bar_stds, color=bar_colors,
           edgecolor='black', linewidth=0.8, capsize=5,
           error_kw={'elinewidth': 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(r'Final best $Y$ ($\downarrow$)')
    ax.set_title(f'{benchmark} — final performance')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f'{benchmark}_final_bar.pdf')
    fig.savefig(bar_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {bar_path}")

    # ---- Text summary table ----
    print(f"\n{'='*65}")
    print(f"  {benchmark}  |  {len(raw)} seeds × {len(list(methods_traces.values())[0][0])-1} trials")
    print(f"{'='*65}")
    print(f"  {'Method':<26}  {'@T/2':>8}  {'Final':>8}  (std)")
    print(f"  {'-'*55}")
    for m in method_order:
        if m not in methods_traces:
            continue
        traces = np.array(methods_traces[m])
        mid = len(traces[0]) // 2
        t_half = [t[mid] for t in methods_traces[m]]
        finals = [t[-1] for t in methods_traces[m]]
        print(f"  {m:<26}  {np.mean(t_half):>8.3f}  {np.mean(finals):>8.3f}"
              f"  ±{np.std(finals):.3f}")
    print()
    return methods_traces


# ===================================================================
# Driver
# ===================================================================

def generate_all_figures(results_path, output_dir=None):
    """Generate all figures from a results pickle."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    graph_name = results[0]['experiment'] if results else 'Unknown'
    if output_dir is None:
        output_dir = os.path.dirname(results_path)

    os.makedirs(output_dir, exist_ok=True)

    # 1. DAG
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_dag(graph_name, ax=ax)
    path = os.path.join(output_dir, f'{graph_name}_dag.pdf')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {path}")

    # 2. Coarsening lattice
    plot_coarsening_lattice(
        graph_name,
        os.path.join(output_dir, f'{graph_name}_lattice.pdf'))

    # 3. Two-panel convergence + final performance (main result figure)
    plot_convergence_and_final(
        results, graph_name,
        os.path.join(output_dir, f'{graph_name}_comparison.pdf'))

    # 4. Performance vs number of partition elements
    plot_performance_vs_num_parts(
        results, graph_name,
        os.path.join(output_dir, f'{graph_name}_performance_vs_parts.pdf'))

    # 5. Standalone convergence
    plot_convergence(
        results, graph_name,
        os.path.join(output_dir, f'{graph_name}_convergence_standalone.pdf'))

    # Print summary
    print_summary_table(results, graph_name)

    print("\nAll figures generated.")
