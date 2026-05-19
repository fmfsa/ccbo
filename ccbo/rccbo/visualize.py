"""
Visualization for RCCBO experiments.

Produces convergence plots with refinement markers and partition evolution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_rccbo_convergence(results_dict, title=None, save_path=None):
    """
    Plot convergence curves for multiple methods including RCCBO.

    Parameters
    ----------
    results_dict : dict
        {method_name: {'global_opt': list, 'refinement_trials': list (optional)}}
    title : str, optional
    save_path : str, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {
        'BO': '#888888',
        'CBO': '#2ca02c',
        'CCBO (identity)': '#1f77b4',
        'CCBO ({B,C,D},{E})': '#ff7f0e',
        'RCCBO': '#d62728',
    }

    for method_name, result in results_dict.items():
        global_opt = result['global_opt']
        color = colors.get(method_name, None)
        trials = np.arange(len(global_opt))

        ax.plot(trials, global_opt, label=method_name, color=color, linewidth=2)

        # Mark refinement points for RCCBO
        if 'refinement_trials' in result and result['refinement_trials']:
            for rt in result['refinement_trials']:
                if rt < len(global_opt):
                    ax.axvline(x=rt, color=color or 'red', linestyle='--',
                               alpha=0.5, linewidth=1)
                    ax.annotate(f'refine', (rt, global_opt[rt]),
                                textcoords="offset points", xytext=(5, 10),
                                fontsize=8, color=color or 'red', alpha=0.7)

    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('Best Y (lower is better)', fontsize=12)
    ax.set_title(title or 'RCCBO Convergence', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_rccbo_with_partition_evolution(results_dict, rccbo_key='RCCBO',
                                        title=None, save_path=None):
    """
    Paper Fig 4 — convergence (top) + RCCBO partition evolution (bottom),
    sharing the trial axis.

    Expects ``results_dict[rccbo_key]`` to carry ``partition_history`` and
    ``refinement_trials``. For multi-seed inputs, pass a single representative
    seed's result (e.g. the one closest to the median final Y).
    """
    from ccbo.rccbo.partition_ops import partition_description

    rccbo = results_dict.get(rccbo_key)
    total_trials = len(rccbo['global_opt']) if rccbo else max(
        len(r['global_opt']) for r in results_dict.values())

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]})

    colors = {
        'BO': '#888888', 'CBO': '#2ca02c',
        'CCBO (identity)': '#1f77b4', 'CCBO ({B,C,D},{E})': '#ff7f0e',
        'RCCBO': '#d62728',
    }

    for name, result in results_dict.items():
        g = result['global_opt']
        ax_top.plot(np.arange(len(g)), g, label=name,
                    color=colors.get(name), linewidth=2)

    if rccbo and rccbo.get('refinement_trials'):
        for rt in rccbo['refinement_trials']:
            ax_top.axvline(x=rt, color='#d62728', linestyle='--',
                           alpha=0.5, linewidth=1)

    ax_top.set_ylabel('Best Y (lower is better)', fontsize=11)
    ax_top.set_title(title or 'RCCBO convergence and partition evolution',
                     fontsize=13)
    ax_top.legend(fontsize=9, loc='upper right')
    ax_top.grid(True, alpha=0.3)

    if rccbo:
        hist = rccbo.get('partition_history', [])
        refinements = rccbo.get('refinement_trials', [])
        boundaries = [0] + list(refinements) + [total_trials]
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            part = hist[i] if i < len(hist) else None
            desc = 'Pure BO' if part is None else partition_description(part)
            n_clusters = 0 if part is None else len(part)
            color = plt.cm.Blues(0.3 + 0.7 * i / max(len(hist), 1))
            ax_bot.barh(0, end - start, left=start, height=0.6, color=color,
                        edgecolor='black', linewidth=0.5)
            ax_bot.text((start + end) / 2, 0,
                        f'{n_clusters} clusters\n{desc}' if n_clusters else desc,
                        ha='center', va='center', fontsize=7, wrap=True)
    ax_bot.set_xlim(0, total_trials)
    ax_bot.set_yticks([])
    ax_bot.set_xlabel('Trial', fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_partition_evolution(partition_history, refinement_trials, total_trials,
                            title=None, save_path=None):
    """
    Visualize how the partition evolves over time.

    Parameters
    ----------
    partition_history : list of list of frozenset
    refinement_trials : list of int
    total_trials : int
    """
    from ccbo.rccbo.partition_ops import partition_description

    fig, ax = plt.subplots(1, 1, figsize=(12, 3))

    boundaries = [0] + refinement_trials + [total_trials]

    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if i < len(partition_history):
            part = partition_history[i]
            if part is None:
                desc = "Pure BO (no partition)"
                n_clusters = 0
            else:
                desc = partition_description(part)
                n_clusters = len(part)
        else:
            desc = "?"
            n_clusters = 0

        color = plt.cm.Blues(0.3 + 0.7 * i / max(len(partition_history), 1))
        ax.barh(0, end - start, left=start, height=0.5, color=color,
                edgecolor='black', linewidth=0.5)
        label = f'{n_clusters} clusters\n{desc}' if n_clusters > 0 else desc
        ax.text((start + end) / 2, 0, label,
                ha='center', va='center', fontsize=7, wrap=True)

    ax.set_xlim(0, total_trials)
    ax.set_xlabel('Trial', fontsize=12)
    ax.set_yticks([])
    ax.set_title(title or 'Partition Evolution', fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()
