"""Supplementary cost-indexed figure (engine v3): best-so-far population
objective against initialization-inclusive cumulative cost.

Engine-v3 traces charge the initial design on row 0 and the refinement
arms' split design at the trigger row, so ``cum_cost`` is directly the
total budget.  Curves are step functions (best-so-far at the last logged
point with cost <= budget), averaged over seeds on a common cost grid.
Each curve is shown only from the largest initial cost to the smallest
final cost across its seeds, so it does not extrapolate beyond recorded runs.

Run:  PYTHONPATH=. python scripts/plot_cost_indexed.py --dir results/v3/minimal
"""

import os
import sys
import glob
import argparse

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ccbo import minibench as mb
from scripts.publication_methods import publication_name, required_files, write_plot_provenance

STYLE = {  # arm -> (label, colour, linestyle)
    "BO": ("BO (joint arm)", "#E6B800", ":"),
    "BOS": ("BO-S (all subsets)", "#999999", "-."),
    "CBO": ("CBO", "#0072B2", "-"),
    "CBONP": ("CBO, no prior", "#0072B2", "--"),
    "QCBO": ("QCBO", "#D55E00", "-"),
    "QCBONP": ("QCBO, no prior", "#D55E00", "--"),
    "HQCBOGF": ("HQCBO", "#009E73", "-"),
}
# Letter-only panel labels; conditions belong in the caption:
# (a) ParallelParent A0, (b) FrontDoor B0, (c) MediatedChain C0.
PANELS = [
    ("(a)", mb.PP_NAME, "A0", ["BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP"], False),
    ("(b)", mb.FD_NAME, "B0", ["BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP"], False),
    ("(c)", mb.MC_NAME, "C0",
     ["BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP", "HQCBOGF"], True),
]


def _letter(ax, s):
    ax.text(0.0, 1.02, s, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left")


def _units(d, scm, cond, arm):
    out = []
    for f in required_files(d, f"{scm}_{cond}_{arm}"):
        df = pd.read_csv(f)
        out.append((df["cum_cost"].to_numpy(float), df["best_y"].to_numpy(float)))
    return out


def _step_curve(units, grid):
    vals = np.full((len(units), len(grid)), np.nan)
    for i, (c, y) in enumerate(units):
        idx = np.searchsorted(c, grid + 1e-9, side="right") - 1
        ok = (idx >= 0) & (grid <= c[-1] + 1e-9)
        vals[i, ok] = y[idx[ok]]
    n = np.sum(~np.isnan(vals), axis=0)
    m = np.full(len(grid), np.nan)
    se = np.full(len(grid), np.nan)
    present = n > 0
    m[present] = np.nanmean(vals[:, present], axis=0)
    multiple = n > 1
    se[multiple] = np.nanstd(vals[:, multiple], axis=0, ddof=1) / np.sqrt(n[multiple])
    return m, se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/v3/minimal")
    ap.add_argument("--out", default="paper/figures/cost_indexed.pdf")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax, (title, scm, cond, arms, logy) in zip(axes, PANELS):
        allc = []
        data = {}
        for arm in arms:
            units = _units(args.dir, scm, cond, arm)
            if units:
                data[arm] = units
                allc += [c[-1] for c, _ in units] + [c[0] for c, _ in units]
        if not data:
            _letter(ax, title)
            continue
        grid = np.unique(np.concatenate([c for units in data.values() for c, _ in units]))
        for arm, units in data.items():
            _, color, ls = STYLE[arm]
            label = publication_name(arm)
            m, se, n = _step_curve(units, grid)
            full = n == len(units)          # only budgets every seed reached
            ax.plot(grid[full], m[full], color=color, ls=ls, lw=1.4, label=label, drawstyle="steps-post")
            ax.fill_between(grid[full], (m - se)[full], (m + se)[full],
                            color=color, alpha=0.15, lw=0, step="post")
        if logy:
            ax.set_yscale("log")
        _letter(ax, title)
        ax.set_xlabel("cumulative cost", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"best $E[Y\,|\,\mathrm{do}]$", fontsize=8)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    write_plot_provenance(args.out, args.dir, [(s,c,a) for _,s,c,arms,_ in PANELS for a in arms])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
