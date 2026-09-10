"""Supplementary cost-indexed figure (engine v3): best-so-far population
objective against initialization-inclusive cumulative cost.

Engine-v3 traces charge the initial design on row 0 and the refinement
arms' split design at the trigger row, so ``cum_cost`` is directly the
total budget.  Curves are step functions (best-so-far at the last logged
point with cost <= budget), averaged over seeds on a common cost grid;
seeds whose run has not reached a budget yet are excluded from the mean at
that budget (n shown as a thin band edge).

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

STYLE = {  # arm -> (label, colour, linestyle)
    "BO": ("BO (joint arm)", "#555555", ":"),
    "BOS": ("BO-S (all subsets)", "#999999", "-."),
    "CBO": ("CBO", "#0072B2", "-"),
    "CBONP": ("CBO, no prior", "#0072B2", "--"),
    "QCBO": ("QCBO", "#D55E00", "-"),
    "QCBONP": ("QCBO, no prior", "#D55E00", "--"),
    "HQCBO": ("HQCBO (graph-informed)", "#009E73", "-"),
    "HQCBOGF": ("HQCBO-GF (graph-free)", "#56B4E9", "--"),
}
PANELS = [
    ("(a) ParallelParent A0", mb.PP_NAME, "A0", ["BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP"], False),
    ("(b) FrontDoor B0", mb.FD_NAME, "B0", ["BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP"], False),
    ("(c) MediatedChain C0", mb.MC_NAME, "C0",
     ["BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP", "HQCBO", "HQCBOGF"], True),
]


def _units(d, scm, cond, arm):
    out = []
    for f in sorted(glob.glob(os.path.join(d, f"{scm}_{cond}_{arm}_seed*.csv"))):
        df = pd.read_csv(f)
        out.append((df["cum_cost"].to_numpy(float), df["best_y"].to_numpy(float)))
    return out


def _step_curve(units, grid):
    vals = np.full((len(units), len(grid)), np.nan)
    for i, (c, y) in enumerate(units):
        idx = np.searchsorted(c, grid + 1e-9, side="right") - 1
        ok = idx >= 0
        vals[i, ok] = y[idx[ok]]
    n = np.sum(~np.isnan(vals), axis=0)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(vals, axis=0)
        se = np.nanstd(vals, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
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
            ax.set_title(title + " (no data)", fontsize=9)
            continue
        grid = np.linspace(min(allc), max(allc), 200)
        for arm, units in data.items():
            label, color, ls = STYLE[arm]
            m, se, n = _step_curve(units, grid)
            full = n == len(units)          # only budgets every seed reached
            ax.plot(grid[full], m[full], color=color, ls=ls, lw=1.4, label=label)
            ax.fill_between(grid[full], (m - se)[full], (m + se)[full],
                            color=color, alpha=0.15, lw=0)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("cumulative cost (incl. initial design)", fontsize=8)
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
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
