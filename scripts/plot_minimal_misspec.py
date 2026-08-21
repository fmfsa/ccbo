"""Fig. 2: MinimalBench misspecification trajectories (Experiments A + B).

Panel (a) — exploration-set corruption (ParallelParent, A1): fine CBO under
the correct graph reaches y* = 0; under the quotient-redundant deletion of
X1->Y it permanently plateaus at 4.25; QCBO's trajectories under A0 and A1
coincide exactly; the quotient-visible A3 (negative control) visibly shifts
QCBO.

Panel (b) — prior corruption (FrontDoor, B1: assumed intra-cluster
confounder X1<->M): both fine arms — including the optimal do(M) — drop to
the uninformative tier, so fine CBO forfeits its do-calculus head start and
recovers through interventional data (arm membership intact); QCBO under B1
coincides exactly with B0. (ParallelParent's A2 bow is reported in Table 1:
its sup|Delta| is 0 even for fine CBO, because the corrupted singleton arm
never matters — prior corruption bites only when the corrupted prior sits
on an arm that matters.)

Style follows scripts/plot_family_suite.py (Okabe-Ito, mean +- s.e. bands).

Run:  PYTHONPATH=. python scripts/plot_minimal_misspec.py \
          [--dir results/minimal] [--out paper/figures/minimal_misspec.pdf]
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

C_BASE, C_QUOT, C_AUX, C_NEG = "#0072B2", "#D55E00", "#E69F00", "#CC79A7"


def _trajs(results_dir, cond, arm, scm=mb.PP_NAME):
    out = []
    for f in sorted(glob.glob(os.path.join(
            results_dir, f"{scm}_{cond}_{arm}_seed*.csv"))):
        out.append(pd.read_csv(f)["best_y"].tolist())
    return out


def _band(ax, trajs, color, label, ls="-", lw=1.6, alpha=0.18):
    if not trajs:
        return
    n = min(len(t) for t in trajs)
    arr = np.array([t[:n] for t in trajs])
    m = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0 * m
    x = np.arange(n)
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(x, m - se, m + se, color=color, alpha=alpha, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/minimal")
    ap.add_argument("--out", default="paper/figures/minimal_misspec.pdf")
    args = ap.parse_args()

    # Stack the panels for a compact, single-column conference figure.
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.25), sharex=True)
    gap = mb.pp_gap()

    # ---- (a) Experiment A: exploration-set corruption ---------------------
    ax = axes[0]
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.axhline(gap, color=C_BASE, lw=0.8, ls=":", alpha=0.7)
    _band(ax, _trajs(args.dir, "A0", "CBO"), C_BASE, "CBO, correct")
    _band(ax, _trajs(args.dir, "A1", "CBO"), C_BASE,
          r"CBO, del $X_1\to Y$", ls="--")
    _band(ax, _trajs(args.dir, "A0", "QCBO"), C_QUOT, "QCBO, correct",
          lw=2.2)
    # coincident by the contract: thin black dashed overlay on the thick line
    _band(ax, _trajs(args.dir, "A1", "QCBO"), "black",
          r"QCBO, del $X_1\to Y$ ($\equiv$)", ls=(0, (2, 2)), lw=0.8,
          alpha=0.0)
    _band(ax, _trajs(args.dir, "A3", "QCBO"), C_NEG,
          r"QCBO, assume $X_1\leftrightarrow Y$", ls="-.", lw=1.3)
    ax.set_title("(a) exploration-set corruption", fontsize=9)

    # ---- (b) Experiment B: prior corruption (FrontDoor) -------------------
    ax = axes[1]
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    _band(ax, _trajs(args.dir, "B0", "CBO", mb.FD_NAME), C_BASE,
          "CBO, correct")
    _band(ax, _trajs(args.dir, "B1", "CBO", mb.FD_NAME), C_AUX,
          r"CBO, assume $X_1\leftrightarrow M$", ls="--")
    _band(ax, _trajs(args.dir, "B0", "QCBO", mb.FD_NAME), C_QUOT,
          "QCBO, correct", lw=2.2)
    _band(ax, _trajs(args.dir, "B1", "QCBO", mb.FD_NAME), "black",
          r"QCBO, assume $X_1\leftrightarrow M$ ($\equiv$)",
          ls=(0, (2, 2)), lw=0.8, alpha=0.0)
    ax.set_title("(b) prior corruption (FrontDoor)", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=6.5, frameon=True, framealpha=0.9,
                  facecolor="white", edgecolor="none", loc="upper right")
    axes[1].set_xlabel("trial", fontsize=8)
    fig.supylabel(r"best $E[Y\,|\,\mathrm{do}]$", fontsize=8, x=0.01)

    fig.tight_layout(h_pad=0.8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    # Invariance stamp for the caption: sup-norm paired deltas.
    for scm, base, cond in ((mb.PP_NAME, "A0", "A1"), (mb.PP_NAME, "A0", "A2"),
                            (mb.FD_NAME, "B0", "B1")):
        d = []
        for f0 in sorted(glob.glob(os.path.join(
                args.dir, f"{scm}_{base}_QCBO_seed*.csv"))):
            fc = f0.replace(f"_{base}_", f"_{cond}_")
            if os.path.exists(fc):
                y0 = pd.read_csv(f0)["best_y"].to_numpy(float)
                yc = pd.read_csv(fc)["best_y"].to_numpy(float)
                d.append(np.abs(y0 - yc).max())
        if d:
            print(f"QCBO sup|Delta| {base}-vs-{cond}: {max(d):.3g}")


if __name__ == "__main__":
    main()
