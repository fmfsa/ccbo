"""Ablation figure (engine v3): what the causal prior and the arm menu buy.

(a) ParallelParent, correct graph A0 and the arm-excluding A1: fine CBO with
    and without causal priors, the structure-free all-subsets BO-S, and joint
    BO.  (b) FrontDoor B0 / B1: CBO vs CBO-no-prior and QCBO vs QCBO-no-prior
    (the prior-corruption experiment against a no-prior reference).
(c) MediatedChain: graph-informed HQCBO vs graph-free HQCBO-GF after the
    shared trigger, with fixed QCBO and fine CBO as references.

Run:  PYTHONPATH=. python scripts/plot_minimal_ablations.py --dir results/v3/minimal
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

C = {"CBO": "#0072B2", "QCBO": "#D55E00", "HQCBO": "#009E73", "HQCBOGF": "#56B4E9",
     "BO": "#555555", "BOS": "#999999", "AUX": "#E69F00"}


def _trajs(d, scm, cond, arm):
    return [pd.read_csv(f)["best_y"].to_numpy(float) for f in
            sorted(glob.glob(os.path.join(d, f"{scm}_{cond}_{arm}_seed*.csv")))]


def _band(ax, trajs, color, label, ls="-", lw=1.4, alpha=0.15):
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
    ap.add_argument("--dir", default="results/v3/minimal")
    ap.add_argument("--out", default="paper/figures/minimal_ablations.pdf")
    args = ap.parse_args()
    d = args.dir

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    ax = axes[0]
    _band(ax, _trajs(d, mb.PP_NAME, "A0", "CBO"), C["CBO"], "CBO (A0)")
    _band(ax, _trajs(d, mb.PP_NAME, "A0", "CBONP"), C["CBO"], "CBO, no prior (A0)", ls="--")
    _band(ax, _trajs(d, mb.PP_NAME, "A1", "CBO"), C["AUX"], r"CBO (A1: del $X_1\to Y$)")
    _band(ax, _trajs(d, mb.PP_NAME, "A1", "CBONP"), C["AUX"], "CBO, no prior (A1)", ls="--")
    _band(ax, _trajs(d, mb.PP_NAME, "A0", "BOS"), C["BOS"], "BO-S (all subsets)", ls="-.")
    _band(ax, _trajs(d, mb.PP_NAME, "A0", "BO"), C["BO"], "BO (joint arm)", ls=":")
    ax.axhline(mb.pp_gap(), color=C["AUX"], lw=0.7, ls=":", alpha=0.6)
    ax.set_title("(a) ParallelParent: prior vs arm menu", fontsize=9)

    ax = axes[1]
    _band(ax, _trajs(d, mb.FD_NAME, "B0", "CBO"), C["CBO"], "CBO (B0)")
    _band(ax, _trajs(d, mb.FD_NAME, "B1", "CBO"), C["AUX"], r"CBO (B1: assume $X_1\leftrightarrow M$)")
    _band(ax, _trajs(d, mb.FD_NAME, "B0", "CBONP"), C["CBO"], "CBO, no prior", ls="--")
    _band(ax, _trajs(d, mb.FD_NAME, "B0", "QCBO"), C["QCBO"], "QCBO", lw=2.0)
    _band(ax, _trajs(d, mb.FD_NAME, "B0", "QCBONP"), C["QCBO"], "QCBO, no prior", ls="--")
    _band(ax, _trajs(d, mb.FD_NAME, "B0", "BOS"), C["BOS"], "BO-S", ls="-.")
    ax.set_title("(b) FrontDoor: prior corruption vs no prior", fontsize=9)

    ax = axes[2]
    _band(ax, _trajs(d, mb.MC_NAME, "C0", "CBO"), C["CBO"], "CBO (fine)", alpha=0.08)
    _band(ax, _trajs(d, mb.MC_NAME, "C0", "QCBO"), C["QCBO"], "QCBO (fixed)", alpha=0.08)
    _band(ax, _trajs(d, mb.MC_NAME, "C0", "HQCBO"), C["HQCBO"], "HQCBO (graph-informed)", lw=1.8)
    _band(ax, _trajs(d, mb.MC_NAME, "C0", "HQCBOGF"), C["HQCBOGF"], "HQCBO-GF (graph-free)", ls="--", lw=1.8)
    o = mb.ORACLE[mb.MC_NAME]
    ax.axhline(o["v_pi"], color=C["QCBO"], lw=0.7, ls=":", alpha=0.6)
    ax.axhline(o["y_star"], color="black", lw=0.7, ls=":", alpha=0.6)
    ax.set_yscale("log")
    ax.set_title("(c) MediatedChain: refinement variants", fontsize=9)

    for ax in axes:
        ax.set_xlabel("trial", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.5, frameon=True, framealpha=0.9, facecolor="white",
                  edgecolor="none", loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"best $E[Y\,|\,\mathrm{do}]$", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
