"""Fig. 3: price of coarsening and its recovery by refinement (Experiment C).

MediatedChain: fine CBO reaches y* = 0.04 through do(X1); fixed-partition
QCBO is capped at V(Pi) = 4 (the whole-cluster arm clamps X2 into the
policy box); HQCBO runs coarse until the plateau trigger, splits {X1, X2},
and descends to y*. The figure reports best-so-far value with the two
analytic reference lines and the mean trigger time. Cumulative incumbent
regret is reported numerically in the paper rather than repeated graphically.

Style follows scripts/plot_family_suite.py (Okabe-Ito, mean +- s.e. bands).

Run:  PYTHONPATH=. python scripts/plot_minimal_refine.py \
          [--dir results/minimal] [--out paper/figures/minimal_refine.pdf]
"""

import os
import sys
import glob
import json
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

C_BASE, C_QUOT, C_REF = "#0072B2", "#D55E00", "#009E73"
C_BO = "#555555"
LABELS = {"BO": ("BO (joint arm, no graph)", C_BO, ":"),
          "CBO": ("CBO (fine)", C_BASE, "-"),
          "QCBO": ("QCBO (fixed coarse)", C_QUOT, "-"),
          "HQCBO": ("HQCBO (refine)", C_REF, "-")}


def _trajs(results_dir, arm):
    out = []
    for f in sorted(glob.glob(os.path.join(
            results_dir, f"{mb.MC_NAME}_C0_{arm}_seed*.csv"))):
        out.append(pd.read_csv(f)["best_y"].to_numpy(float))
    return out


def _band(ax, arr, color, label, ls="-", lw=1.6):
    m = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0 * m
    x = np.arange(arr.shape[1])
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/minimal")
    ap.add_argument("--out", default="paper/figures/minimal_refine.pdf")
    args = ap.parse_args()

    o = mb.ORACLE[mb.MC_NAME]
    fig, ax1 = plt.subplots(figsize=(3.45, 2.25))

    data = {}
    for arm in ("BO", "CBO", "QCBO", "HQCBO"):
        ts = _trajs(args.dir, arm)
        if ts:
            n = min(len(t) for t in ts)
            data[arm] = np.array([t[:n] for t in ts])

    # ---- top: best-so-far -------------------------------------------------
    ax1.axhline(o["v_pi"], color=C_QUOT, lw=0.8, ls=":", alpha=0.8)
    ax1.axhline(o["y_star"], color="black", lw=0.8, ls=":", alpha=0.8)
    ax1.annotate(r"$V(\Pi)=4$", xy=(0.99, o["v_pi"]), xycoords=("axes fraction", "data"),
                 xytext=(0.99, o["v_pi"] * 1.35), fontsize=7, color=C_QUOT,
                 ha="right")
    ax1.annotate(r"$y^{*}=0.04$", xy=(0.99, o["y_star"]),
                 xycoords=("axes fraction", "data"),
                 xytext=(0.99, o["y_star"] * 2.6), fontsize=7, color="black",
                 ha="right")
    for arm, arr in data.items():
        label, color, ls = LABELS[arm]
        _band(ax1, arr, color, label, ls)
    # mean trigger time
    info_path = os.path.join(args.dir, "refine_info.json")
    if os.path.exists(info_path):
        with open(info_path) as fh:
            info = json.load(fh)
        trig = [v["trigger_step"] for k, v in info.items()
                if k.startswith(mb.MC_NAME) and v["trigger_step"] is not None]
        if trig:
            ax1.axvline(np.mean(trig), color=C_REF, lw=0.8, ls="--",
                        alpha=0.6)
            ax1.annotate(rf"trigger $\bar t={np.mean(trig):.1f}$",
                         xy=(np.mean(trig), 0.98),
                         xycoords=("data", "axes fraction"),
                         fontsize=7, color=C_REF, ha="left", va="top",
                         xytext=(np.mean(trig) + 1, 0.98))
    ax1.set_yscale("log")
    ax1.set_ylabel(r"best $E[Y\,|\,\mathrm{do}]$", fontsize=8)
    ax1.set_xlabel("trial", fontsize=8)
    ax1.legend(fontsize=6.5, frameon=True, framealpha=0.9,
               facecolor="white", edgecolor="none", loc="center right")
    ax1.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    # Caption stamps.
    for arm, arr in data.items():
        print(f"{arm}: final {arr[:, -1].mean():.3f} "
              f"+- {arr[:, -1].std(ddof=1) / np.sqrt(len(arr)):.3f}"
              if len(arr) > 1 else f"{arm}: final {arr[:, -1].mean():.3f}")


if __name__ == "__main__":
    main()
