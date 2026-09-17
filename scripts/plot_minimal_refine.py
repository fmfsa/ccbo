"""Fig. 3: price of coarsening and its recovery by refinement (Experiment C).

MediatedChain: fine CBO reaches y* = 0.04 through do(X1); fixed-partition
QCBO is capped at V(Pi) = 4 (the whole-cluster arm clamps X2 into the
policy box); HQCBO (graph-free refinement, arm code HQCBOGF) runs coarse
until the plateau trigger, splits {X1, X2} from the declared hierarchy
alone, and descends to y*.

Left panel: best-so-far population objective (log scale) with unlabeled
reference lines at V(Pi), y* and the mean trigger trial. Right panel:
cumulative incumbent regret (r_0 excluded) with the unlabeled linear floor
Delta(Pi) t = 3.96 t. No titles or annotations: those values belong in the
manuscript caption.

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
from scripts.publication_methods import publication_name, required_files, write_plot_provenance

C_BASE, C_QUOT, C_REF = "#0072B2", "#D55E00", "#009E73"
C_BO = "#E6B800"
C_GF = "#56B4E9"
LABELS = {"BO": ("BO", C_BO, ":"),
          "CBO": ("CBO", C_BASE, "-"),
          "QCBO": ("QCBO", C_QUOT, "-"),
          "HQCBOGF": ("HQCBO", C_REF, "-")}


def _trajs(results_dir, arm):
    out = []
    for f in required_files(results_dir, f"{mb.MC_NAME}_C0_{arm}"):
        out.append(pd.read_csv(f)["best_y"].to_numpy(float))
    return out


def _regret(arr, y_star):
    r = arr - y_star
    r[:, 0] = 0.0
    return np.cumsum(r, axis=1)


def _band(ax, arr, color, label, ls="-", lw=1.6):
    m = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0 * m
    x = np.arange(arr.shape[1])
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, lw=0)


def _letter(ax, s):
    ax.text(0.0, 1.02, s, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/minimal")
    ap.add_argument("--out", default="paper/figures/minimal_refine.pdf")
    args = ap.parse_args()

    o = mb.ORACLE[mb.MC_NAME]
    delta = o["v_pi"] - o["y_star"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45, 1.9))

    data = {}
    for arm in ("BO", "CBO", "QCBO", "HQCBOGF"):
        ts = _trajs(args.dir, arm)
        if ts:
            n = min(len(t) for t in ts)
            data[arm] = np.array([t[:n] for t in ts])

    # ---- left: best-so-far (log) -----------------------------------------
    ax1.axhline(o["v_pi"], color=C_QUOT, lw=0.8, ls=":", alpha=0.8)
    ax1.axhline(o["y_star"], color="black", lw=0.8, ls=":", alpha=0.8)
    for arm, arr in data.items():
        _, color, ls = LABELS[arm]
        label = publication_name(arm)
        _band(ax1, arr, color, label, ls)
    info_path = os.path.join(args.dir, "refine_info.json")
    trig = []
    if os.path.exists(info_path):
        with open(info_path) as fh:
            info = json.load(fh)
        # engine v3 keys carry the arm ({scm}_{cond}_{arm}_seed{N}); the
        # graph-free arm's trigger is the plateau of the shared QCBO prefix.
        trig = [v["trigger_step"] for k, v in info.items()
                if k.startswith(mb.MC_NAME) and "HQCBOGF" in k
                and v.get("trigger_step") is not None]
    if trig:
        ax1.axvline(np.mean(trig), color=C_REF, lw=0.8, ls="--", alpha=0.6)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"best $E[Y\,|\,\mathrm{do}]$", fontsize=8)

    # ---- right: cumulative incumbent regret with linear floor ------------
    for arm, arr in data.items():
        _, color, ls = LABELS[arm]
        label = publication_name(arm)
        _band(ax2, _regret(arr, o["y_star"]), color, label, ls)
    t = np.arange(ax2.get_xlim()[1] + 1)
    ax2.plot(t, delta * t, color="black", lw=0.8, ls=":", alpha=0.7)
    ax2.set_ylabel(r"$R_t$", fontsize=8)

    _letter(ax1, "(a)")
    _letter(ax2, "(b)")
    for ax in (ax1, ax2):
        ax.set_xlabel("trial", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, -0.1),
               columnspacing=1.0, handlelength=2.0)

    fig.tight_layout(w_pad=0.6)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    write_plot_provenance(args.out, args.dir, [(mb.MC_NAME, "C0", a) for a in data])

    # Caption stamps.
    if trig:
        print(f"mean trigger trial: {np.mean(trig):.1f} (n={len(trig)}, min={min(trig)}, max={max(trig)})")
    for arm, arr in data.items():
        reg = _regret(arr, o["y_star"])[:, -1]
        se = (arr[:, -1].std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        rse = (reg.std(ddof=1) / np.sqrt(len(reg))) if len(reg) > 1 else 0.0
        print(f"{arm}: final {arr[:, -1].mean():.3f} +- {se:.3f}; "
              f"R_T {reg.mean():.2f} +- {rse:.2f}")


if __name__ == "__main__":
    main()
