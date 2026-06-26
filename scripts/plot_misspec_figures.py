"""Generate the two ClusterBench10 misspecification figures for the paper.

F1  figures/clusterbench10_overlays.pdf -- best-so-far trajectories, correct
    (solid) vs intra bow P1 (dashed), per method (reads the per-run CSVs the
    driver writes); QCBO-coarse's two curves coincide.
F2  figures/clusterbench10_dial.pdf -- left: dFinalY vs intra-cluster severity
    (P0,S1,S2,S3); right: dFinalY under the inter-cluster bow (Pic). QCBO-coarse
    is flat at 0 on the left and jumps on the right.

Run:  PYTHONPATH=. python scripts/plot_misspec_figures.py
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ccbo import benchmark

JSON = "results/clusterbench10_misspec.json"
RUNDIR = os.path.join(benchmark.BENCH_ROOT, "results", "_misspec")
OUTDIR = "paper/figures"
METHOD_ORDER = ["BO", "CBO", "QCBO-finest", "QCBO-coarse"]
COLORS = {"BO": "#888888", "CBO": "#1f77b4",
          "QCBO-finest": "#ff7f0e", "QCBO-coarse": "#d62728"}


def _mean_traj(label, pid):
    """Mean best-so-far trajectory over seeds for (method label, perturbation)."""
    files = sorted(glob.glob(os.path.join(RUNDIR, f"{label}_{pid}_seed*.csv")))
    trajs = []
    for f in files:
        try:
            trajs.append(pd.read_csv(f)["current_optimal"].to_numpy())
        except Exception:
            pass
    if not trajs:
        return None
    n = min(len(t) for t in trajs)
    return np.mean([t[:n] for t in trajs], axis=0)


def fig_overlays(methods):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in methods:
        c = COLORS.get(m, None)
        t0, t1 = _mean_traj(m, "P0"), _mean_traj(m, "P1")
        if t0 is not None:
            ax.plot(range(len(t0)), t0, color=c, lw=1.8, label=f"{m} (correct)")
        if t1 is not None:
            ax.plot(range(len(t1)), t1, color=c, lw=1.8, ls="--",
                    label=f"{m} (misspec.)")
    ax.set_xlabel("trial"); ax.set_ylabel("best-so-far $Y$")
    ax.set_title("Correct (solid) vs intra-cluster bow (dashed)")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "clusterbench10_overlays.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def fig_dial(data, methods):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    sev = [("P0", 0), ("S1", 1), ("S2", 2), ("S3", 3)]
    for m in methods:
        xs, ys = [], []
        for pid, k in sev:
            cell = data["methods"][m].get(pid)
            if cell is not None:
                xs.append(k); ys.append(cell["dGAP"][0])
        if xs:
            axL.plot(xs, ys, marker="o", color=COLORS.get(m), label=m)
    axL.axhline(0, color="k", lw=0.6, ls=":")
    axL.set_xlabel("number of intra-cluster edits"); axL.set_ylabel(r"$\Delta$GAP@100")
    axL.set_title("Intra-cluster severity"); axL.set_xticks([0, 1, 2, 3])
    axL.legend(fontsize=7, frameon=False)

    vals = [data["methods"][m]["Pic"]["dGAP"][0] for m in methods
            if "Pic" in data["methods"][m]]
    labs = [m for m in methods if "Pic" in data["methods"][m]]
    axR.bar(range(len(labs)), vals, color=[COLORS.get(m) for m in labs])
    axR.axhline(0, color="k", lw=0.6)
    axR.set_xticks(range(len(labs)))
    axR.set_xticklabels([m.replace("QCBO-", "Q-") for m in labs], rotation=30, fontsize=7)
    axR.set_ylabel(r"$\Delta$GAP@100"); axR.set_title("Inter-cluster bow (Pic)")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "clusterbench10_dial.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def main():
    with open(JSON) as f:
        data = json.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    methods = [m for m in METHOD_ORDER if m in data["methods"]]
    fig_overlays(methods)
    fig_dial(data, methods)


if __name__ == "__main__":
    main()
