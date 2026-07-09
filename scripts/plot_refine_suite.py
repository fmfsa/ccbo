"""Cross-dataset best-so-far + cumulative-regret figure for QCBO-refine.

paper/figures/refine_suite.pdf -- one row per dataset, two columns
(left: best-so-far Y, mean +/- s.e.; right: cumulative incumbent regret R_t).
Datasets: the two ClusterBench variants + the survey datasets refine was run on
(synthetic_2, ecology, healthcare). The dashed vertical line marks the mean
trigger step. Shows the story the numbers carry: refine matches coarse where
lossless, rescues it where lossy (synthetic_2, 10L), and its cumulative regret
tracks the better of {coarse, CBO} rather than coarse's linear growth.

Reads results/clusterbench10_refine.json.
Run:  PYTHONPATH=. python scripts/plot_refine_suite.py
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = "results/clusterbench10_refine.json"
OUTDIR = "paper/figures"

DATASETS = [
    ("ClusterBench10", "ClusterBench10 (lossless)"),
    ("ClusterBench10L", "ClusterBench10L (lossy)"),
    ("synthetic_2", "synthetic_2 (lossy)"),
    ("ecology", "ecology (max)"),
    ("healthcare", "healthcare"),
]
# (json key, legend label, color); CBO falls back to QCBO-finest (Prop. 1).
DRAW = [("BO", "BO", "#E69F00"),
        ("CBO", r"CBO $\equiv$ Q-finest", "#0072B2"),
        ("QCBO-coarse", "Q-coarse", "#D55E00"),
        ("QCBO-refine", "Q-refine", "#009E73")]


def main():
    with open(JSON) as f:
        data = json.load(f)["datasets"]
    dss = [(k, lab) for k, lab in DATASETS if k in data]
    n = len(dss)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 1.85 * n))
    if n == 1:
        axes = axes[None, :]

    for row, (key, label) in enumerate(dss):
        e = data[key]
        ystar, task = e["y_star"], e["task"]
        trigs = [t for t in e["trigger_steps"].values() if t is not None]
        t_mean = float(np.mean(trigs)) if trigs else None
        axY, axR = axes[row]
        for m, leg, c in DRAW:
            mk = m if m in e["curves"] else ("QCBO-finest" if m == "CBO" else m)
            if mk not in e["curves"]:
                continue
            mean = np.asarray(e["curves"][mk]["mean"], dtype=float)
            sem = np.asarray(e["curves"][mk]["sem"], dtype=float)
            x = np.arange(len(mean))
            axY.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.15, lw=0)
            axY.plot(x, mean, color=c, lw=1.6, label=leg)
            r = np.maximum(mean - ystar if task == "min" else ystar - mean, 0.0)
            R = np.concatenate([[0.0], np.cumsum(r[1:])])  # r_0 excluded
            axR.plot(x, R, color=c, lw=1.6)
        if t_mean is not None:
            for ax in (axY, axR):
                ax.axvline(t_mean, color="#009E73", lw=0.9, ls="--", alpha=0.6)
        axY.axhline(ystar, color="k", lw=0.7, ls=":")
        axY.set_ylabel(f"{label}\nbest-so-far $Y$", fontsize=8.5)
        axR.set_ylabel("cum. regret $R_t$", fontsize=8.5)
        for ax in (axY, axR):
            ax.tick_params(labelsize=7.5)
            if row < n - 1:
                ax.set_xticklabels([])
        if row == 0:
            axY.legend(fontsize=7, frameon=False, ncol=2, loc="best")
            axY.set_title("best-so-far $Y$", fontsize=9)
            axR.set_title("cumulative regret", fontsize=9)
        if row == n - 1:
            axY.set_xlabel("trial", fontsize=9)
            axR.set_xlabel("trial", fontsize=9)

    fig.tight_layout(h_pad=0.6, w_pad=1.2)
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "refine_suite.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
