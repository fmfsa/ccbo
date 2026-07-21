"""Figure: QCBO-refine on ClusterBench10L (the lossy variant).

figures/clusterbench10L_refine.pdf -- two stacked panels (single column).
Top: best-so-far Y (mean +/- s.e. over seeds) for BO, CBO (=Q-finest),
Q-coarse, Q-refine, with y* and the mean trigger step marked. Q-refine rides
Q-coarse until the trigger, then descends to y*.
Bottom: running cumulative incumbent regret R_t = sum_{u<=t} (Y_u - y*):
Q-coarse's is linear in t (the price paid forever, Prop. 5), Q-refine's
flattens after the trigger (the price as a bounded transient, Remark
rem:regret).

Reads results/clusterbench10_refine.json (written by
scripts/run_clusterbench10_refine.py). Fails loudly if a method is missing.

Run:  PYTHONPATH=. python scripts/plot_refine_figure.py
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = "results/clusterbench10_refine.json"
OUTDIR = "paper/figures"
DS = "ClusterBench10L"

# Okabe-Ito, consistent with the other paper figures.
DRAW = [("BO", "BO", "#E69F00"),
        ("CBO", "CBO", "#0072B2"),
        ("QCBO-coarse", "QCBO", "#D55E00"),
        ("QCBO-refine", "HQCBO", "#009E73")]


def main():
    with open(JSON) as f:
        data = json.load(f)
    entry = data["datasets"][DS]
    ystar, task = entry["y_star"], entry["task"]
    triggers = [t for t in entry["trigger_steps"].values() if t is not None]
    t_mean = float(np.mean(triggers)) if triggers else None

    fig, (axY, axR) = plt.subplots(2, 1, figsize=(3.5, 3.8), sharex=True)
    for m, label, c in DRAW:
        if m == "CBO" and m not in entry["curves"]:
            m = "QCBO-finest"   # CBO == QCBO-finest on 10L (Prop. 1)
        assert m in entry["curves"], f"method {m} missing from {JSON}"
        mean = np.asarray(entry["curves"][m]["mean"], dtype=float)
        sem = np.asarray(entry["curves"][m]["sem"], dtype=float)
        x = np.arange(len(mean))
        axY.fill_between(x, mean - sem, mean + sem, color=c, alpha=0.15, lw=0)
        axY.plot(x, mean, color=c, lw=1.8, label=label)
        r = np.maximum(mean - ystar if task == "min" else ystar - mean, 0.0)
        R = np.concatenate([[0.0], np.cumsum(r[1:])])  # r_0 excluded (shared init)
        axR.plot(x, R, color=c, lw=1.8)

    # Zoom to the near-optimum region (init incumbents ~12-15 would compress
    # the plateau-vs-descent contrast the panel exists to show).
    axY.set_ylim(-0.25, 6.0)
    axY.axhline(ystar, color="k", lw=0.8, ls=":")
    axY.annotate(r"$y^\star$", xy=(96, ystar), fontsize=8,
                 xytext=(96, ystar + 0.25))
    if t_mean is not None:
        for ax in (axY, axR):
            ax.axvline(t_mean, color="#009E73", lw=0.9, ls="--", alpha=0.7)
        axY.annotate("refine", xy=(t_mean, 5.7), fontsize=8,
                     color="#009E73", ha="left", va="top",
                     xytext=(t_mean + 2, 5.7))
    axY.set_ylabel("best-so-far $Y$", fontsize=10)
    axY.tick_params(labelsize=8)
    axY.legend(fontsize=7.5, frameon=False, loc="upper right")
    axR.set_xlabel("trial", fontsize=10)
    axR.set_ylabel("cumulative regret $R_t$", fontsize=10)
    axR.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "clusterbench10L_refine.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (mean trigger step: {t_mean})")


if __name__ == "__main__":
    main()
