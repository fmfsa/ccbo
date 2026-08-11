"""Family-suite figure: each quotient method vs its base on the base paper's
own experiments, in that paper's own metric.

Row 1  CBO vs QCBO      Toy / Synthetic / Coral      best E[Y|do] (min), 10 seeds
Row 2  DCBO vs QDCBO    stat / ind / nonstat (T=3)   best Y per slice (min), 10 seeds
Row 3  MCBO vs QMCBO    ToyGraph / PSAGraph          best reward (max), 5 seeds

Sources: results/family_cbo/{ds}_{arm}_seed{N}.csv (trial, best_y, cum_cost);
results/qdcbo/{algo}_{setup}_..._seed{s}_...csv
(time_index, trial_index, best_so_far_value; slices concatenated on the x
axis); results/qmcbo/trial_results_{label}_{env}_{s}.csv
(trial_number, current_optimal; QMCBO = the joint cluster mechanisms).

Run:  PYTHONPATH=. python scripts/plot_family_suite.py

NOTE: the per-seed CSVs live on the experiment server (results/ is
gitignored); regenerate the checked-in PDF there before release.
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper/figures/family_suite.pdf"
CBO_DIR = "results/family_cbo"
QDCBO_DIR = "results/qdcbo"
QMCBO_DIR = "results/qmcbo"

METHOD_STYLE = {
    "CBO": ("#0072B2", "-"),
    "QCBO": ("#E69F00", "--"),
    "DCBO": ("#009E73", "-"),
    "QDCBO": ("#CC79A7", "--"),
    "MCBO": ("#D55E00", "-"),
    "QMCBO": ("#56B4E9", "--"),
}
DCBO_SENTINEL = 1e7

CBO_PANELS = [("ToyGraph", "Toy"), ("CompleteGraph", "Synthetic"),
              ("SimplifiedCoralGraph", "Coral")]
DCBO_PANELS = [("stat", "stat"), ("ind", "ind"), ("nonstat", "nonstat")]
MCBO_PANELS = [("ToyGraph", "ToyGraph"), ("PSAGraph", "PSAGraph")]


def _band(ax, trajs, color, ls, label):
    """Plot mean +- s.e. of a list of equal-length series."""
    n = min(len(t) for t in trajs)
    arr = np.array([t[:n] for t in trajs])
    m = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0 * m
    x = np.arange(n)
    ax.plot(x, m, color=color, lw=1.6, ls=ls, label=label)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, lw=0)


def cbo_trajs(ds, arm):
    out = []
    for f in sorted(glob.glob(os.path.join(CBO_DIR, f"{ds}_{arm}_seed*.csv"))):
        out.append(pd.read_csv(f)["best_y"].tolist())
    return out


def dcbo_trajs(setup, algo):
    """Concatenate the three time slices; replace the trial-0 sentinel with
    the slice's first real incumbent (the DCBO stack's own convention)."""
    out = []
    for f in sorted(glob.glob(os.path.join(
            QDCBO_DIR, f"{algo}_{setup}_best_so_far_seed*_reps1.csv"))):
        df = pd.read_csv(f)
        series = []
        for t in sorted(df["time_index"].unique()):
            v = df[df["time_index"] == t]["best_so_far_value"].tolist()
            if v and v[0] >= DCBO_SENTINEL / 2:
                v[0] = v[1] if len(v) > 1 else v[0]
            series.extend(v)
        out.append(series)
    return out


def mcbo_trajs(env, label):
    out = []
    for f in sorted(glob.glob(os.path.join(
            QMCBO_DIR, f"trial_results_{label}_{env}_*.csv"))):
        out.append(pd.read_csv(f)["current_optimal"].tolist())
    return out


def main():
    import argparse
    global OUT, CBO_DIR, QDCBO_DIR, QMCBO_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results",
                    help="root holding family_cbo/, qdcbo/, qmcbo/")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    CBO_DIR = os.path.join(args.results_dir, "family_cbo")
    QDCBO_DIR = os.path.join(args.results_dir, "qdcbo")
    QMCBO_DIR = os.path.join(args.results_dir, "qmcbo")
    OUT = args.out

    fig, axes = plt.subplots(3, 3, figsize=(8.6, 6.4))

    for j, (ds, title) in enumerate(CBO_PANELS):
        ax = axes[0][j]
        base, quot = cbo_trajs(ds, "CBO"), cbo_trajs(ds, "QCBO")
        if base:
            _band(ax, base, *METHOD_STYLE["CBO"], label="CBO")
        if quot:
            _band(ax, quot, *METHOD_STYLE["QCBO"], label="QCBO")
        ax.set_title(title, fontsize=10)
        if j == 0:
            ax.set_ylabel("best $E[Y|do]$", fontsize=9)

    for j, (setup, title) in enumerate(DCBO_PANELS):
        ax = axes[1][j]
        base, quot = dcbo_trajs(setup, "dcbo"), dcbo_trajs(setup, "qdcbo")
        if base:
            _band(ax, base, *METHOD_STYLE["DCBO"], label="DCBO")
        if quot:
            _band(ax, quot, *METHOD_STYLE["QDCBO"], label="QDCBO")
        for xs in (10, 20):
            ax.axvline(xs - 0.5, color="black", lw=0.5, alpha=0.25)
        ax.set_title(title + " (T=3)", fontsize=10)
        if j == 0:
            ax.set_ylabel("best $Y$ per slice", fontsize=9)

    for j, (env, title) in enumerate(MCBO_PANELS):
        ax = axes[2][j]
        base, quot = mcbo_trajs(env, "MCBO"), mcbo_trajs(env, "QMCBO")
        if base:
            _band(ax, base, *METHOD_STYLE["MCBO"], label="MCBO")
        if quot:
            _band(ax, quot, *METHOD_STYLE["QMCBO"], label="QMCBO")
        ax.set_title(title, fontsize=10)
        if j == 0:
            ax.set_ylabel("best reward", fontsize=9)
        ax.set_xlabel("trial", fontsize=9)
    axes[1][0].set_xlabel("trial (slices concatenated)", fontsize=8)

    # legend panel: one named entry per method
    ax = axes[2][2]
    ax.axis("off")
    for method, (color, ls) in METHOD_STYLE.items():
        ax.plot([], [], color=color, ls=ls, lw=1.6, label=method)
    ax.legend(loc="center", fontsize=9, frameon=False, ncol=2)

    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    fig.tight_layout(h_pad=1.2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
