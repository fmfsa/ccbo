"""Generate the two ClusterBench10 misspecification figures for the paper.

F1  figures/clusterbench10_overlays.pdf -- 2x4 grid, one column per method.
    Top row: best-so-far Y (mean +/- s.e. band over seeds), correct DAG
    (solid) vs intra-cluster bow P1 (dashed), full range (no clipping).
    Bottom row: simple regret of the incumbent (best-so-far - y*) on a log
    scale, so early and late behaviour are both legible. QCBO-coarse's two
    curves coincide exactly -- the visual counterpart of Prop. 2 -- and each
    panel is annotated with its invariance flag.
F2  figures/clusterbench10_dial.pdf -- dPA-GAP for the whole perturbation
    field, split at the exact scope boundary of Prop. 2: left panel = the
    quotient-invisible (protected) perturbations, right panel = the
    quotient-visible controls where the guarantee is allowed to break.
    Invariant cells (trajectory equality across all seeds) are drawn as
    filled markers; others as open markers.

Both figures fail loudly (assert) if an expected method/perturbation is missing
from the JSON, rather than silently plotting incomplete data.

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
METHODS = ["BO", "CBO", "QCBO-finest", "QCBO-coarse"]
# Okabe-Ito (colorblind-safe), matching the paper's TikZ palette.
COLORS = {"BO": "#999999", "CBO": "#0072B2",
          "QCBO-finest": "#E69F00", "QCBO-coarse": "#D55E00"}
# Perturbations split at the Prop. 2 scope boundary.
PROTECTED = ["P1", "P2", "P3", "S1", "S2", "S3", "P6"]   # quotient-invisible
VISIBLE = ["P5", "Pic"]                                   # quotient-visible
PERT_TEX = {
    "P1": "P$_1$", "P2": "P$_2$", "P3": "P$_3$",
    "S1": "S$_1$", "S2": "S$_2$", "S3": "S$_3$",
    "P5": "P$_5$", "P6": "P$_6$", "Pic": "P$_{ic}$",
}


def _short(m):
    return m.replace("QCBO-", "Q-")


def _cell(data, m, pid):
    assert m in data["methods"], f"method {m} missing from {JSON}"
    cell = data["methods"][m].get(pid)
    assert cell is not None, f"perturbation {pid} missing for {m} in {JSON}"
    return cell


def _traj_stats(label, pid):
    """(mean, sem) best-so-far trajectory over seeds for (method, pert)."""
    files = sorted(glob.glob(os.path.join(RUNDIR, f"{label}_{pid}_seed*.csv")))
    assert files, f"no CSVs for {label}/{pid} under {RUNDIR}"
    trajs = [pd.to_numeric(pd.read_csv(f)["current_optimal"],
                           errors="coerce").to_numpy(dtype=float)
             for f in files]
    n = min(len(t) for t in trajs)
    T = np.array([t[:n] for t in trajs])
    mean = T.mean(axis=0)
    sem = (T.std(axis=0, ddof=1) / np.sqrt(len(T))) if len(T) > 1 \
        else np.zeros(n)
    return mean, sem


def fig_overlays(data):
    ystar = data["y_star"]
    task = data.get("task", "min")
    fig, axes = plt.subplots(2, len(METHODS), figsize=(9.0, 4.6),
                             sharex=True, sharey="row")
    for j, m in enumerate(METHODS):
        c = COLORS[m]
        m0, s0 = _traj_stats(m, "P0")
        m1, s1 = _traj_stats(m, "P1")
        x0, x1 = np.arange(len(m0)), np.arange(len(m1))
        bid = _cell(data, m, "P1")["byte_identical_to_P0"]

        # Top: best-so-far Y, full range, +/- s.e. band.
        ax = axes[0, j]
        ax.fill_between(x0, m0 - s0, m0 + s0, color=c, alpha=0.18, lw=0)
        ax.plot(x0, m0, color=c, lw=1.8, label="correct DAG")
        ax.plot(x1, m1, color=c, lw=1.6, ls="--", label="misspecified")
        if bid:
            ax.set_title(f"{_short(m)}  (invariant ✓)", fontsize=10)
            ax.annotate("curves coincide", xy=(0.97, 0.55),
                        xycoords="axes fraction", ha="right", fontsize=8,
                        style="italic", color=c)
        else:
            ax.fill_between(x1, m1 - s1, m1 + s1, color=c, alpha=0.12, lw=0)
            ax.set_title(f"{_short(m)}  (perturbed ✗)", fontsize=10)
        ax.tick_params(labelsize=8)

        # Bottom: simple regret of the incumbent, log scale.
        ax = axes[1, j]
        r0 = np.maximum(m0 - ystar if task == "min" else ystar - m0, 1e-12)
        r1 = np.maximum(m1 - ystar if task == "min" else ystar - m1, 1e-12)
        ax.plot(x0, r0, color=c, lw=1.8)
        ax.plot(x1, r1, color=c, lw=1.6, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("trial", fontsize=10)
        ax.tick_params(labelsize=8)

    axes[0, 0].set_ylabel("best-so-far $Y$", fontsize=10)
    axes[1, 0].set_ylabel(r"simple regret $r_t$", fontsize=10)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Correct DAG (solid) vs intra-cluster bow $P_1$ (dashed); "
                 "bands: $\\pm$ s.e. over seeds", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUTDIR, "clusterbench10_overlays.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def fig_dial(data):
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.0, 3.2), sharey=True,
        gridspec_kw={"width_ratios": [len(PROTECTED), len(VISIBLE) + 0.6]})
    # Several methods sit at (or near) 0, so each method gets a distinct
    # marker and size so coinciding points remain individually visible.
    MARKER = {"BO": ("o", 10), "CBO": ("s", 8),
              "QCBO-finest": ("D", 6), "QCBO-coarse": ("^", 5)}
    OFF = {"BO": -0.24, "CBO": -0.08, "QCBO-finest": 0.08, "QCBO-coarse": 0.24}

    def _panel(ax, pids):
        for m in METHODS:
            xs = np.arange(len(pids)) + OFF[m]
            ys = [_cell(data, m, p)["dPAGAP"][0] for p in pids]
            es = [_cell(data, m, p)["dPAGAP"][1] for p in pids]
            bids = [_cell(data, m, p)["byte_identical_to_P0"] for p in pids]
            c = COLORS[m]
            mk, ms = MARKER[m]
            ax.errorbar(xs, ys, yerr=es, color=c, fmt="none",
                        elinewidth=1.0, capsize=2, zorder=2)
            for x, y, b in zip(xs, ys, bids):
                ax.plot([x], [y], marker=mk, ms=ms, mec=c, zorder=3,
                        mfc=(c if b else "white"),
                        label=_short(m) if x == xs[0] else None)
        ax.axhline(0, color="k", lw=1.0, ls=":", zorder=1)
        ax.set_xticks(range(len(pids)))
        ax.set_xticklabels([PERT_TEX[p] for p in pids], fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    _panel(axL, PROTECTED)
    _panel(axR, VISIBLE)
    axL.set_ylabel(r"$\Delta$PA-GAP (misspecified $-$ correct)", fontsize=10)
    axL.set_title("Quotient-invisible edits (Prop. 2: guarantee holds)",
                  fontsize=10)
    axR.set_title("Quotient-visible edits\n(guarantee out of scope)",
                  fontsize=10)
    # Legend: dedupe (one entry per method), filled = invariant.
    handles, labels = axL.get_legend_handles_labels()
    seen, hs, ls = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); hs.append(h); ls.append(l)
    axL.legend(hs, ls, fontsize=8, frameon=False, loc="upper left", ncol=2)
    axL.annotate("filled marker = trajectory byte-identical to correct DAG",
                 xy=(0.01, 0.02), xycoords="axes fraction", fontsize=8,
                 style="italic")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "clusterbench10_dial.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main():
    with open(JSON) as f:
        data = json.load(f)
    os.makedirs(OUTDIR, exist_ok=True)
    for m in METHODS:
        assert m in data["methods"], f"method {m} missing from {JSON}"
    fig_overlays(data)
    fig_dial(data)


if __name__ == "__main__":
    main()
