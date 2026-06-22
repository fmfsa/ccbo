"""Score the head-to-head baselines + QCBO and emit the comparison table.

Reads the per-seed progress CSVs archived by `run_baselines_suite.py`
(`third_party/CausalBO_Benchmark/results/<DIR>/<ds>/<DIR>_<ds>_<T>-trials_seed<S>_progress.csv`),
scores each seed with the benchmark's own `metrics.GAP` / `metrics.PA_GAP`
against `observational_datasets/<ds>_theoretical_best.json`, aggregates
mean +/- s.e. (ddof=1) across seeds, merges the QCBO rows from
`results/qcbo_benchmark_results.json`, and writes:

  - results/baseline_comparison.json          (full GAP/PA-GAP/final per cell)
  - paper/tables/benchmark_comparison.tex      (GAP@100 matrix, methods x datasets)
  - paper/tables/benchmark_comparison_pagap.tex (PA-GAP@100 twin, for the appendix)

The benchmark `CBO` baseline (their implementation) and `QCBO-finest` (our
identity partition) are *distinct* rows; their proximity is a sanity check on
the QCBO-finest anchor (Prop. 1 is about our internal CBO backend, not theirs).

Run after the baseline suite (and after the primary run wrote the QCBO json):
    PYTHONPATH=. python scripts/emit_baseline_table.py --trials 100
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "third_party", "CausalBO_Benchmark")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
from metrics.GAP import GAP          # noqa: E402
from metrics.PA_GAP import PA_GAP    # noqa: E402

CURATED = [("toyGraph", "ToyGraph"), ("synthetic_2", "Synthetic-2"),
           ("synthetic", "Synthetic"), ("healthcare", "Healthcare"),
           ("epidemiology", "Epidemiology"), ("ecology", "Ecology")]

# (results-dir, display label). QCBO-* are filled from the json, not CSVs.
METHODS = [("BO", "BO"), ("CBO", "CBO"), ("CEO", "CEO"), ("CoCaBO", "CoCaBO"),
           ("DCBO", "DCBO"), ("MCBO", "MCBO"),
           ("QCBO-finest", "\\QCBO{}-finest"), ("QCBO-coarse", "\\QCBO{}-coarse")]
QCBO_KEYS = {"QCBO-finest", "QCBO-coarse"}


def y_star(ds):
    f = os.path.join(BENCH, "observational_datasets", f"{ds}_theoretical_best.json")
    return json.load(open(f))["global_best_value"]


def task_of(ds):
    return yaml.safe_load(open(os.path.join(BENCH, "configs", f"{ds}.yaml"))).get("task", "min")


def mean_se(xs):
    a = np.asarray(xs, float)
    if a.size == 0:
        return None
    se = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return [float(a.mean()), se]


def score_method_csvs(dir_, ds, trials):
    """Return {'gap':(m,se),'pagap':(m,se),'final':(m,se),'n':k} over seed CSVs."""
    yb, task = y_star(ds), task_of(ds)
    pat = os.path.join(BENCH, "results", dir_, ds, f"{dir_}_{ds}_{trials}-trials_seed*_progress.csv")
    gaps, pagaps, finals = [], [], []
    for csv in sorted(glob.glob(pat)):
        df = pd.read_csv(csv)
        col = "current_optimal" if "current_optimal" in df.columns else "best_so_far_value"
        vals = df[col].tolist()
        if len(vals) < 2:
            continue
        v = vals[:trials]
        g = GAP(yb); g.calculate_GAP(v, task)
        p = PA_GAP(yb); p.calculate_PA_GAP(v, task)
        gaps.append(g.GAP_value)
        pagaps.append(p.PA_GAP_value)
        finals.append(min(v) if task == "min" else max(v))
    if not gaps:
        return None
    return dict(gap=mean_se(gaps), pagap=mean_se(pagaps), final=mean_se(finals), n=len(gaps))


def qcbo_from_json(key, ds):
    js = os.path.join(REPO, "results", "qcbo_benchmark_results.json")
    if not os.path.exists(js):
        return None
    data = json.load(open(js))
    if ds not in data or key not in data[ds].get("methods", {}):
        return None
    s = data[ds]["methods"][key]
    return dict(gap=s.get("gap100"), pagap=s.get("pagap100"), final=s.get("final"),
                n=data[ds].get("n_seeds"))


def cell(pair, best=False):
    if pair is None:
        return "---"
    m, se = pair
    body = f"{m:.2f} \\pm {se:.2f}"
    return f"$\\mathbf{{{body}}}$" if best else f"${body}$"


def emit_matrix(comp, metric, out, caption_note):
    """methods x datasets matrix of `metric` (gap/pagap), bold best per column."""
    lines = []
    # best per dataset (higher = better for both GAP and PA-GAP)
    best = {}
    for ds, _ in CURATED:
        vals = [(mk, comp[mk][ds][metric][0]) for mk, _ in METHODS
                if comp.get(mk, {}).get(ds) and comp[mk][ds].get(metric)]
        if vals:
            best[ds] = max(vals, key=lambda t: t[1])[0]
    prev_qcbo = False
    for mk, label in METHODS:
        is_q = mk in QCBO_KEYS
        if is_q and not prev_qcbo and lines:   # separator between field and QCBO block
            lines.append("\\midrule")
        row = [label]
        for ds, _ in CURATED:
            c = comp.get(mk, {}).get(ds)
            row.append(cell(c[metric] if c else None, best.get(ds) == mk))
        lines.append(" & ".join(row) + " \\\\")
        prev_qcbo = is_q
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n\\bottomrule\n")
    print(f"wrote {out}  ({caption_note})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--methods", type=str, default="",
                    help="comma-separated method keys to include (default: all)")
    args = ap.parse_args()

    global METHODS
    if args.methods:
        keep = [m.strip() for m in args.methods.split(",")]
        METHODS = [(k, lbl) for (k, lbl) in METHODS if k in keep]

    comp = {}
    for mk, _ in METHODS:
        comp[mk] = {}
        for ds, _ in CURATED:
            if mk in QCBO_KEYS:
                comp[mk][ds] = qcbo_from_json(mk, ds)
            else:
                comp[mk][ds] = score_method_csvs(mk, ds, args.trials)

    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    with open(os.path.join(REPO, "results", "baseline_comparison.json"), "w") as f:
        json.dump(comp, f, indent=2)

    tdir = os.path.join(REPO, "paper", "tables")
    os.makedirs(tdir, exist_ok=True)
    emit_matrix(comp, "gap", os.path.join(tdir, "benchmark_comparison.tex"),
                "GAP@%d, higher=better" % args.trials)
    emit_matrix(comp, "pagap", os.path.join(tdir, "benchmark_comparison_pagap.tex"),
                "PA-GAP@%d" % args.trials)

    # console summary
    print("\nGAP@%d (mean+/-se):" % args.trials)
    hdr = "method".ljust(14) + "".join(n.center(16) for _, n in CURATED)
    print(hdr)
    for mk, label in METHODS:
        cells = []
        for ds, _ in CURATED:
            c = comp[mk].get(ds)
            cells.append((f"{c['gap'][0]:.2f}+/-{c['gap'][1]:.2f}" if c and c.get('gap') else "--").center(16))
        print(mk.ljust(14) + "".join(cells))


if __name__ == "__main__":
    main()
