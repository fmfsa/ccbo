"""Emit the cross-family price-of-coarsening table.

One block per base-method family, quotient vs base ONLY, each on the
experiment suite the base method itself ran (where the base did not run, the
quotient does not either):

  CBO  vs QCBO   -- the 6 CausalBO-survey datasets (5 seeds, T=100); source:
                    paper/tables/benchmark_gap.tex (the checked-in scored
                    cells; CBO is QCBO at the identity partition, Prop. 1).
  DCBO vs QDCBO  -- DCBO's own DBN setups stat/ind/nonstat (10 seeds, T=3,
                    10 trials per slice); source: the runner CSVs under
                    third_party/.../DCBO/results/_qdcbo/.
  MCBO vs QMCBO  -- MCBO's environments (5 seeds, T=100); source:
                    results/qmcbo_pilot.json (QMCBO = the joint variant).

Columns: base final Y, quotient final Y (mean +- s.e.) and the realized price,
signed so that a positive price always means the quotient did worse in the
task's own direction (min or max).

Run:  PYTHONPATH=. python scripts/emit_family_price_table.py
"""

import os
import re
import glob
import json

import numpy as np
import pandas as pd

OUT = "paper/tables/family_price.tex"
QDCBO_DIR = "third_party/CausalBO_Benchmark/baselines/DCBO/results/_qdcbo"

# task direction per row (min: price = quotient - base; max: base - quotient)
CBO_TASKS = {"ToyGraph": "min", "Synthetic-2": "min", "Synthetic": "min",
             "Healthcare": "min", "Epidemiology": "min", "Ecology": "max"}
MCBO_ENVS = [("ToyGraph", "max"), ("Synthetic_2", "max"), ("PSAGraph", "max")]
DCBO_SETUPS = [("stat", "min"), ("ind", "min"), ("nonstat", "min")]


def _sem(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _fmt(mean, sem):
    return f"${mean:.2f}{{\\scriptstyle\\,\\pm\\,}}{sem:.2f}$"


def cbo_family():
    """Parse the checked-in benchmark_gap.tex fragment (finest==CBO rows and
    coarse==QCBO rows carry final Y as the first numeric column)."""
    src = open("paper/tables/benchmark_gap.tex").read()
    num = r"\$(-?\d+\.\d+) \\pm (\d+\.\d+)\$"
    rows = []
    for block in src.split(r"\multirow")[1:]:
        ds = re.match(r"\{2\}\{\*\}\{([^}]*)\}", block).group(1)
        pairs = re.findall(num, block)
        # column order per row: finalY, GAP, PA-GAP; row order: finest, coarse
        base = tuple(map(float, pairs[0]))
        quot = tuple(map(float, pairs[3]))
        rows.append((ds, CBO_TASKS.get(ds, "min"), base, quot))
    return rows


def dcbo_family():
    rows = []
    for setup, task in DCBO_SETUPS:
        vals = {}
        for algo in ("dcbo", "qdcbo"):
            finals = []
            for f in sorted(glob.glob(
                    os.path.join(QDCBO_DIR, f"{algo}_{setup}_best_so_far_"
                                            f"seed*_reps1.csv"))):
                df = pd.read_csv(f)
                last_t = df["time_index"].max()
                finals.append(float(
                    df[df["time_index"] == last_t]["best_so_far_value"].iloc[-1]))
            if finals:
                vals[algo] = (float(np.mean(finals)), _sem(finals), len(finals))
        if "dcbo" in vals and "qdcbo" in vals:
            rows.append((setup, task, vals["dcbo"][:2], vals["qdcbo"][:2],
                         vals["dcbo"][2]))
    return rows


def mcbo_family():
    d = json.load(open("results/qmcbo_pilot.json"))["e1"]
    rows = []
    for env, task in MCBO_ENVS:
        b = d.get(f"{env}|MCBO")
        q = d.get(f"{env}|QMCBOJ")
        if b and q:
            rows.append((env.replace("_", r"\_"), task,
                         (b["final_mean"], b["final_sem"]),
                         (q["final_mean"], q["final_sem"])))
    return rows


def price(task, base, quot):
    p = (quot[0] - base[0]) if task == "min" else (base[0] - quot[0])
    return f"${p:+.2f}$"


def main():
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"\textbf{Family} & \textbf{Experiment} & \textbf{Base final $Y$} & "
             r"\textbf{Quotient final $Y$} & \textbf{Price} & \textbf{Task} \\",
             r"\midrule"]

    for ds, task, base, quot in cbo_family():
        lines.append(
            rf"\CBO{{}} vs \QCBO{{}} & {ds} & {_fmt(*base)} & {_fmt(*quot)} & "
            rf"{price(task, base, quot)} & {task} \\")
    lines.append(r"\midrule")
    for setup, task, base, quot, n in dcbo_family():
        lines.append(
            rf"DCBO vs \QDCBO{{}} & {setup} (T=3) & {_fmt(*base)} & "
            rf"{_fmt(*quot)} & {price(task, base, quot)} & {task} \\")
    lines.append(r"\midrule")
    for env, task, base, quot in mcbo_family():
        lines.append(
            rf"MCBO vs \QMCBO{{}} & {env} & {_fmt(*base)} & {_fmt(*quot)} & "
            rf"{price(task, base, quot)} & {task} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
