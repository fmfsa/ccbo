"""Emit the family price table: quotient vs base finals on the base paper's
own experiments (same sources as scripts/plot_family_suite.py).

  CBO  vs QCBO   Toy / Synthetic / Coral      (min, 10 seeds, 40 trials)
  DCBO vs QDCBO  stat / ind / nonstat, T=3    (min, 10 seeds, final slice)
  MCBO vs QMCBO  ToyGraph / PSAGraph          (max, 5 seeds, 100 trials)

Price is signed so a positive value always means the quotient ended worse in
the task's own direction.

Run:  PYTHONPATH=. python scripts/emit_family_price_table.py
"""

import os
import glob

import numpy as np
import pandas as pd

OUT = "paper/tables/family_price.tex"
CBO_DIR = "results/family_cbo"
QDCBO_DIR = "third_party/CausalBO_Benchmark/baselines/DCBO/results/_qdcbo"
QMCBO_DIR = "third_party/CausalBO_Benchmark/results/_qmcbo"

CBO_ROWS = [("ToyGraph", "Toy", "min"), ("CompleteGraph", "Synthetic", "min"),
            ("SimplifiedCoralGraph", "Coral", "min")]
DCBO_ROWS = [("stat", "min"), ("ind", "min"), ("nonstat", "min")]
MCBO_ROWS = [("ToyGraph", "max"), ("PSAGraph", "max")]


def _sem(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _fmt(vals):
    return (f"${np.mean(vals):.2f}{{\\scriptstyle\\,\\pm\\,}}{_sem(vals):.2f}$",
            float(np.mean(vals)))


def _price(task, b, q):
    p = (q - b) if task == "min" else (b - q)
    return f"${p:+.2f}$"


def cbo_finals(ds, arm):
    return [pd.read_csv(f)["best_y"].iloc[-1] for f in
            sorted(glob.glob(os.path.join(CBO_DIR, f"{ds}_{arm}_seed*.csv")))]


def dcbo_finals(setup, algo):
    out = []
    for f in sorted(glob.glob(os.path.join(
            QDCBO_DIR, f"{algo}_{setup}_best_so_far_seed*_reps1.csv"))):
        df = pd.read_csv(f)
        last_t = df["time_index"].max()
        out.append(float(
            df[df["time_index"] == last_t]["best_so_far_value"].iloc[-1]))
    return out


def mcbo_finals(env, label):
    return [pd.read_csv(f)["current_optimal"].iloc[-1] for f in
            sorted(glob.glob(os.path.join(
                QMCBO_DIR, f"trial_results_{label}_{env}_*.csv")))]


def main():
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"\textbf{Family} & \textbf{Experiment} & \textbf{Base final} & "
             r"\textbf{Quotient final} & \textbf{Price} & \textbf{Task} \\",
             r"\midrule"]

    for ds, title, task in CBO_ROWS:
        b, q = cbo_finals(ds, "CBO"), cbo_finals(ds, "QCBO")
        if not (b and q):
            print(f"  skip {ds}: {len(b)}/{len(q)} runs")
            continue
        fb, mb = _fmt(b)
        fq, mq = _fmt(q)
        lines.append(rf"\CBO{{}} vs \QCBO{{}} & {title} & {fb} & {fq} & "
                     rf"{_price(task, mb, mq)} & {task} \\")
    lines.append(r"\midrule")
    for setup, task in DCBO_ROWS:
        b, q = dcbo_finals(setup, "dcbo"), dcbo_finals(setup, "qdcbo")
        if not (b and q):
            print(f"  skip {setup}")
            continue
        fb, mb = _fmt(b)
        fq, mq = _fmt(q)
        lines.append(rf"DCBO vs \QDCBO{{}} & {setup} (T=3) & {fb} & {fq} & "
                     rf"{_price(task, mb, mq)} & {task} \\")
    lines.append(r"\midrule")
    for env, task in MCBO_ROWS:
        b, q = mcbo_finals(env, "MCBO"), mcbo_finals(env, "QMCBOJ")
        if not (b and q):
            print(f"  skip {env}: {len(b)}/{len(q)} runs")
            continue
        fb, mb = _fmt(b)
        fq, mq = _fmt(q)
        env_tex = env.replace("_", r"\_")
        lines.append(rf"MCBO vs \QMCBO{{}} & {env_tex} & {fb} & {fq} & "
                     rf"{_price(task, mb, mq)} & {task} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
