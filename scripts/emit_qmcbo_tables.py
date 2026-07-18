"""Emit the QMCBO pilot tables + JSON for the paper.

Reads trial CSVs from third_party/.../results/_qmcbo (E1, correct model view)
and _qmcbo/_e2 (intra-cluster misspecification of the model view) and writes:

  results/qmcbo_pilot.json      -- stats + trajectories summary
  paper/tables/qmcbo_e1.tex     -- MCBO vs QMCBO-joint (vs -ind ablation):
                                   final Y, wall-clock per unit, per env
  paper/tables/qmcbo_e2.tex     -- invariance: byte-identical trajectory
                                   counts + mean |Dfinal| correct-vs-perturbed

Algo labels in filenames: MCBO (stock), QMCBO (v1 per-coordinate = ind
ablation), QMCBOJ (joint cluster mechanisms — the paper's QMCBO).

Run:  PYTHONPATH=. conda run -n ccbo python scripts/emit_qmcbo_tables.py
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

QDIR = "third_party/CausalBO_Benchmark/results/_qmcbo"
OUTJSON = "results/qmcbo_pilot.json"
OUTDIR = "paper/tables"

ENVS = [("ToyGraph", 2.172), ("Synthetic_2", 2.152), ("PSAGraph", -4.13)]
ALGOS = ["MCBO", "QMCBOJ", "QMCBO"]
ALABEL = {"MCBO": "MCBO", "QMCBOJ": r"\QMCBO{}",
          "QMCBO": r"\QMCBO{}-ind {\footnotesize(ablation)}"}


def _collect(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "trial_results_*.csv"))):
        m = re.match(r"trial_results_(MCBO|QMCBOJ|QMCBO)_([A-Za-z_0-9]+)_(\d+)\.csv",
                     os.path.basename(p))
        if not m:
            continue
        algo, env, seed = m.group(1), m.group(2), int(m.group(3))
        tr = pd.read_csv(p)["current_optimal"].astype(float).tolist()
        info_p = p.replace(".csv", "_info.json")
        secs = json.load(open(info_p)).get("secs") if os.path.exists(info_p) \
            else None
        out.setdefault((env, algo), {})[seed] = {"traj": tr, "secs": secs}
    return out


def _mp(mean, sem, fmt="%.3f"):
    return (fmt % mean) + r"{\scriptstyle\,\pm\,}" + (fmt % sem)


def main():
    e1 = _collect(QDIR)
    e2 = _collect(os.path.join(QDIR, "_e2"))

    stats = {}
    for (env, algo), seeds in e1.items():
        finals = [v["traj"][-1] for v in seeds.values()]
        secs = [v["secs"] for v in seeds.values() if v["secs"]]
        stats[(env, algo)] = {
            "final_mean": float(np.mean(finals)),
            "final_sem": float(np.std(finals, ddof=1) / np.sqrt(len(finals)))
            if len(finals) > 1 else 0.0,
            "secs": float(np.mean(secs)) if secs else None,
            "n": len(finals)}

    identity = {}
    for (env, algo), seeds in e2.items():
        base = e1.get((env, algo), {})
        n_id, deltas = 0, []
        for s, v in seeds.items():
            if s not in base:
                continue
            a, b = base[s]["traj"], v["traj"]
            if len(a) == len(b) and all(x == y for x, y in zip(a, b)):
                n_id += 1
            deltas.append(abs(a[-1] - b[-1]))
        identity[(env, algo)] = {"identical": n_id, "n": len(deltas),
                                 "mean_abs_dfinal": float(np.mean(deltas))
                                 if deltas else None}

    # ---- JSON --------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    js = {"e1": {f"{e}|{a}": v for (e, a), v in stats.items()},
          "e2_identity": {f"{e}|{a}": v for (e, a), v in identity.items()}}
    with open(OUTJSON, "w") as f:
        json.dump(js, f, indent=2)
    print(f"wrote {OUTJSON}")

    # ---- E1 table ----------------------------------------------------------
    os.makedirs(OUTDIR, exist_ok=True)
    lines = [r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"\textbf{Env} & \textbf{Method} & \textbf{Final $Y$} & "
             r"\textbf{s/unit} \\", r"\midrule"]
    for env, opt in ENVS:
        first = True
        env_tex = env.replace("_", r"\_")
        for algo in ALGOS:
            c = stats.get((env, algo))
            if c is None:
                continue
            envcell = (env_tex + r" {\footnotesize($y^\star{=}" + str(opt)
                       + r"$)}") if first else ""
            first = False
            secs = f"{c['secs']:.0f}" if c["secs"] else "--"
            lines.append(
                rf"{envcell} & {ALABEL[algo]} & "
                rf"${_mp(c['final_mean'], c['final_sem'])}$ & {secs} \\")
        lines.append(r"\addlinespace[2pt]")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
    p = os.path.join(OUTDIR, "qmcbo_e1.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {p}")

    # ---- E2 table ----------------------------------------------------------
    lines = [r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"\textbf{Env} & \textbf{Method} & "
             r"\textbf{byte-identical} & $\overline{|\Delta \text{final}|}$ \\",
             r"\midrule"]
    for env, _ in ENVS:
        first = True
        for algo in ["MCBO", "QMCBOJ"]:
            c = identity.get((env, algo))
            if c is None:
                continue
            envcell = env.replace("_", r"\_") if first else ""
            first = False
            d = (f"{c['mean_abs_dfinal']:.3f}"
                 if c["mean_abs_dfinal"] is not None else "--")
            lines.append(
                rf"{envcell} & {ALABEL[algo]} & "
                rf"{c['identical']}/{c['n']} & {d} \\")
        lines.append(r"\addlinespace[2pt]")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
    p = os.path.join(OUTDIR, "qmcbo_e2.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
