"""Emit the QMCBO pilot tables + JSON for the paper.

Reads trial CSVs from results/qmcbo (E1, correct model view) and
results/qmcbo/_e2 (intra-cluster misspecification of the model view) and
writes:

  results/qmcbo_pilot.json      -- stats + trajectories summary
  paper/tables/qmcbo_e1.tex     -- MCBO vs QMCBO: final Y, wall-clock per
                                   unit, per env
  paper/tables/qmcbo_e2.tex     -- invariance: identical-final-value counts
                                   + mean |Dfinal| correct-vs-perturbed
  paper/tables/qmcbo_e2_decisions.tex -- separate final/trajectory/decision counts

Algo labels in filenames: MCBO (stock), QMCBO (joint cluster mechanisms —
the paper's QMCBO); QMCBO-ind files (per-coordinate dev ablation) are
ignored.

Run:  PYTHONPATH=. conda run -n ccbo python scripts/emit_qmcbo_tables.py
"""

import glob
import json
import os
import re
from pathlib import Path

from ccbo.trace_checks import compare_qmcbo_pair, validate_qmcbo_metadata

import numpy as np
import pandas as pd

QDIR = "results/qmcbo"
OUTJSON = "results/qmcbo_pilot.json"
OUTDIR = "paper/tables"

ENVS = [("ToyGraph", 2.172), ("PSAGraph", -4.13)]
ALGOS = ["MCBO", "QMCBO"]
ALABEL = {"MCBO": "MCBO", "QMCBO": r"\QMCBO{}"}


def _collect(d, expected_trials=100, *, perturbed=False):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "trial_results_*.csv"))):
        m = re.match(
            r"trial_results_(MCBO|QMCBO-ind|QMCBO)_([A-Za-z_0-9]+)_(\d+)\.csv",
            os.path.basename(p))
        if not m or m.group(1) == "QMCBO-ind":
            continue
        algo, env, seed = m.group(1), m.group(2), int(m.group(3))
        tr = pd.read_csv(p)["current_optimal"].astype(float).tolist()
        info_p = p.replace(".csv", "_info.json")
        secs = json.load(open(info_p)).get("secs") if os.path.exists(info_p) \
            else None
        sidecar = Path(p.replace(".csv", ".decisions.json"))
        if not sidecar.exists():
            raise ValueError(f"missing decision evidence: {sidecar}")
        log = json.loads(sidecar.read_text())
        try:
            validate_qmcbo_metadata(log, env, algo, seed, expected_trials,
                                    perturbed=perturbed)
        except ValueError as exc:
            raise ValueError(f"{sidecar}: {exc}") from exc
        if len(tr) != expected_trials:
            raise ValueError(f"{p}: expected {expected_trials} trials, got {len(tr)}")
        out.setdefault((env, algo), {})[seed] = {"traj": tr, "secs": secs, "log": log}
    return out


def paired_identity(base, perturbed, expected_seeds, expected_trials):
    expected = set(range(expected_seeds))
    if set(base) != expected or set(perturbed) != expected:
        raise ValueError(f"expected paired seeds {sorted(expected)}; "
                         f"base={sorted(base)}, perturbed={sorted(perturbed)}")
    pairs = [compare_qmcbo_pair(base[s]["traj"], perturbed[s]["traj"],
                               base[s]["log"], perturbed[s]["log"], expected_trials)
             for s in sorted(expected)]
    return {
        "n": len(pairs),
        **{key: sum(p[key] for p in pairs) for key in
           ("identical_final", "identical_incumbent_trajectory", "identical_decisions_and_scores")},
        "mean_abs_dfinal": float(np.mean([p["abs_final_deviation"] for p in pairs])),
        **{key: max(p[key] for p in pairs) for key in
           ("max_abs_incumbent_deviation", "max_abs_intervention_deviation", "max_abs_score_deviation")},
    }


def _mp(mean, sem, fmt="%.3f"):
    return (fmt % mean) + r"{\scriptstyle\,\pm\,}" + (fmt % sem)


def main():
    import argparse
    global QDIR, OUTJSON, OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results",
                    help="root holding qmcbo/ (and qmcbo/_e2/)")
    ap.add_argument("--summary", default=None,
                    help="override the JSON summary path")
    ap.add_argument("--tables-dir", default="paper/tables")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    if args.seeds < 1 or args.trials < 1:
        ap.error("seeds and trials must be positive")
    OUTDIR = args.tables_dir
    QDIR = os.path.join(args.results_dir, "qmcbo")
    OUTJSON = args.summary or os.path.join(args.results_dir,
                                           "qmcbo_pilot.json")
    e1 = _collect(QDIR, args.trials)
    e2 = _collect(os.path.join(QDIR, "_e2"), args.trials, perturbed=True)
    expected = {(env, algo) for env, _ in ENVS for algo in ALGOS}
    if set(e1) != expected or set(e2) != expected:
        raise ValueError("incomplete or unexpected model-based experiment grid")

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

    identity = {key: paired_identity(e1[key], e2[key], args.seeds, args.trials)
                for key in sorted(expected)}

    # ---- JSON --------------------------------------------------------------
    Path(OUTJSON).parent.mkdir(parents=True, exist_ok=True)
    js = {"schema": 2, "equality": "exact numeric equality; final, incumbent, and X/score separated", "e1": {f"{e}|{a}": v for (e, a), v in stats.items()},
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
             r"\textbf{same final} & $\overline{|\Delta \text{final}|}$ \\",
             r"\midrule"]
    for env, _ in ENVS:
        first = True
        for algo in ["MCBO", "QMCBO"]:
            c = identity.get((env, algo))
            if c is None:
                continue
            envcell = env.replace("_", r"\_") if first else ""
            first = False
            d = (f"{c['mean_abs_dfinal']:.3f}"
                 if c["mean_abs_dfinal"] is not None else "--")
            lines.append(
                rf"{envcell} & {ALABEL[algo]} & "
                rf"{c['identical_final']}/{c['n']} & {d} \\")
        lines.append(r"\addlinespace[2pt]")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
    p = os.path.join(OUTDIR, "qmcbo_e2.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {p}")

    decision_lines = [r"\begin{tabular}{llrrr}", r"\toprule",
                      r"Env & Method & Final & Incumbents & Decisions/scores \\", r"\midrule"]
    for env, _ in ENVS:
        for algo in ALGOS:
            c = identity[(env, algo)]
            cells = [env, ALABEL[algo]] + [f"{c[k]}/{c['n']}" for k in
                     ("identical_final", "identical_incumbent_trajectory", "identical_decisions_and_scores")]
            decision_lines.append(" & ".join(cells) + r" \\")
    decision_lines += [r"\bottomrule", r"\end{tabular}"]
    Path(OUTDIR, "qmcbo_e2_decisions.tex").write_text("\n".join(decision_lines) + "\n")


if __name__ == "__main__":
    main()
