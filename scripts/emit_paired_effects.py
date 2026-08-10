"""Emit per-task PAIRED final-performance effects for the family suite.

Successor to ``emit_family_price_table.py``. The old table reported separate
mean±s.e. finals plus a signed "Price" column; this emitter reports, per
task, the *paired* per-seed difference

    d_i = quotient_i - base_i   (min tasks)
    d_i = base_i - quotient_i   (max tasks)

so that **positive = the quotient ended worse in the task's own direction**,
with a paired-t 95% confidence interval. These are finite-budget empirical
effects — they do NOT estimate the theoretical coarsening price
Delta(Pi) = V(Pi) - V(Pi_bot), and may legitimately be negative.

Outputs
-------
paper/tables/family_effects.tex        main-text table (Table 2)
paper/tables/qmcbo_paired.tex          per-seed differences for the n=5 MCBO
                                       family (descriptive; App. F)
artifacts/summaries/paired_effects.json  full per-seed detail driving §5.4 prose

Run:  PYTHONPATH=. python scripts/emit_paired_effects.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CBO_ROWS = [("ToyGraph", "Toy"), ("CompleteGraph", "Synthetic"),
            ("SimplifiedCoralGraph", "Coral")]
DCBO_ROWS = ["stat", "ind", "nonstat"]
MCBO_ROWS = ["ToyGraph", "PSAGraph"]

SEED_RES = (re.compile(r"seed(\d+)"), re.compile(r"_(\d+)\.csv$"))


def _by_seed(paths: list[Path], final_fn) -> dict[int, float]:
    out = {}
    for p in paths:
        m = next((m for rx in SEED_RES if (m := rx.search(p.name))), None)
        if not m:
            raise ValueError(f"cannot parse seed from {p.name}")
        seed = int(m.group(1))
        if seed in out:
            raise ValueError(f"duplicate seed {seed} at {p.name}")
        out[seed] = final_fn(p)
    return out


def _final_best_y(p: Path) -> float:
    return float(pd.read_csv(p)["best_y"].iloc[-1])


def _final_dcbo(p: Path) -> float:
    df = pd.read_csv(p)
    last_t = df["time_index"].max()
    return float(df[df["time_index"] == last_t]["best_so_far_value"].iloc[-1])


def _final_mcbo(p: Path) -> float:
    return float(pd.read_csv(p)["current_optimal"].iloc[-1])


def paired(base: dict[int, float], quot: dict[int, float], task: str) -> dict:
    if set(base) != set(quot):
        raise ValueError(f"seed sets differ: base {sorted(base)} vs "
                         f"quotient {sorted(quot)}")
    seeds = sorted(base)
    b = np.array([base[s] for s in seeds])
    q = np.array([quot[s] for s in seeds])
    d = (q - b) if task == "min" else (b - q)
    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {
        "task": task, "n": n, "seeds": seeds,
        "base_mean": float(np.mean(b)), "base_sem": float(stats.sem(b)),
        "quotient_mean": float(np.mean(q)), "quotient_sem": float(stats.sem(q)),
        "d_mean": mean, "d_sd": sd, "ci95": [mean - half, mean + half],
        "d_per_seed": [float(x) for x in d],
        "n_worse": int(np.sum(d > 0)), "n_better": int(np.sum(d < 0)),
        "n_tie": int(np.sum(d == 0)),
    }


def _msem(mean: float, sem: float) -> str:
    return f"${mean:.2f}{{\\scriptstyle\\,\\pm\\,}}{sem:.2f}$"


def _delta_cell(r: dict) -> str:
    """Adaptive-precision Δ [95% CI] cell: never round a real effect to 0.00."""
    lo, hi = r["ci95"]
    if all(x == 0.0 for x in r["d_per_seed"]):
        return r"$0$ (exact)"
    dp = 2
    while dp < 4 and (round(lo, dp) == round(hi, dp)
                      or round(r["d_mean"], dp) == 0.0):
        dp += 1
    return (f"${r['d_mean']:+.{dp}f}$ "
            f"$[{lo:+.{dp}f},\\,{hi:+.{dp}f}]$")


def _row(label: str, r: dict) -> str:
    return (f"{label} & {_msem(r['base_mean'], r['base_sem'])} & "
            f"{_msem(r['quotient_mean'], r['quotient_sem'])} & "
            f"{_delta_cell(r)} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--tables-dir", type=Path, default=Path("paper/tables"))
    ap.add_argument("--summary", type=Path,
                    default=Path("artifacts/summaries/paired_effects.json"))
    args = ap.parse_args()
    root = args.results_dir

    results: dict[str, dict] = {}

    def collect(key, base_glob, quot_glob, final_fn, task, subdir):
        base = _by_seed(sorted((root / subdir).glob(base_glob)), final_fn)
        quot = _by_seed(sorted((root / subdir).glob(quot_glob)), final_fn)
        results[key] = paired(base, quot, task)

    for ds, title in CBO_ROWS:
        collect(f"cbo/{title}", f"{ds}_CBO_seed*.csv", f"{ds}_QCBO_seed*.csv",
                _final_best_y, "min", "family_cbo")
    for setup in DCBO_ROWS:
        collect(f"dcbo/{setup}",
                f"dcbo_{setup}_best_so_far_seed*_reps1.csv",
                f"qdcbo_{setup}_best_so_far_seed*_reps1.csv",
                _final_dcbo, "min", "qdcbo")
    for env in MCBO_ROWS:
        collect(f"mcbo/{env}",
                f"trial_results_MCBO_{env}_*.csv",
                f"trial_results_QMCBO_{env}_*.csv",
                _final_mcbo, "max", "qmcbo")

    # ---- main table -------------------------------------------------------
    def header(text):
        return rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{text}}}}} \\"

    n_cbo = results["cbo/Toy"]["n"]
    n_dcbo = results["dcbo/stat"]["n"]
    n_mcbo = results["mcbo/ToyGraph"]["n"]
    lines = [
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"\textbf{Experiment} & \textbf{Base final} & \textbf{Quotient final}"
        r" & \textbf{Paired effect $\Delta$ [95\% CI]} \\",
        r"\midrule",
        header(rf"\CBO{{}} vs \QCBO{{}} (min; $n{{=}}{n_cbo}$ paired seeds)"),
    ]
    for ds, title in CBO_ROWS:
        lines.append(_row(title, results[f"cbo/{title}"]))
    lines += [r"\addlinespace[2pt]",
              header(rf"DCBO vs \QDCBO{{}} (min; $n{{=}}{n_dcbo}$ paired seeds)")]
    for setup in DCBO_ROWS:
        lines.append(_row(f"{setup} (T=3)", results[f"dcbo/{setup}"]))
    lines += [r"\addlinespace[2pt]",
              header(rf"MCBO vs \QMCBO{{}} (max; $n{{=}}{n_mcbo}$ paired seeds; "
                     r"descriptive)")]
    for env in MCBO_ROWS:
        lines.append(_row(env, results[f"mcbo/{env}"]))
    lines += [r"\bottomrule", r"\end{tabular}"]

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    out_main = args.tables_dir / "family_effects.tex"
    out_main.write_text("% Auto-generated by scripts/emit_paired_effects.py"
                        " -- do not edit.\n" + "\n".join(lines) + "\n")
    print(f"wrote {out_main}")

    # ---- per-seed MCBO table (n=5: show the raw paired values) ------------
    m_lines = [r"\begin{tabular}{@{}l" + "c" * n_mcbo + r"@{}}", r"\toprule",
               r"\textbf{Env} & " +
               " & ".join(f"seed {s}" for s in results["mcbo/ToyGraph"]["seeds"])
               + r" \\", r"\midrule"]
    for env in MCBO_ROWS:
        r = results[f"mcbo/{env}"]
        cells = " & ".join(f"${d:+.3g}$" for d in r["d_per_seed"])
        m_lines.append(rf"{env} & {cells} \\")
    m_lines += [r"\bottomrule", r"\end{tabular}"]
    out_m = args.tables_dir / "qmcbo_paired.tex"
    out_m.write_text("% Auto-generated by scripts/emit_paired_effects.py"
                     " -- do not edit.\n" + "\n".join(m_lines) + "\n")
    print(f"wrote {out_m}")

    # ---- JSON summary ------------------------------------------------------
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "convention": "d_i = quotient_i - base_i for min tasks, "
                      "base_i - quotient_i for max tasks; positive = quotient "
                      "worse in the task's own direction. Paired-t 95% CI. "
                      "Finite-budget empirical effects, not estimates of "
                      "Delta(Pi).",
        "generated_by": "scripts/emit_paired_effects.py",
        "results": results,
    }
    args.summary.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.summary}")

    for key, r in results.items():
        lo, hi = r["ci95"]
        print(f"  {key:16s} n={r['n']:2d}  Δ={r['d_mean']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  worse/tie/better="
              f"{r['n_worse']}/{r['n_tie']}/{r['n_better']}")


if __name__ == "__main__":
    main()
