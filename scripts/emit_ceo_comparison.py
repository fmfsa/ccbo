"""Aggregate the CEO comparator against the archived CBO/QCBO minimal traces.

For every MinimalBench condition (including the observable-identical aliases
A3<-A0 and B1<-B0), reads the per-seed finals of

  CBO / QCBO   from the archived minimal-suite traces
                 ({scm}_{cond}_{arm}_seed{s}.csv, last best_y), and
  CEO          from results/ceo_minimal (CEO_{scm}_{cond}_seed{s}.csv,
                 last current_optimal; run_ceo_minimal.py protocol),

pairs them by seed, and emits

  paper/tables/ceo_minimal.tex          condition x {CBO, QCBO, CEO} finals,
                                        paired Delta(CEO - QCBO) with a
                                        paired-t 95% CI (min task: positive =
                                        CEO worse), and CEO's mean final
                                        posterior mass on the true graph
  artifacts/summaries/ceo_comparison.json   full per-seed detail

Run:  PYTHONPATH=. python scripts/emit_ceo_comparison.py \
          --minimal-dir <extracted>/minimal [--partial]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONDS = [  # (scm, cond, label, alias_of)
    ("ParallelParent", "A0", "PP A0 (correct)", None),
    ("ParallelParent", "A1", r"PP A1 (del $X_1{\to}Y$)", None),
    ("ParallelParent", "A2", r"PP A2 (add $X_1{\to}X_2$)", None),
    ("ParallelParent", "A3", r"PP A3 (assume $X_1\leftrightarrow Y$)", "A0"),
    ("FrontDoor", "B0", "FD B0 (correct)", None),
    ("FrontDoor", "B1", r"FD B1 (assume $X_1\leftrightarrow M$)", "B0"),
    ("MediatedChain", "C0", "MC C0 (correct)", None),
]
SEEDS = list(range(30))


def finals_minimal(d, scm, cond, arm):
    out = {}
    for s in SEEDS:
        p = d / f"{scm}_{cond}_{arm}_seed{s}.csv"
        if p.exists():
            out[s] = float(pd.read_csv(p)["best_y"].iloc[-1])
    return out


def finals_ceo(d, scm, cond):
    vals, metas = {}, {}
    for s in SEEDS:
        p = d / f"CEO_{scm}_{cond}_seed{s}.csv"
        if p.exists():
            vals[s] = float(pd.read_csv(p)["current_optimal"].iloc[-1])
            mp = d / f"CEO_{scm}_{cond}_seed{s}.meta.json"
            metas[s] = json.load(open(mp)) if mp.exists() else {}
    return vals, metas


def truth_mass(meta):
    pool, post = meta.get("pool_pids"), meta.get("posterior_final")
    true_cond = meta.get("true_cond")
    if not pool or not post or true_cond not in pool:
        return None
    return float(post[pool.index(true_cond)])


def msem(vals):
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return f"${m:.3f}{{\\scriptstyle\\,\\pm\\,}}{se:.3f}$"


def paired_ci(a, b):
    """Delta = a - b on common seeds (min task: positive = a worse)."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    m, sd, n = float(np.mean(d)), float(np.std(d, ddof=1)), len(d)
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return m, m - half, m + half, n, [float(x) for x in d]


def delta_cell(m, lo, hi):
    dp = 2
    while dp < 4 and (round(lo, dp) == round(hi, dp) or round(m, dp) == 0.0):
        dp += 1
    return f"${m:+.{dp}f}$ $[{lo:+.{dp}f},\\,{hi:+.{dp}f}]$"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ceo-dir", type=Path, default=Path("results/ceo_minimal"))
    ap.add_argument("--minimal-dir", type=Path, required=True,
                    help="directory of archived minimal-suite traces")
    ap.add_argument("--tables-dir", type=Path, default=Path("paper/tables"))
    ap.add_argument("--summary", type=Path,
                    default=Path("artifacts/summaries/ceo_comparison.json"))
    ap.add_argument("--partial", action="store_true",
                    help="allow incomplete seed sets (preview only)")
    args = ap.parse_args()

    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"\textbf{Condition} & \textbf{\CBO{}} & \textbf{\QCBO{}} & "
             r"\textbf{\CEO{}} & \textbf{$\Delta$(\CEO{}$-$\QCBO{}) [95\% CI]}"
             r" & \textbf{$P$(truth)} \\", r"\midrule"]
    payload = {}
    for scm, cond, label, alias_of in CONDS:
        cbo = finals_minimal(args.minimal_dir, scm, cond, "CBO")
        qcbo = finals_minimal(args.minimal_dir, scm, cond, "QCBO")
        ceo, metas = finals_ceo(args.ceo_dir, scm, cond)
        common = sorted(set(cbo) & set(qcbo) & set(ceo))
        n = len(common)
        if not args.partial:
            assert n == len(SEEDS), (
                f"{scm} {cond}: only {n}/{len(SEEDS)} complete paired seeds "
                f"(CBO {len(cbo)}, QCBO {len(qcbo)}, CEO {len(ceo)}); "
                "run with --partial to preview")
        if n == 0:
            print(f"  {scm} {cond}: no complete pairs yet, skipped")
            continue
        masses = [m for m in (truth_mass(metas[s]) for s in common)
                  if m is not None]
        dm, lo, hi, np_, d_per = paired_ci(
            {s: ceo[s] for s in common}, {s: qcbo[s] for s in common})
        alias_tag = rf"\;(=\,{alias_of})" if alias_of else ""
        mass_cell = f"${np.mean(masses):.2f}$" if masses else "--"
        lines.append(
            f"{label}{alias_tag} & {msem([cbo[s] for s in common])} & "
            f"{msem([qcbo[s] for s in common])} & "
            f"{msem([ceo[s] for s in common])} & "
            f"{delta_cell(dm, lo, hi)} & {mass_cell} \\\\")
        payload[f"{scm}/{cond}"] = {
            "label": label, "alias_of": alias_of, "n": np_,
            "cbo_finals": {s: cbo[s] for s in common},
            "qcbo_finals": {s: qcbo[s] for s in common},
            "ceo_finals": {s: ceo[s] for s in common},
            "delta_ceo_minus_qcbo": {"mean": dm, "ci95": [lo, hi],
                                     "per_seed": d_per},
            "posterior_mass_on_truth_mean":
                float(np.mean(masses)) if masses else None,
        }
        mass_str = f"  P(truth)={np.mean(masses):.3f}" if masses else ""
        print(f"  {scm} {cond}: n={np_}  "
              f"CEO {np.mean([ceo[s] for s in common]):.4f}  "
              f"D(CEO-QCBO)={dm:+.4f} [{lo:+.4f}, {hi:+.4f}]{mass_str}")
    lines += [r"\bottomrule", r"\end{tabular}"]

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    out = args.tables_dir / "ceo_minimal.tex"
    out.write_text("% Auto-generated by scripts/emit_ceo_comparison.py"
                   " -- do not edit.\n" + "\n".join(lines) + "\n")
    print(f"wrote {out}")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({
        "protocol": "run_ceo_minimal.py (prespecified): same SCMs, "
                    "conditions, seeds 0-29, T=50/60, n_init=3/arm, "
                    "n_obs=100, closed-form E[Y|do] objective; min task, "
                    "positive Delta = CEO worse. A3/B1 are observable-"
                    "identical aliases of A0/B0.",
        "generated_by": "scripts/emit_ceo_comparison.py",
        "conditions": payload,
    }, indent=2) + "\n")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
