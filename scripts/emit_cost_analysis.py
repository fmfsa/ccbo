"""Initialization-inclusive cost accounting from the raw traces (engine v3).

Engine v3 traces already charge the initial interventional design on row 0
(``cum_cost = init_cost``) and, for the refinement arms, the split design at
the trigger row; the per-unit ``.decisions.json`` sidecar records the
exploration set and those costs.  This emitter therefore reads, not
reconstructs, the accounting for MinimalBench and the CBO family.  The CEO
comparator archive is unchanged (row 0 at logged cost 0), so its init cost is
still added from its ``.meta.json`` as before.

Per (experiment, method):

  |ES|          number of arms in the exploration set (refined arms listed too)
  init cost     initial design cost (+ split design for HQCBO)
  logged cost   mean optimizer-executed cost over seeds (total - init - split)
  total cost    mean final cum_cost over seeds
  Y @ C*        mean best-so-far Y when every method is stopped at the SAME
                initialization-inclusive budget C* = the smallest final cost
                across all methods and seeds in the group (step interpolation)

Outputs paper/tables/cost_accounting.tex and
artifacts/summaries/cost_accounting.json.

Run:  PYTHONPATH=. python scripts/emit_cost_analysis.py --results-dir results/v3 \
          [--ceo-dir results/v3/ceo_minimal]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.publication_methods import publication_name, required_files
from ccbo.decision_log import read as read_log, sidecar_path

warnings.filterwarnings("ignore")

MINIMAL_ARMS = ["BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP"]
MINIMAL_GROUPS = [
    ("ParallelParent A0 (correct)", "ParallelParent", "A0", MINIMAL_ARMS),
    ("ParallelParent A1 (del $X_1{\\to}Y$)", "ParallelParent", "A1", MINIMAL_ARMS),
    ("FrontDoor B0 (correct)", "FrontDoor", "B0", MINIMAL_ARMS),
    ("MediatedChain C0", "MediatedChain", "C0",
     MINIMAL_ARMS + ["HQCBOGF"]),
]
FAMILY_ROWS = [("ToyGraph", "Toy"), ("CompleteGraph", "Synthetic"),
               ("SimplifiedCoralGraph", "Coral")]
FAMILY_ARMS = ["BO", "CBO", "QCBO"]

LABEL = {"HQCBOGF": "HQCBO"}


def _label(arm: str) -> str:
    return publication_name(arm, latex=True)


def _assert_arms(df: pd.DataFrame, declared: set[str]) -> None:
    seen = {a for a in df["arm"].dropna() if a and a != "init"}
    unknown = {s for s in seen if "+".join(sorted(s.split("+"))) not in declared}
    if unknown:
        raise AssertionError(f"trace arms {unknown} not in declared ES {declared}")


def value_at_budget(cost: np.ndarray, y: np.ndarray, budget: float):
    """Best-so-far Y at the last point with total cost <= budget (step fn)."""
    if budget > cost[-1] + 1e-9:
        return None
    idx = np.where(cost <= budget + 1e-9)[0]
    return float(y[idx[-1]]) if len(idx) else None


def summarize(rows_per_seed: list[tuple[np.ndarray, np.ndarray]], budget: float):
    vals = [value_at_budget(c, y, budget) for c, y in rows_per_seed]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def validate_trace_costs(df, log, source="trace"):
    """Reconcile every cost increment against actions and split sidecar events."""
    total = df.cum_cost.to_numpy(float)
    if not np.all(np.isfinite(total)) or np.any(np.diff(total) < 0):
        raise ValueError(f"{source}: nonfinite or decreasing cost")
    rows = log["rows"]
    if len(rows) != len(df) or not np.allclose(total, [r["cum_cost"] for r in rows], rtol=0, atol=1e-9):
        raise ValueError(f"{source}: CSV/sidecar cost mismatch")
    refine = log.get("refine") or {}
    split = float(refine.get("split_init_cost", 0) or 0)
    expected = []
    for row in rows[1:]:
        arm = row.get("arm") or ""
        cost = len(arm.split("+")) if arm and arm != "init" else 0
        if row["trial"] == refine.get("trigger_step"):
            cost += split
        expected.append(cost)
    if not np.allclose(np.diff(total), expected, rtol=0, atol=1e-9):
        raise ValueError(f"{source}: costs do not reconcile with interventions and split initialization")


def traced_method(files: list[Path]) -> dict:
    """Accounting for one (experiment, arm) from engine-v3 traces + sidecars."""
    per_seed, finals, logged, inits, splits = [], [], [], [], []
    es_all: list[str] = []
    for f in files:
        df = pd.read_csv(f)
        log = read_log(sidecar_path(str(f)))
        es = ["+".join(a) for a in log["init"]["es"]]
        refined = ["+".join(a) for a in (log.get("refine") or {}).get("refined_es", [])]
        for a in es + refined:
            if a not in es_all:
                es_all.append(a)
        ic = float(log["init"]["init_cost"])
        sc = float((log.get("refine") or {}).get("split_init_cost", 0.0) or 0.0)
        declared = {"+".join(sorted(a.split("+"))) for a in es + refined}
        _assert_arms(df, declared)
        total = df["cum_cost"].values.astype(float)
        assert abs(total[0] - ic) < 1e-9, (f, total[0], ic)
        validate_trace_costs(df, log, f)
        per_seed.append((total, df["best_y"].values.astype(float)))
        finals.append(float(df["best_y"].iloc[-1]))
        inits.append(ic)
        splits.append(sc)
        logged.append(float(total[-1] - ic - sc))
    assert per_seed, f"no traces for {files}"
    return {
        "es": es_all, "n_arms": len(es_all),
        "init_cost": float(np.mean(inits)),
        "split_init_cost": float(np.mean(splits)),
        "logged_cost_mean": float(np.mean(logged)),
        "total_cost_mean": float(np.mean([c[-1] for c, _ in per_seed])),
        "final_y_mean": float(np.mean(finals)),
        "final_y_sem": float(np.std(finals, ddof=1) / np.sqrt(len(finals))),
        "n_seeds": len(finals),
        "_per_seed": per_seed,
    }


def ceo_method(ceo_dir: Path, scm: str, cond: str) -> dict:
    """CEO rows from run_ceo_minimal outputs (archive unchanged: trajectory
    rows start at logged cost 0, init cost and per-trial costs in .meta.json)."""
    per_seed, finals, logged = [], [], []
    ic = n_arms = None
    es = None
    for f in sorted(ceo_dir.glob(f"CEO_{scm}_{cond}_seed*.csv")):
        meta = json.loads(
            f.with_name(f.name.replace(".csv", ".meta.json")).read_text())
        ic, es = meta["init_cost"], meta["es"]
        n_arms = len(es)
        ptc = np.asarray(meta["per_trial_cost"], dtype=float)
        df = pd.read_csv(f)
        assert len(df) == len(ptc), (f, len(df), len(ptc))
        if len(ptc) == 0 or ptc[0] != 0 or np.any(ptc < 0):
            raise ValueError(f"{f}: invalid legacy CEO cost schedule")
        total = np.cumsum(ptc) + ic
        per_seed.append((total, df["current_optimal"].values.astype(float)))
        finals.append(float(df["current_optimal"].iloc[-1]))
        logged.append(float(ptc.sum()))
    assert per_seed, f"no CEO units for {scm} {cond} in {ceo_dir}"
    return {
        "es": ["+".join(a) for a in es], "n_arms": n_arms,
        "init_cost": float(ic), "split_init_cost": 0.0,
        "logged_cost_mean": float(np.mean(logged)),
        "total_cost_mean": float(np.mean([c[-1] for c, _ in per_seed])),
        "final_y_mean": float(np.mean(finals)),
        "final_y_sem": float(np.std(finals, ddof=1) / np.sqrt(len(finals))),
        "n_seeds": len(finals),
        "_per_seed": per_seed,
    }


def emit_group(name: str, methods: dict[str, dict]) -> tuple[list[str], dict]:
    cstar = min(c[-1] for m in methods.values() for c, _ in m["_per_seed"])
    lines = [rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{name}"
             rf" ($C^\ast{{=}}{cstar:.0f}$)}}}} \\"]
    payload = {"C_star": cstar, "methods": {},
               "budget_determining_methods": [a for a,m in methods.items()
                   if any(abs(c[-1]-cstar)<1e-9 for c,_ in m["_per_seed"])],
               "initialization_exceeds_budget": [a for a,m in methods.items()
                   if any(c[0]>cstar+1e-9 for c,_ in m["_per_seed"])]}
    for label, m in methods.items():
        at = summarize(m["_per_seed"], cstar)
        at_cell = f"${at[0]:.3f}{{\\scriptstyle\\,\\pm\\,}}{at[1]:.3f}$" \
            if at else "n/a"
        init_cell = (f"{m['init_cost']:.0f}$+${m['split_init_cost']:.0f}"
                     if m["split_init_cost"] else f"{m['init_cost']:.0f}")
        lines.append(
            rf"{_label(label)} & {m['n_arms']} & {init_cell} & "
            rf"{m['logged_cost_mean']:.0f} & {m['total_cost_mean']:.0f} & "
            rf"${m['final_y_mean']:.3f}{{\scriptstyle\,\pm\,}}"
            rf"{m['final_y_sem']:.3f}$ & {at_cell} \\")
        payload["methods"][label] = {k: v for k, v in m.items()
                                     if not k.startswith("_")}
        payload["methods"][label]["y_at_c_star"] = at
        payload["methods"][label]["archive_key"] = label
        payload["methods"][label]["publication_name"] = publication_name(label)
    return lines, payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/v3"))
    ap.add_argument("--tables-dir", type=Path, default=Path("paper/tables"))
    ap.add_argument("--summary", type=Path,
                    default=Path("artifacts/summaries/cost_accounting.json"))
    ap.add_argument("--ceo-dir", type=Path, default=None,
                    help="results/v3/ceo_minimal-style dir; adds CEO rows to "
                         "the MinimalBench groups")
    ap.add_argument("--skip-family", action="store_true")
    args = ap.parse_args()
    root = args.results_dir

    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r"\textbf{Method} & $|\mathrm{ES}|$ & \textbf{Init} & "
        r"\textbf{Logged} & \textbf{Total} & \textbf{Final $Y$} & "
        r"\textbf{$Y$ at $C^\ast$} \\", r"\midrule"]
    payload = {}

    for name, scm, cond, methods in MINIMAL_GROUPS:
        block = {}
        for m in methods:
            files = required_files(root / "minimal", f"{scm}_{cond}_{m}", 30)
            if files:
                block[m] = traced_method(files)
        if args.ceo_dir:
            required_files(args.ceo_dir, f"CEO_{scm}_{cond}", 30)
            block["CEO"] = ceo_method(args.ceo_dir, scm, cond)
        if not block:
            continue
        glines, gpayload = emit_group(name, block)
        lines += glines + [r"\addlinespace[2pt]"]
        payload[f"{scm} {cond}"] = gpayload
        print(f"  {scm} {cond}: " + ", ".join(
            f"{m} init={b['init_cost']:.0f}+{b['split_init_cost']:.0f} "
            f"total={b['total_cost_mean']:.0f}" for m, b in block.items()))

    if not args.skip_family:
        for dataset, title in FAMILY_ROWS:
            block = {}
            for arm in FAMILY_ARMS:
                files = required_files(root / "family_cbo", f"{dataset}_{arm}", 10)
                if files:
                    block[arm] = traced_method(files)
            if not block:
                continue
            glines, gpayload = emit_group(f"{title} (family suite)", block)
            lines += glines + [r"\addlinespace[2pt]"]
            payload[f"family {title}"] = gpayload
            print(f"  family {title}: " + ", ".join(
                f"{m} |ES|={b['n_arms']} init={b['init_cost']:.0f}"
                for m, b in block.items()))

    if lines[-1] == r"\addlinespace[2pt]":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    out = args.tables_dir / "cost_accounting.tex"
    out.write_text("% Auto-generated by scripts/emit_cost_analysis.py"
                   " -- do not edit.\n" + "\n".join(lines) + "\n")
    print(f"wrote {out}")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({
        "convention": "Per-variable unit costs. Engine-v3 traces charge the "
                      "initial design on row 0 (n_init points per arm) and the "
                      "refinement arms' split design at the trigger row; "
                      "'logged' = total - init - split. CEO rows (unchanged "
                      "archive) add init and per-trial costs from .meta.json. "
                      "Y at C* uses step interpolation at the smallest final "
                      "cost across all methods and seeds in the group. "
                      "No values are extrapolated beyond recorded runs.",
        "generated_by": "scripts/emit_cost_analysis.py",
        "groups": payload,
    }, indent=2, default=float) + "\n")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
