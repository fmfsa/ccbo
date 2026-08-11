"""Initialization-inclusive cost accounting from the archived raw traces.

The runners log ``cum_cost`` starting at 0 *after* the initial interventional
design (3 points per arm on MinimalBench, 10 on the CBO family; per-variable
unit costs), so methods with different arm counts consume different unlogged
initialization budgets at equal T. This emitter reconstructs, per
(experiment, method):

  |ES|          number of arms in the exploration set
  init cost     n_init * sum_{s in ES} |s|   (per-variable unit costs)
  logged cost   mean final cum_cost over seeds
  total cost    init + logged (HQCBO adds 3 points per newly exposed arm at
                its per-seed split trial, from refine_info.json)
  Y @ C*        mean final-so-far Y when every method is stopped at the SAME
                initialization-inclusive budget C* = the smallest total cost
                across the group's methods

The trajectory *through* initialization is unlogged and is NOT reconstructed:
each method's curve starts at its post-initialization incumbent placed at its
full initialization cost; budgets below that are reported as unavailable.

Outputs paper/tables/cost_accounting.tex and
artifacts/summaries/cost_accounting.json.

Run:  PYTHONPATH=. python scripts/emit_cost_analysis.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N_INIT_MINIMAL = 3
N_INIT_FAMILY = 10

# MinimalBench exploration sets (documented in ccbo/minibench.py and asserted
# below against the arm labels actually recorded in the traces).
MINIMAL_ES = {
    ("ParallelParent", "A0", "CBO"): [["X1"], ["X2"], ["X1", "X2"]],
    ("ParallelParent", "A0", "QCBO"): [["X1", "X2"]],
    ("ParallelParent", "A1", "CBO"): [["X2"]],
    ("ParallelParent", "A1", "QCBO"): [["X1", "X2"]],
    ("FrontDoor", "B0", "CBO"): [["X1"], ["M"]],
    ("FrontDoor", "B0", "QCBO"): [["X1", "M"]],
    ("MediatedChain", "C0", "CBO"): [["X1"], ["X2"]],
    ("MediatedChain", "C0", "QCBO"): [["X1", "X2"]],
    ("MediatedChain", "C0", "HQCBO"): [["X1", "X2"]],   # coarse phase
}
HQCBO_SPLIT_ARMS = [["X1"], ["X2"]]                     # exposed at the split

MINIMAL_GROUPS = [
    ("ParallelParent A0 (correct)", "ParallelParent", "A0",
     ["CBO", "QCBO"]),
    ("ParallelParent A1 (del $X_1{\\to}Y$)", "ParallelParent", "A1",
     ["CBO", "QCBO"]),
    ("FrontDoor B0 (correct)", "FrontDoor", "B0", ["CBO", "QCBO"]),
    ("MediatedChain C0", "MediatedChain", "C0", ["CBO", "QCBO", "HQCBO"]),
]

FAMILY_ROWS = [("ToyGraph", "Toy"), ("CompleteGraph", "Synthetic"),
               ("SimplifiedCoralGraph", "Coral")]

SEED_RE = re.compile(r"seed(\d+)\.csv$")


def arm_cost(arm: list[str]) -> int:
    return len(arm)


def init_cost(es: list[list[str]], n_init: int) -> int:
    return n_init * sum(arm_cost(a) for a in es)


def _minimal_files(root: Path, scm: str, cond: str, method: str):
    return sorted((root / "minimal").glob(f"{scm}_{cond}_{method}_seed*.csv"))


def _assert_arms(df: pd.DataFrame, es: list[list[str]], extra=()) -> None:
    declared = {"+".join(sorted(a)) for a in es} | {"+".join(sorted(a)) for a in extra}
    seen = {a for a in df["arm"].dropna() if a and a != "init"}
    unknown = {s for s in seen if "+".join(sorted(s.split("+"))) not in declared}
    if unknown:
        raise AssertionError(f"trace arms {unknown} not in declared ES {declared}")


def value_at_budget(cost: np.ndarray, y: np.ndarray, budget: float):
    """Best-so-far Y at the last point with total cost <= budget (step fn)."""
    idx = np.where(cost <= budget + 1e-9)[0]
    return float(y[idx[-1]]) if len(idx) else None


def summarize(rows_per_seed: list[tuple[np.ndarray, np.ndarray]], budget: float):
    vals = [value_at_budget(c, y, budget) for c, y in rows_per_seed]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def minimal_method(root: Path, scm: str, cond: str, method: str,
                   trigger: dict[str, int] | None) -> dict:
    es = MINIMAL_ES[(scm, cond, method)]
    ic = init_cost(es, N_INIT_MINIMAL)
    split_ic = init_cost(HQCBO_SPLIT_ARMS, N_INIT_MINIMAL) if method == "HQCBO" else 0
    per_seed, finals, logged = [], [], []
    for f in _minimal_files(root, scm, cond, method):
        df = pd.read_csv(f)
        _assert_arms(df, es, HQCBO_SPLIT_ARMS if method == "HQCBO" else ())
        total = df["cum_cost"].values.astype(float) + ic
        if method == "HQCBO":
            seed = int(SEED_RE.search(f.name).group(1))
            t_split = trigger[f"{scm}_{cond}_seed{seed}"]
            total = total + np.where(df["trial"].values >= t_split, split_ic, 0)
        per_seed.append((total, df["best_y"].values.astype(float)))
        finals.append(float(df["best_y"].iloc[-1]))
        logged.append(float(df["cum_cost"].iloc[-1]))
    return {
        "es": ["+".join(a) for a in es], "n_arms": len(es),
        "init_cost": ic, "split_init_cost": split_ic,
        "logged_cost_mean": float(np.mean(logged)),
        "total_cost_mean": float(np.mean([c[-1] for c, _ in per_seed])),
        "final_y_mean": float(np.mean(finals)),
        "final_y_sem": float(np.std(finals, ddof=1) / np.sqrt(len(finals))),
        "_per_seed": per_seed,
    }


def ceo_method(ceo_dir: Path, scm: str, cond: str) -> dict:
    """CEO rows from run_ceo_minimal outputs: trajectory (current_optimal,
    T+1 rows, row 0 = initial incumbent at logged cost 0) + per-trial costs
    and init cost from the .meta.json (per-variable unit costs; CEO's
    acquisition divides by the same per-variable cost, cost_type=1)."""
    import json as _json
    per_seed, finals, logged = [], [], []
    ic = n_arms = None
    es = None
    for f in sorted(ceo_dir.glob(f"CEO_{scm}_{cond}_seed*.csv")):
        meta = _json.loads(
            f.with_name(f.name.replace(".csv", ".meta.json")).read_text())
        ic, es = meta["init_cost"], meta["es"]
        n_arms = len(es)
        ptc = np.asarray(meta["per_trial_cost"], dtype=float)
        df = pd.read_csv(f)
        assert len(df) == len(ptc), (f, len(df), len(ptc))
        total = np.cumsum(ptc) + ic
        per_seed.append((total, df["current_optimal"].values.astype(float)))
        finals.append(float(df["current_optimal"].iloc[-1]))
        logged.append(float(ptc.sum()))
    assert per_seed, f"no CEO units for {scm} {cond} in {ceo_dir}"
    return {
        "es": ["+".join(a) for a in es], "n_arms": n_arms,
        "init_cost": ic, "split_init_cost": 0,
        "logged_cost_mean": float(np.mean(logged)),
        "total_cost_mean": float(np.mean([c[-1] for c, _ in per_seed])),
        "final_y_mean": float(np.mean(finals)),
        "final_y_sem": float(np.std(finals, ddof=1) / np.sqrt(len(finals))),
        "_per_seed": per_seed,
    }


def family_es(dataset: str, arm: str) -> list[list[str]]:
    """Exploration set exactly as the family runner builds it (same code path)."""
    import sys
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "scripts"))
    from run_cbo_family import PARTITIONS, _load_graph  # noqa: E402
    from ccbo.coarsened_graph import CoarsenedGraph      # noqa: E402
    np.random.seed(0)
    graph, obs, _ = _load_graph(dataset, 100)
    cg = CoarsenedGraph(graph, PARTITIONS[(dataset, arm)], dataset, obs,
                        num_mc_samples=2000)
    MIS, _, _ = cg.get_sets()
    return [sorted(s) for s in MIS]


def family_method(root: Path, dataset: str, arm: str, es: list[list[str]]) -> dict:
    ic = init_cost(es, N_INIT_FAMILY)
    per_seed, finals, logged = [], [], []
    for f in sorted((root / "family_cbo").glob(f"{dataset}_{arm}_seed*.csv")):
        df = pd.read_csv(f)
        per_seed.append((df["cum_cost"].values.astype(float) + ic,
                         df["best_y"].values.astype(float)))
        finals.append(float(df["best_y"].iloc[-1]))
        logged.append(float(df["cum_cost"].iloc[-1]))
    return {
        "es": ["+".join(a) for a in es], "n_arms": len(es),
        "init_cost": ic, "split_init_cost": 0,
        "logged_cost_mean": float(np.mean(logged)),
        "total_cost_mean": float(np.mean([c[-1] for c, _ in per_seed])),
        "final_y_mean": float(np.mean(finals)),
        "final_y_sem": float(np.std(finals, ddof=1) / np.sqrt(len(finals))),
        "_per_seed": per_seed,
    }


def emit_group(name: str, methods: dict[str, dict]) -> tuple[list[str], dict]:
    cstar = min(m["total_cost_mean"] for m in methods.values())
    lines = [rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{name}"
             rf" ($C^\ast{{=}}{cstar:.0f}$)}}}} \\"]
    payload = {"C_star": cstar, "methods": {}}
    for label, m in methods.items():
        at = summarize(m["_per_seed"], cstar)
        at_cell = f"${at[0]:.3f}{{\\scriptstyle\\,\\pm\\,}}{at[1]:.3f}$" \
            if at else "n/a"
        init_cell = (f"{m['init_cost']}$+${m['split_init_cost']}"
                     if m["split_init_cost"] else f"{m['init_cost']}")
        lines.append(
            rf"{label} & {m['n_arms']} & {init_cell} & "
            rf"{m['logged_cost_mean']:.0f} & {m['total_cost_mean']:.0f} & "
            rf"${m['final_y_mean']:.3f}{{\scriptstyle\,\pm\,}}"
            rf"{m['final_y_sem']:.3f}$ & {at_cell} \\")
        payload["methods"][label] = {k: v for k, v in m.items()
                                     if not k.startswith("_")}
        payload["methods"][label]["y_at_c_star"] = at
    return lines, payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--tables-dir", type=Path, default=Path("paper/tables"))
    ap.add_argument("--summary", type=Path,
                    default=Path("artifacts/summaries/cost_accounting.json"))
    ap.add_argument("--skip-family", action="store_true",
                    help="skip the CBO-family block (needs the GPy stack)")
    ap.add_argument("--ceo-dir", type=Path, default=None,
                    help="results/ceo_minimal-style dir; adds CEO rows to "
                         "the MinimalBench groups (needs per_trial_cost in "
                         "the .meta.json sidecars)")
    args = ap.parse_args()
    root = args.results_dir

    trigger = {k: rec["trigger_step"] for k, rec in json.loads(
        (root / "minimal" / "refine_info.json").read_text()).items()}
    # refine_info keys are 'MediatedChain_C0_seed<N>' — normalize just in case.
    trigger = {k: v for k, v in trigger.items()}

    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r"\textbf{Method} & $|\mathrm{ES}|$ & \textbf{Init} & "
        r"\textbf{Logged} & \textbf{Total} & \textbf{Final $Y$} & "
        r"\textbf{$Y$ at $C^\ast$} \\", r"\midrule"]
    payload = {}

    for name, scm, cond, methods in MINIMAL_GROUPS:
        block = {m: minimal_method(root, scm, cond, m, trigger)
                 for m in methods}
        if args.ceo_dir and list(args.ceo_dir.glob(
                f"CEO_{scm}_{cond}_seed*.csv")):
            block["CEO"] = ceo_method(args.ceo_dir, scm, cond)
        glines, gpayload = emit_group(name, block)
        lines += glines + [r"\addlinespace[2pt]"]
        payload[f"{scm} {cond}"] = gpayload
        print(f"  {scm} {cond}: " + ", ".join(
            f"{m} init={b['init_cost']} total={b['total_cost_mean']:.0f}"
            for m, b in block.items()))

    if not args.skip_family:
        for dataset, title in FAMILY_ROWS:
            block = {}
            for arm in ("CBO", "QCBO"):
                es = family_es(dataset, arm)
                block[arm] = family_method(root, dataset, arm, es)
            glines, gpayload = emit_group(f"{title} (family suite)", block)
            lines += glines + [r"\addlinespace[2pt]"]
            payload[f"family {title}"] = gpayload
            print(f"  family {title}: " + ", ".join(
                f"{m} |ES|={b['n_arms']} init={b['init_cost']}"
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
        "convention": "Per-variable unit costs. Init = n_init * sum of arm "
                      "sizes (3/arm MinimalBench, 10/arm family). Logged "
                      "cum_cost excludes init; totals include it. HQCBO adds "
                      "3 points per arm exposed at its per-seed split trial. "
                      "The init-phase trajectory is unlogged: curves start at "
                      "the post-init incumbent placed at full init cost; "
                      "Y at C* uses step interpolation above that point.",
        "generated_by": "scripts/emit_cost_analysis.py",
        "groups": payload,
    }, indent=2, default=float) + "\n")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
