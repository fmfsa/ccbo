"""Trace-archive gate: verify the raw per-seed traces behind every paper number.

Given a results root holding the four suite directories

    <root>/minimal/      MinimalBench units      (Experiments A-C, Figs 2-3, Tab 1)
    <root>/family_cbo/   CBO vs QCBO family      (Fig 4 top, Tab 2 rows 1-3)
    <root>/qdcbo/        DCBO vs QDCBO (+ _e2/)  (Fig 4 middle, Tab 2 rows 4-6)
    <root>/qmcbo/        MCBO vs QMCBO (+ _e2/)  (Fig 4 bottom, Tab 2 rows 7-8, Tabs 3/5)

this script (1) checks the exact expected file grid by *parsing* filenames,
(2) recomputes the published aggregates from the raw rows and compares them at
the paper's displayed precision, and (3) re-checks the exact-invariance pairs
predicted by the quotient contract (protected conditions in
``ccbo.minibench.PERTURBATIONS``).

Exit code 0 only if every hard gate passes.  Run with the ``ccbo`` env::

    PYTHONPATH=. ~/venvs/ccbo/bin/python scripts/verify_traces.py --results-dir <root>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ccbo.metrics import cumulative_regret
from ccbo.minibench import FD_NAME, MC_NAME, PERTURBATIONS, PP_NAME

FAILURES: list[str] = []
NOTES: list[str] = []


def check(ok: bool, msg: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {msg}")
    if not ok:
        FAILURES.append(msg)


def note(msg: str) -> None:
    print(f"[info] {msg}")
    NOTES.append(msg)


def close(mean: float, published: float, decimals: int) -> bool:
    """True when ``mean`` rounds to the published value at its precision."""
    return abs(mean - published) <= 0.5 * 10 ** (-decimals) + 1e-12


# ---------------------------------------------------------------- minimal ---

MINIMAL_SEEDS = range(30)


def minimal_units() -> list[tuple[str, str, str]]:
    units = []
    for p in PERTURBATIONS:
        arms = ["CBO", "QCBO"]
        if p["scm"] == MC_NAME and p["id"] == "C0":
            arms.append("HQCBO")
        for arm in arms:
            units.append((p["scm"], p["id"], arm))
    return units


def load_minimal(root: Path, scm: str, cond: str, arm: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(root / "minimal" / f"{scm}_{cond}_{arm}_seed{seed}.csv")


def verify_minimal(root: Path) -> None:
    d = root / "minimal"
    missing = [
        f"{scm}_{cond}_{arm}_seed{s}.csv"
        for (scm, cond, arm) in minimal_units()
        for s in MINIMAL_SEEDS
        if not (d / f"{scm}_{cond}_{arm}_seed{s}.csv").exists()
    ]
    check(not missing, f"minimal grid complete (450 units); missing: {missing[:5]}")
    check((d / "refine_info.json").exists(), "minimal refine_info.json present")
    if missing:
        return

    # Exact paired invariance on protected QCBO conditions (decisions + values).
    for scm, base in ((PP_NAME, "A0"), (FD_NAME, "B0")):
        for p in PERTURBATIONS:
            if p["scm"] != scm or p["id"] == base:
                continue
            diffs, decisions_equal = [], True
            for s in MINIMAL_SEEDS:
                a = load_minimal(root, scm, base, "QCBO", s)
                b = load_minimal(root, scm, p["id"], "QCBO", s)
                diffs.append(float(np.max(np.abs(a.best_y.values - b.best_y.values))))
                decisions_equal &= a.arm.equals(b.arm) and a.x_values.equals(b.x_values)
            sup = max(diffs)
            if p["protected"]:
                check(sup == 0.0 and decisions_equal,
                      f"QCBO invariant under {scm} {p['id']} (sup|Δ|={sup:g}, "
                      f"decisions equal={decisions_equal})")
            else:
                check(sup > 0.0,
                      f"QCBO differs under quotient-visible {scm} {p['id']} "
                      f"(negative control, sup|Δ|={sup:g})")

    # Experiment B numerics: fine-CBO finals and paired ΔR50 (y* = 0).
    fd_finals = {c: [load_minimal(root, FD_NAME, c, "CBO", s).best_y.iloc[-1]
                     for s in MINIMAL_SEEDS] for c in ("B0", "B1")}
    check(close(float(np.mean(fd_finals["B0"])), 0.001, 3),
          f"FrontDoor CBO B0 final {np.mean(fd_finals['B0']):.4f} ~ 0.001")
    check(close(float(np.mean(fd_finals["B1"])), 0.006, 3),
          f"FrontDoor CBO B1 final {np.mean(fd_finals['B1']):.4f} ~ 0.006")
    dr = [cumulative_regret(load_minimal(root, FD_NAME, "B1", "CBO", s).best_y.values, 0.0)
          - cumulative_regret(load_minimal(root, FD_NAME, "B0", "CBO", s).best_y.values, 0.0)
          for s in MINIMAL_SEEDS]
    m, se = float(np.mean(dr)), float(np.std(dr, ddof=1) / np.sqrt(len(dr)))
    check(close(m, 6.1, 1), f"FrontDoor paired ΔR50 {m:.2f}±{se:.2f} ~ 6.1±1.0")

    # Experiment C: finals 0.040 / 4.000 / 0.040 and trigger step 7.5.
    for arm, pub in (("CBO", 0.040), ("QCBO", 4.000), ("HQCBO", 0.040)):
        finals = [load_minimal(root, MC_NAME, "C0", arm, s).best_y.iloc[-1]
                  for s in MINIMAL_SEEDS]
        check(close(float(np.mean(finals)), pub, 3),
              f"MediatedChain {arm} final {np.mean(finals):.4f} ~ {pub:.3f}")
    info = json.loads((d / "refine_info.json").read_text())
    trig = [rec["trigger_step"] for rec in info.values() if rec.get("trigger_step")]
    tm = float(np.mean(trig))
    check(len(trig) == 30 and close(tm, 7.5, 1),
          f"refinement trigger mean {tm:.2f} over {len(trig)} seeds ~ 7.5")

    # Experiment A (informational): fine CBO under A1 plateaus near the 4.25 gap.
    a1 = float(np.mean([load_minimal(root, PP_NAME, "A1", "CBO", s).best_y.iloc[-1]
                        for s in MINIMAL_SEEDS]))
    note(f"ParallelParent CBO A1 mean final {a1:.3f} (analytic floor 4.25)")


# ----------------------------------------------------------------- family ---

FAMILY = {  # dataset -> (published CBO final, published QCBO final), 2 decimals
    "ToyGraph": (-2.16, -2.13),
    "CompleteGraph": (-3.14, -1.25),
    "SimplifiedCoralGraph": (36.08, 36.42),
}


def verify_family(root: Path) -> None:
    d = root / "family_cbo"
    missing = [f"{ds}_{arm}_seed{s}.csv"
               for ds in FAMILY for arm in ("CBO", "QCBO") for s in range(10)
               if not (d / f"{ds}_{arm}_seed{s}.csv").exists()]
    check(not missing, f"family_cbo grid complete (60); missing: {missing[:5]}")
    if missing:
        return
    for ds, (pub_cbo, pub_qcbo) in FAMILY.items():
        for arm, pub in (("CBO", pub_cbo), ("QCBO", pub_qcbo)):
            finals = [pd.read_csv(d / f"{ds}_{arm}_seed{s}.csv").best_y.iloc[-1]
                      for s in range(10)]
            m = float(np.mean(finals))
            check(close(m, pub, 2), f"family {ds} {arm} final {m:.3f} ~ {pub:.2f}")


# ------------------------------------------------------------------ qdcbo ---

QDCBO_PUB = {"stat": (-6.14, -6.43), "ind": (-3.12, -5.57), "nonstat": (8.03, 6.33)}


def qdcbo_name(method: str, setup: str, seed: int, e2: bool) -> str:
    tag = f"{setup}_e2" if e2 else setup
    return f"{method}_{tag}_best_so_far_seed{seed}_T3_trials10_reps1.csv"


def verify_qdcbo(root: Path) -> None:
    d = root / "qdcbo"
    missing = [qdcbo_name(m, s, n, e2)
               for m in ("dcbo", "qdcbo") for s in QDCBO_PUB for n in range(20)
               for e2 in (False, True)
               if not (d / ("_e2" if e2 else "") / qdcbo_name(m, s, n, e2)).exists()]
    check(not missing, f"qdcbo grid complete (120 E1 + 120 E2); missing: {missing[:5]}")
    if missing:
        return
    for setup, (pub_d, pub_q) in QDCBO_PUB.items():
        for method, pub in (("dcbo", pub_d), ("qdcbo", pub_q)):
            finals = [pd.read_csv(d / qdcbo_name(method, setup, n, False))
                      .best_so_far_value.iloc[-1] for n in range(20)]
            m = float(np.mean(finals))
            check(close(m, pub, 2), f"qdcbo {setup} {method} final {m:.3f} ~ {pub:.2f}")
    # QDCBO must be trajectory-identical under the intra-slice E2 edit.
    bad = []
    for setup in QDCBO_PUB:
        for n in range(20):
            e1 = pd.read_csv(d / qdcbo_name("qdcbo", setup, n, False))
            e2 = pd.read_csv(d / "_e2" / qdcbo_name("qdcbo", setup, n, True))
            if not np.array_equal(e1.best_so_far_value.values,
                                  e2.best_so_far_value.values):
                bad.append((setup, n))
    check(not bad, f"QDCBO invariant under E2 on all 60 units; diffs: {bad[:5]}")


# ------------------------------------------------------------------ qmcbo ---

QMCBO_PUB = {  # env -> (MCBO final, QMCBO final), 2 decimals
    "ToyGraph": (1.14, 2.15),
    "PSAGraph": (-5.15, -5.15),
}
QMCBO_E2_IDENTICAL = {  # (env, method) -> (identical count, mean |Δfinal| 3dp)
    ("ToyGraph", "MCBO"): (2, 0.299),
    ("ToyGraph", "QMCBO"): (5, 0.000),
    ("PSAGraph", "MCBO"): (0, 0.000),
    ("PSAGraph", "QMCBO"): (5, 0.000),
}


def verify_qmcbo(root: Path) -> None:
    d = root / "qmcbo"
    missing = [f"{sub}trial_results_{m}_{env}_{s}.csv"
               for env in QMCBO_PUB for m in ("MCBO", "QMCBO") for s in range(5)
               for sub in ("", "_e2/")
               if not (d / sub / f"trial_results_{m}_{env}_{s}.csv").exists()]
    check(not missing, f"qmcbo grid complete (20 E1 + 20 E2); missing: {missing[:5]}")
    if missing:
        return
    for env, (pub_m, pub_q) in QMCBO_PUB.items():
        for method, pub in (("MCBO", pub_m), ("QMCBO", pub_q)):
            finals = [pd.read_csv(d / f"trial_results_{method}_{env}_{s}.csv")
                      .current_optimal.iloc[-1] for s in range(5)]
            m = float(np.mean(finals))
            check(close(m, pub, 2), f"qmcbo {env} {method} final {m:.4f} ~ {pub:.2f}")
    for (env, method), (pub_n, pub_delta) in QMCBO_E2_IDENTICAL.items():
        n_id, deltas = 0, []
        for s in range(5):
            e1 = pd.read_csv(d / f"trial_results_{method}_{env}_{s}.csv")
            e2 = pd.read_csv(d / "_e2" / f"trial_results_{method}_{env}_{s}.csv")
            same = np.array_equal(e1.current_optimal.values, e2.current_optimal.values)
            n_id += int(same)
            deltas.append(abs(e1.current_optimal.iloc[-1] - e2.current_optimal.iloc[-1]))
        md = float(np.mean(deltas))
        check(n_id == pub_n and close(md, pub_delta, 3),
              f"qmcbo E2 {env} {method}: {n_id}/5 identical (pub {pub_n}/5), "
              f"mean|Δfinal| {md:.4f} ~ {pub_delta:.3f}")


# ------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="root holding minimal/, family_cbo/, qdcbo/, qmcbo/")
    args = ap.parse_args()
    root = args.results_dir.resolve()
    print(f"Verifying traces under {root}\n")
    verify_minimal(root)
    verify_family(root)
    verify_qdcbo(root)
    verify_qmcbo(root)
    print(f"\n{len(FAILURES)} failure(s).")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
