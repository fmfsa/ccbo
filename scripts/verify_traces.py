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
    # BO added 2026-08-21 (rerun/online-obs-v2): every condition also runs the
    # plain-BO baseline arm, so the grid is 660 units (450 causal + 210 BO).
    units = []
    for p in PERTURBATIONS:
        arms = ["BO", "CBO", "QCBO"]
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
    check(not missing, f"minimal grid complete (660 units); missing: {missing[:5]}")
    check((d / "refine_info.json").exists(), "minimal refine_info.json present")
    if missing:
        return

    # BO reads no graph structure, so its trajectory must be byte-identical
    # across every misspecification condition of an SCM (free harness check;
    # Table 1's BO sup|Delta| column must read 0 on every row).
    for scm, conds in ((PP_NAME, ("A0", "A1", "A2", "A3")),
                       (FD_NAME, ("B0", "B1"))):
        bad = []
        for s in MINIMAL_SEEDS:
            payloads = {c: (d / f"{scm}_{c}_BO_seed{s}.csv").read_bytes()
                        for c in conds}
            if len(set(payloads.values())) != 1:
                bad.append(s)
        check(not bad,
              f"BO byte-identical across {scm} conditions {conds}; "
              f"leaking seeds: {bad[:5]}")

    # Online-observation budget: n_obs=100, batch 20, N_max=150 allows at most
    # ceil(50/20) = 3 observe actions per unit; BO takes none by construction.
    over, bo_obs = [], []
    for (scm, cond, arm) in minimal_units():
        for s in MINIMAL_SEEDS:
            df = load_minimal(root, scm, cond, arm, s)
            body = df.iloc[1:]
            n_obs = int((body.arm.isna()
                         | (body.arm.astype(str).str.strip() == "")).sum())
            if arm == "BO" and n_obs > 0:
                bo_obs.append((scm, cond, s))
            elif arm != "BO" and n_obs > 3:
                over.append((scm, cond, arm, s, n_obs))
    check(not over, f"observe actions per unit <= 3 (budget cap); over: {over[:5]}")
    check(not bo_obs, f"BO units take no observe actions; violations: {bo_obs[:5]}")

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
    # Published values refreshed 2026-08-21 from the online-observation rerun
    # (results/v2/minimal; artifacts/summaries/minimal_exact.json).
    fd_finals = {c: [load_minimal(root, FD_NAME, c, "CBO", s).best_y.iloc[-1]
                     for s in MINIMAL_SEEDS] for c in ("B0", "B1")}
    check(close(float(np.mean(fd_finals["B0"])), 0.0001, 4),
          f"FrontDoor CBO B0 final {np.mean(fd_finals['B0']):.5f} ~ 0.0001")
    check(close(float(np.mean(fd_finals["B1"])), 0.0019, 4),
          f"FrontDoor CBO B1 final {np.mean(fd_finals['B1']):.5f} ~ 0.0019")
    dr = [cumulative_regret(load_minimal(root, FD_NAME, "B1", "CBO", s).best_y.values, 0.0)
          - cumulative_regret(load_minimal(root, FD_NAME, "B0", "CBO", s).best_y.values, 0.0)
          for s in MINIMAL_SEEDS]
    m, se = float(np.mean(dr)), float(np.std(dr, ddof=1) / np.sqrt(len(dr)))
    check(close(m, 6.2, 1), f"FrontDoor paired ΔR50 {m:.2f}±{se:.2f} ~ 6.2±1.1")

    # Experiment C: finals 0.040 / 4.000 / 0.040 and trigger step 7.3.
    for arm, pub in (("CBO", 0.040), ("QCBO", 4.000), ("HQCBO", 0.040)):
        finals = [load_minimal(root, MC_NAME, "C0", arm, s).best_y.iloc[-1]
                  for s in MINIMAL_SEEDS]
        check(close(float(np.mean(finals)), pub, 3),
              f"MediatedChain {arm} final {np.mean(finals):.4f} ~ {pub:.3f}")
    info = json.loads((d / "refine_info.json").read_text())
    trig = [rec["trigger_step"] for rec in info.values() if rec.get("trigger_step")]
    tm = float(np.mean(trig))
    check(len(trig) == 30 and close(tm, 7.3, 1),
          f"refinement trigger mean {tm:.2f} over {len(trig)} seeds ~ 7.3")

    # Experiment A (informational): fine CBO under A1 plateaus near the 4.25 gap.
    a1 = float(np.mean([load_minimal(root, PP_NAME, "A1", "CBO", s).best_y.iloc[-1]
                        for s in MINIMAL_SEEDS]))
    note(f"ParallelParent CBO A1 mean final {a1:.3f} (analytic floor 4.25)")


# ----------------------------------------------------------------- family ---

# dataset -> (published BO, CBO, QCBO finals), 2 decimals.  Refreshed
# 2026-08-21 from the online-observation rerun (results/v2/family_cbo);
# BO baseline arm added in the same rerun (90-unit grid).
FAMILY = {
    "ToyGraph": (-2.17, -2.16, -2.17),
    "CompleteGraph": (-0.63, -3.47, -1.28),
    "SimplifiedCoralGraph": (9279.47, 36.07, 36.45),
}


def verify_family(root: Path) -> None:
    d = root / "family_cbo"
    missing = [f"{ds}_{arm}_seed{s}.csv"
               for ds in FAMILY for arm in ("BO", "CBO", "QCBO")
               for s in range(10)
               if not (d / f"{ds}_{arm}_seed{s}.csv").exists()]
    check(not missing, f"family_cbo grid complete (90); missing: {missing[:5]}")
    if missing:
        return
    for ds, (pub_bo, pub_cbo, pub_qcbo) in FAMILY.items():
        for arm, pub in (("BO", pub_bo), ("CBO", pub_cbo), ("QCBO", pub_qcbo)):
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

QMCBO_SEEDS = 20   # extended 5 -> 20 on 2026-08-11 (LSF job 29078186)
QMCBO_PUB = {  # env -> (MCBO final, QMCBO final), 2 decimals
    "ToyGraph": (1.39, 2.16),
    "PSAGraph": (-5.15, -5.15),
}
QMCBO_E2_IDENTICAL = {  # (env, method) -> (identical count, mean |Δfinal| 3dp)
    ("ToyGraph", "MCBO"): (9, 0.277),
    ("ToyGraph", "QMCBO"): (20, 0.000),
    ("PSAGraph", "MCBO"): (0, 0.001),
    ("PSAGraph", "QMCBO"): (20, 0.000),
}


def verify_qmcbo(root: Path) -> None:
    d = root / "qmcbo"
    missing = [f"{sub}trial_results_{m}_{env}_{s}.csv"
               for env in QMCBO_PUB for m in ("MCBO", "QMCBO") for s in range(QMCBO_SEEDS)
               for sub in ("", "_e2/")
               if not (d / sub / f"trial_results_{m}_{env}_{s}.csv").exists()]
    check(not missing, f"qmcbo grid complete ({QMCBO_SEEDS*4} E1 + {QMCBO_SEEDS*4} E2); missing: {missing[:5]}")
    if missing:
        return
    for env, (pub_m, pub_q) in QMCBO_PUB.items():
        for method, pub in (("MCBO", pub_m), ("QMCBO", pub_q)):
            finals = [pd.read_csv(d / f"trial_results_{method}_{env}_{s}.csv")
                      .current_optimal.iloc[-1] for s in range(QMCBO_SEEDS)]
            m = float(np.mean(finals))
            check(close(m, pub, 2), f"qmcbo {env} {method} final {m:.4f} ~ {pub:.2f}")
    for (env, method), (pub_n, pub_delta) in QMCBO_E2_IDENTICAL.items():
        n_id, deltas = 0, []
        for s in range(QMCBO_SEEDS):
            e1 = pd.read_csv(d / f"trial_results_{method}_{env}_{s}.csv")
            e2 = pd.read_csv(d / "_e2" / f"trial_results_{method}_{env}_{s}.csv")
            same = np.array_equal(e1.current_optimal.values, e2.current_optimal.values)
            n_id += int(same)
            deltas.append(abs(e1.current_optimal.iloc[-1] - e2.current_optimal.iloc[-1]))
        md = float(np.mean(deltas))
        check(n_id == pub_n and close(md, pub_delta, 3),
              f"qmcbo E2 {env} {method}: {n_id}/{QMCBO_SEEDS} identical (pub {pub_n}/{QMCBO_SEEDS}), "
              f"mean|Δfinal| {md:.4f} ~ {pub_delta:.3f}")


# -------------------------------------------------------------------- ceo ---

CEO_UNITS = {  # (scm, cond) -> (trials, executed?, alias_of, pool_pids)
    ("ParallelParent", "A0"): (50, True, None, ["A0", "A1", "A2"]),
    ("ParallelParent", "A1"): (50, True, None, ["A1", "A0", "A2"]),
    ("ParallelParent", "A2"): (50, True, None, ["A2", "A0", "A1"]),
    ("ParallelParent", "A3"): (50, False, "A0", ["A0", "A1", "A2"]),
    ("FrontDoor", "B0"): (50, True, None, ["B0"]),
    ("FrontDoor", "B1"): (50, False, "B0", ["B0"]),
    ("MediatedChain", "C0"): (60, True, None, ["C0"]),
}
# Published CEO finals at 2 decimals (corrected rerun, LSF 29090857).
CEO_PUB: dict | None = {
    ("ParallelParent", "A0"): 1.0,
    ("ParallelParent", "A1"): 0.77,
    ("ParallelParent", "A2"): 0.71,
    ("ParallelParent", "A3"): 1.0,
    ("FrontDoor", "B0"): 0.07,
    ("FrontDoor", "B1"): 0.07,
    ("MediatedChain", "C0"): 0.13,
}
CEO_SEEDS = range(30)


def verify_ceo(root: Path) -> None:
    d = root / "ceo_minimal"
    if not d.exists():
        check(False, "ceo_minimal directory present")
        return
    missing = [f"CEO_{scm}_{cond}_seed{s}.csv"
               for (scm, cond) in CEO_UNITS for s in CEO_SEEDS
               if not (d / f"CEO_{scm}_{cond}_seed{s}.csv").exists()]
    check(not missing, f"ceo grid complete (150 executed + 60 aliases); "
                       f"missing: {missing[:5]}")
    if missing:
        return
    for (scm, cond), (trials, executed, alias_of, pool) in CEO_UNITS.items():
        bad_len, bad_meta, bad_alias = [], [], []
        finals = []
        for s in CEO_SEEDS:
            f = d / f"CEO_{scm}_{cond}_seed{s}.csv"
            df = pd.read_csv(f)
            if len(df) != trials + 1:
                bad_len.append((s, len(df)))
            finals.append(float(df.current_optimal.iloc[-1]))
            meta = json.loads(
                (d / f"CEO_{scm}_{cond}_seed{s}.meta.json").read_text())
            if not (meta.get("cond") == cond and meta.get("trials") == trials
                    and meta.get("ninit") == 3 and meta.get("n_obs") == 100
                    and meta.get("pool_pids") == pool
                    and meta.get("alias_of") == alias_of
                    and len(meta.get("per_trial_cost", [])) == trials + 1
                    and meta.get("per_trial_cost", [None])[0] == 0.0):
                bad_meta.append(s)
            if alias_of is not None:
                base = d / f"CEO_{scm}_{alias_of}_seed{s}.csv"
                if f.read_bytes() != base.read_bytes():
                    bad_alias.append(s)
        check(not bad_len and not bad_meta,
              f"ceo {scm} {cond}: trajectories {trials + 1} rows, metadata "
              f"protocol-consistent (len issues {bad_len[:3]}, "
              f"meta issues {bad_meta[:3]})")
        if alias_of is not None:
            check(not bad_alias,
                  f"ceo alias {scm} {cond} byte-equal to {alias_of}; "
                  f"diffs: {bad_alias[:3]}")
        if CEO_PUB is not None and (scm, cond) in CEO_PUB:
            m = float(np.mean(finals))
            check(close(m, CEO_PUB[(scm, cond)], 2),
                  f"ceo {scm} {cond} final {m:.4f} ~ "
                  f"{CEO_PUB[(scm, cond)]:.2f}")
    if CEO_PUB is None:
        note("ceo published-finals check skipped (CEO_PUB not yet frozen)")


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
    verify_ceo(root)
    print(f"\n{len(FAILURES)} failure(s).")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
