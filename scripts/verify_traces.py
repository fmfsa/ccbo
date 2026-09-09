"""Trace-archive gate: verify the raw per-seed traces behind every paper number.

Given a results root holding the four suite directories

    <root>/minimal/      MinimalBench units      (Experiments A-C, Figs 2-3, Tab 1)
    <root>/family_cbo/   CBO vs QCBO family      (Fig 4 top, Tab 2 rows 1-3)
    <root>/qdcbo/        DCBO vs QDCBO (+ _e2/)  (Fig 4 middle, Tab 2 rows 4-6)
    <root>/qmcbo/        MCBO vs QMCBO (+ _e2/)  (Fig 4 bottom, Tab 2 rows 7-8, Tabs 3/5)

(engine v3, 2026-09: MinimalBench carries 1320 units over the arms
BO/BOS/CBO/CBONP/QCBO/QCBONP (+HQCBO/HQCBOGF on MediatedChain), every unit
has a full-precision ``.decisions.json`` sidecar, and ``cum_cost`` row 0 is
the charged initial design)

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
PLAIN_ARMS = ("BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP")
REFINE_ARMS = ("HQCBO", "HQCBOGF")
N_INIT_MINIMAL = 3
# Prior-variance policy every static-suite sidecar must record (engine v3,
# 2026-09-10): the predictive law of total variance under the fitted GP.
PRIOR_VARIANCE_POLICY = "predictive"

# Published MinimalBench aggregates (paper's displayed precision).  ``None``
# until the engine-v3 numbers are frozen; the structural and invariance
# checks below run regardless.
# 2026-09-09 freeze (LSF 29365450, SHA f2a98e0, variance policy "total")
# retired 2026-09-10: re-frozen after the predictive-policy rerun.
MINIMAL_PUB: dict | None = None


def minimal_units() -> list[tuple[str, str, str]]:
    units = []
    for p in PERTURBATIONS:
        arms = list(PLAIN_ARMS)
        if p["scm"] == MC_NAME and p["id"] == "C0":
            arms += list(REFINE_ARMS)
        for arm in arms:
            units.append((p["scm"], p["id"], arm))
    return units


def load_minimal(root: Path, scm: str, cond: str, arm: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(root / "minimal" / f"{scm}_{cond}_{arm}_seed{seed}.csv")


def load_sidecar(root: Path, scm: str, cond: str, arm: str, seed: int) -> dict:
    from ccbo.decision_log import read
    return read(str(root / "minimal" / f"{scm}_{cond}_{arm}_seed{seed}.decisions.json"))


def _n_observe(df: pd.DataFrame) -> int:
    body = df.iloc[1:]
    return int((body.arm.isna() | (body.arm.astype(str).str.strip() == "")).sum())


def verify_minimal(root: Path) -> None:
    from ccbo.decision_log import decisions_signature
    d = root / "minimal"
    units = minimal_units()
    missing = [f"{scm}_{cond}_{arm}_seed{s}.csv"
               for (scm, cond, arm) in units for s in MINIMAL_SEEDS
               if not (d / f"{scm}_{cond}_{arm}_seed{s}.csv").exists()]
    check(not missing, f"minimal grid complete ({len(units) * 30} units); missing: {missing[:5]}")
    no_side = [f"{scm}_{cond}_{arm}_seed{s}.decisions.json"
               for (scm, cond, arm) in units for s in MINIMAL_SEEDS
               if not (d / f"{scm}_{cond}_{arm}_seed{s}.decisions.json").exists()]
    check(not no_side, f"minimal decision-log sidecars present; missing: {no_side[:5]}")
    check((d / "refine_info.json").exists(), "minimal refine_info.json present")
    if missing or no_side:
        return

    # ---- Structural: row 0 = charged initial design (unit costs), observe
    # budget, gate status, split charge at the trigger.
    bad_init, over, bo_obs, gate_err, bad_split = [], [], [], [], []
    trig = {}
    policies, shas = set(), set()
    for (scm, cond, arm) in units:
        for s in MINIMAL_SEEDS:
            df = load_minimal(root, scm, cond, arm, s)
            log = load_sidecar(root, scm, cond, arm, s)
            policies.add(log["unit"].get("estimator", {}).get("variance_policy"))
            shas.add(log["unit"].get("engine_sha"))
            es = log["init"]["es"]
            ic = float(log["init"]["init_cost"])
            if abs(df.cum_cost.iloc[0] - ic) > 1e-9 or abs(ic - N_INIT_MINIMAL * sum(len(a) for a in es)) > 1e-9:
                bad_init.append((scm, cond, arm, s, float(df.cum_cost.iloc[0]), ic))
            n_obs = _n_observe(df)
            if arm in ("BO", "BOS") and n_obs > 0:
                bo_obs.append((scm, cond, arm, s))
            elif arm not in ("BO", "BOS") and n_obs > 3:
                over.append((scm, cond, arm, s, n_obs))
            if any(v == "error" for v in log["init"].get("gate_status", {}).values()):
                gate_err.append((scm, cond, arm, s))
            if arm in REFINE_ARMS:
                ref = log.get("refine") or {}
                t = ref.get("trigger_step")
                trig[(arm, s)] = t
                if t is not None:
                    charged = df.cum_cost.iloc[t] - df.cum_cost.iloc[t - 1]
                    if charged + 1e-9 < float(ref.get("split_init_cost", 0.0)):
                        bad_split.append((arm, s, float(charged), ref.get("split_init_cost")))
    check(not bad_init, f"row 0 cum_cost == charged initial design (3 pts/arm, unit costs); bad: {bad_init[:3]}")
    check(policies == {PRIOR_VARIANCE_POLICY},
          f"minimal sidecars record variance_policy == {PRIOR_VARIANCE_POLICY!r} (seen {sorted(map(str, policies))})")
    check(len(shas) == 1, f"minimal sidecars record one engine SHA (seen {sorted(map(str, shas))})")
    check(not over, f"observe actions per causal unit <= 3 (budget cap); over: {over[:3]}")
    check(not bo_obs, f"BO / BOS units take no observe actions; violations: {bo_obs[:3]}")
    check(not gate_err, f"no identification-gate error state in any unit; errors: {gate_err[:3]}")
    check(not bad_split, f"split design charged at the trigger row; bad: {bad_split[:3]}")
    same_trig = all(trig.get(("HQCBO", s)) == trig.get(("HQCBOGF", s)) for s in MINIMAL_SEEDS)
    check(same_trig, "HQCBO and HQCBOGF share the QCBO plateau trigger on every seed")

    # ---- BO and BOS read no graph: byte-identical across an SCM's conditions.
    for scm, conds in ((PP_NAME, ("A0", "A1", "A2", "A3")), (FD_NAME, ("B0", "B1"))):
        for arm in ("BO", "BOS"):
            bad = [s for s in MINIMAL_SEEDS
                   if len({(d / f"{scm}_{c}_{arm}_seed{s}.csv").read_bytes() for c in conds}) != 1]
            check(not bad, f"{arm} byte-identical across {scm} conditions {conds}; leaking seeds: {bad[:5]}")

    # ---- Exact paired invariance (CSV + full-precision sidecar signatures).
    for scm, base in ((PP_NAME, "A0"), (FD_NAME, "B0")):
        for p in PERTURBATIONS:
            if p["scm"] != scm or p["id"] == base:
                continue
            for arm in ("QCBO", "QCBONP"):
                diffs, decisions_equal, sig_equal = [], True, True
                for s in MINIMAL_SEEDS:
                    a = load_minimal(root, scm, base, arm, s)
                    b = load_minimal(root, scm, p["id"], arm, s)
                    diffs.append(float(np.max(np.abs(a.best_y.values - b.best_y.values))))
                    decisions_equal &= a.arm.equals(b.arm) and a.x_values.equals(b.x_values)
                    sig_equal &= (decisions_signature(load_sidecar(root, scm, base, arm, s))
                                  == decisions_signature(load_sidecar(root, scm, p["id"], arm, s)))
                sup = max(diffs)
                if p["protected"] or arm == "QCBONP" and p["id"] == "A3":
                    # QCBONP carries no prior, so even the prior-only A3 edit
                    # cannot move it; every protected edit must leave both arms.
                    check(sup == 0.0 and decisions_equal and sig_equal,
                          f"{arm} invariant under {scm} {p['id']} (sup|Δ|={sup:g}, "
                          f"decisions equal={decisions_equal}, full-precision log equal={sig_equal})")
                else:
                    check(sup > 0.0 or not sig_equal,
                          f"{arm} differs under quotient-visible {scm} {p['id']} "
                          f"(negative control, sup|Δ|={sup:g})")

    # ---- Published aggregates (skipped until frozen).
    if MINIMAL_PUB is None:
        note("minimal published-aggregate checks skipped (MINIMAL_PUB not yet frozen)")
        return
    fd_finals = {c: [load_minimal(root, FD_NAME, c, "CBO", s).best_y.iloc[-1]
                     for s in MINIMAL_SEEDS] for c in ("B0", "B1")}
    v, dec = MINIMAL_PUB["fd_b0_cbo_final"]
    check(close(float(np.mean(fd_finals["B0"])), v, dec), f"FrontDoor CBO B0 final {np.mean(fd_finals['B0']):.5f} ~ {v}")
    v, dec = MINIMAL_PUB["fd_b1_cbo_final"]
    check(close(float(np.mean(fd_finals["B1"])), v, dec), f"FrontDoor CBO B1 final {np.mean(fd_finals['B1']):.5f} ~ {v}")
    dr = [cumulative_regret(load_minimal(root, FD_NAME, "B1", "CBO", s).best_y.values, 0.0)
          - cumulative_regret(load_minimal(root, FD_NAME, "B0", "CBO", s).best_y.values, 0.0)
          for s in MINIMAL_SEEDS]
    m, se = float(np.mean(dr)), float(np.std(dr, ddof=1) / np.sqrt(len(dr)))
    v, dec = MINIMAL_PUB["fd_paired_dr50"]
    check(close(m, v, dec), f"FrontDoor paired ΔR50 {m:.2f}±{se:.2f} ~ {v}")
    for arm, pub in MINIMAL_PUB["mc_finals"].items():
        finals = [load_minimal(root, MC_NAME, "C0", arm, s).best_y.iloc[-1] for s in MINIMAL_SEEDS]
        check(close(float(np.mean(finals)), pub, 3), f"MediatedChain {arm} final {np.mean(finals):.4f} ~ {pub:.3f}")
    info = json.loads((d / "refine_info.json").read_text())
    tr = [rec["trigger_step"] for k, rec in info.items()
          if "_HQCBO_" in k and rec.get("trigger_step") is not None]
    tm = float(np.mean(tr))
    v, dec = MINIMAL_PUB["trigger_mean"]
    check(len(tr) == 30 and close(tm, v, dec), f"refinement trigger mean {tm:.2f} over {len(tr)} seeds ~ {v}")


# ----------------------------------------------------------------- family ---

# dataset -> (published BO, CBO, QCBO finals), 2 decimals.  ``None`` until the
# engine-v3 family numbers are frozen.
# 2026-09-09 freeze (LSF 29365451, SHA f2a98e0, variance policy "total") retired
# 2026-09-10: its priors were hash-seed dependent and the policy was replaced.
FAMILY_PUB: dict | None = None
FAMILY_DATASETS = ("ToyGraph", "CompleteGraph", "SimplifiedCoralGraph")
N_INIT_FAMILY = 10


def verify_family(root: Path) -> None:
    from ccbo.decision_log import read
    d = root / "family_cbo"
    missing = [f"{ds}_{arm}_seed{s}.csv"
               for ds in FAMILY_DATASETS for arm in ("BO", "CBO", "QCBO")
               for s in range(10)
               if not (d / f"{ds}_{arm}_seed{s}.csv").exists()
               or not (d / f"{ds}_{arm}_seed{s}.decisions.json").exists()]
    check(not missing, f"family_cbo grid complete (90 units + sidecars); missing: {missing[:5]}")
    if missing:
        return
    bad_init = []
    policies, shas = set(), set()
    for ds in FAMILY_DATASETS:
        for arm in ("BO", "CBO", "QCBO"):
            for s in range(10):
                df = pd.read_csv(d / f"{ds}_{arm}_seed{s}.csv")
                log = read(str(d / f"{ds}_{arm}_seed{s}.decisions.json"))
                policies.add(log["unit"].get("estimator", {}).get("variance_policy"))
                shas.add(log["unit"].get("engine_sha"))
                ic = float(log["init"]["init_cost"])
                expect = N_INIT_FAMILY * sum(len(a) for a in log["init"]["es"])
                if abs(df.cum_cost.iloc[0] - ic) > 1e-9 or abs(ic - expect) > 1e-9:
                    bad_init.append((ds, arm, s, float(df.cum_cost.iloc[0]), ic, expect))
    check(not bad_init, f"family row 0 cum_cost == charged initial design (10 pts/arm); bad: {bad_init[:3]}")
    check(policies == {PRIOR_VARIANCE_POLICY},
          f"family sidecars record variance_policy == {PRIOR_VARIANCE_POLICY!r} (seen {sorted(map(str, policies))})")
    check(len(shas) == 1, f"family sidecars record one engine SHA (seen {sorted(map(str, shas))})")
    if FAMILY_PUB is None:
        note("family published-final checks skipped (FAMILY_PUB not yet frozen)")
        return
    for ds, (pub_bo, pub_cbo, pub_qcbo) in FAMILY_PUB.items():
        for arm, pub in (("BO", pub_bo), ("CBO", pub_cbo), ("QCBO", pub_qcbo)):
            finals = [pd.read_csv(d / f"{ds}_{arm}_seed{s}.csv").best_y.iloc[-1]
                      for s in range(10)]
            m = float(np.mean(finals))
            check(close(m, pub, 2), f"family {ds} {arm} final {m:.3f} ~ {pub:.2f}")


# ------------------------------------------------------------------ qdcbo ---

# Published finals (2 dp); ``None`` until the engine-v3 (corrected stock
# semantics) numbers are frozen.  Pre-v3 archive values were
# {"stat": (-6.14, -6.43), "ind": (-3.12, -5.57), "nonstat": (8.03, 6.33)}.
# Frozen 2026-09-09 from results/v3 (LSF 29365482, engine SHA 9f25b41, corrected stock semantics).
QDCBO_PUB: dict | None = {"stat": [-6.12, -6.4], "ind": [-3.13, -5.55], "nonstat": [7.01, 5.78]}
QDCBO_SETUPS = ("stat", "ind", "nonstat")


def qdcbo_name(method: str, setup: str, seed: int, e2: bool) -> str:
    tag = f"{setup}_e2" if e2 else setup
    return f"{method}_{tag}_best_so_far_seed{seed}_T3_trials10_reps1.csv"


def verify_qdcbo(root: Path) -> None:
    d = root / "qdcbo"
    missing = [qdcbo_name(m, s, n, e2)
               for m in ("dcbo", "qdcbo") for s in QDCBO_SETUPS for n in range(20)
               for e2 in (False, True)
               if not (d / ("_e2" if e2 else "") / qdcbo_name(m, s, n, e2)).exists()]
    check(not missing, f"qdcbo grid complete (120 E1 + 120 E2); missing: {missing[:5]}")
    if missing:
        return
    # Engine v3: every unit carries a decision sidecar, none in stock-quirks mode.
    side = [(m, s, n, e2) for m in ("dcbo", "qdcbo") for s in QDCBO_SETUPS
            for n in range(20) for e2 in (False, True)
            if not (d / ("_e2" if e2 else "") / qdcbo_name(m, s, n, e2)
                    .replace(".csv", ".decisions.json")).exists()]
    check(not side, f"qdcbo decision sidecars present; missing: {side[:5]}")
    if not side:
        quirks = []
        for m in ("dcbo", "qdcbo"):
            for setup in QDCBO_SETUPS:
                for n in range(20):
                    for e2 in (False, True):
                        pl = json.loads((d / ("_e2" if e2 else "") / qdcbo_name(m, setup, n, e2)
                                         .replace(".csv", ".decisions.json")).read_text())
                        if pl["unit"].get("stock_quirks"):
                            quirks.append((m, setup, n, e2))
        check(not quirks, f"qdcbo units run with corrected stock semantics (stock_quirks=False); quirks: {quirks[:5]}")
    if QDCBO_PUB is not None:
        for setup, (pub_d, pub_q) in QDCBO_PUB.items():
            for method, pub in (("dcbo", pub_d), ("qdcbo", pub_q)):
                finals = [pd.read_csv(d / qdcbo_name(method, setup, n, False))
                          .best_so_far_value.iloc[-1] for n in range(20)]
                m = float(np.mean(finals))
                check(close(m, pub, 2), f"qdcbo {setup} {method} final {m:.3f} ~ {pub:.2f}")
    else:
        note("qdcbo published-final checks skipped (QDCBO_PUB not yet frozen)")
    # QDCBO must be trajectory-identical under the intra-slice E2 edit --
    # values (CSV) and, when sidecars exist, decisions (chosen sets + levels).
    bad, bad_dec = [], []
    for setup in QDCBO_SETUPS:
        for n in range(20):
            e1 = pd.read_csv(d / qdcbo_name("qdcbo", setup, n, False))
            e2 = pd.read_csv(d / "_e2" / qdcbo_name("qdcbo", setup, n, True))
            if not np.array_equal(e1.best_so_far_value.values,
                                  e2.best_so_far_value.values):
                bad.append((setup, n))
            p1 = d / qdcbo_name("qdcbo", setup, n, False).replace(".csv", ".decisions.json")
            p2 = d / "_e2" / qdcbo_name("qdcbo", setup, n, True).replace(".csv", ".decisions.json")
            if p1.exists() and p2.exists():
                a, b = json.loads(p1.read_text()), json.loads(p2.read_text())
                if a["per_t"] != b["per_t"]:
                    bad_dec.append((setup, n))
    check(not bad, f"QDCBO invariant under E2 on all 60 units (values); diffs: {bad[:5]}")
    check(not bad_dec, f"QDCBO invariant under E2 on all 60 units (decisions, full precision); diffs: {bad_dec[:5]}")


# ------------------------------------------------------------------ qmcbo ---

QMCBO_SEEDS = 20   # extended 5 -> 20 on 2026-08-11 (LSF job 29078186)
QMCBO_ENVS = ("ToyGraph", "PSAGraph")
# Published finals / E2 counts; ``None`` until the engine-v3 numbers are
# frozen.  Pre-v3 archive: ToyGraph (1.39, 2.16), PSAGraph (-5.15, -5.15);
# E2 identical: ToyGraph MCBO 9/20 (0.277), QMCBO 20/20; PSAGraph MCBO 0/20
# (0.001), QMCBO 20/20.
# Frozen 2026-09-09 from results/v3 (LSF 29365483, engine SHA 9f25b41).
QMCBO_PUB: dict | None = {"ToyGraph": [1.39, 2.16], "PSAGraph": [-5.15, -5.15]}
QMCBO_E2_IDENTICAL: dict | None = {("ToyGraph", "MCBO"): (9, 0.277), ("ToyGraph", "QMCBO"): (20, 0.0), ("PSAGraph", "MCBO"): (0, 0.001), ("PSAGraph", "QMCBO"): (20, 0.0)}


def verify_qmcbo(root: Path) -> None:
    d = root / "qmcbo"
    missing = [f"{sub}trial_results_{m}_{env}_{s}.csv"
               for env in QMCBO_ENVS for m in ("MCBO", "QMCBO") for s in range(QMCBO_SEEDS)
               for sub in ("", "_e2/")
               if not (d / sub / f"trial_results_{m}_{env}_{s}.csv").exists()]
    check(not missing, f"qmcbo grid complete ({QMCBO_SEEDS*4} E1 + {QMCBO_SEEDS*4} E2); missing: {missing[:5]}")
    if missing:
        return
    if QMCBO_PUB is not None:
        for env, (pub_m, pub_q) in QMCBO_PUB.items():
            for method, pub in (("MCBO", pub_m), ("QMCBO", pub_q)):
                finals = [pd.read_csv(d / f"trial_results_{method}_{env}_{s}.csv")
                          .current_optimal.iloc[-1] for s in range(QMCBO_SEEDS)]
                m = float(np.mean(finals))
                check(close(m, pub, 2), f"qmcbo {env} {method} final {m:.4f} ~ {pub:.2f}")
    else:
        note("qmcbo published-final checks skipped (QMCBO_PUB not yet frozen)")
    # QMCBO must be E2-invariant on every seed: values (CSV) and, when the
    # engine-v3 sidecars exist, the chosen interventions X per iteration.
    for env in QMCBO_ENVS:
        n_id, n_dec, n_side = 0, 0, 0
        for s in range(QMCBO_SEEDS):
            e1 = pd.read_csv(d / f"trial_results_QMCBO_{env}_{s}.csv")
            e2 = pd.read_csv(d / "_e2" / f"trial_results_QMCBO_{env}_{s}.csv")
            n_id += int(np.array_equal(e1.current_optimal.values, e2.current_optimal.values))
            p1 = d / f"trial_results_QMCBO_{env}_{s}.decisions.json"
            p2 = d / "_e2" / f"trial_results_QMCBO_{env}_{s}.decisions.json"
            if p1.exists() and p2.exists():
                n_side += 1
                a, b = json.loads(p1.read_text()), json.loads(p2.read_text())
                n_dec += int([(it["X"], it["score"]) for it in a["iterations"]]
                             == [(it["X"], it["score"]) for it in b["iterations"]])
        check(n_id == QMCBO_SEEDS, f"QMCBO E2-invariant values on {env}: {n_id}/{QMCBO_SEEDS}")
        if n_side:
            check(n_dec == n_side, f"QMCBO E2-invariant decisions (X, score) on {env}: {n_dec}/{n_side}")
    if QMCBO_E2_IDENTICAL is not None:
        for (env, method), (pub_n, pub_delta) in QMCBO_E2_IDENTICAL.items():
            n_id, deltas = 0, []
            for s in range(QMCBO_SEEDS):
                e1 = pd.read_csv(d / f"trial_results_{method}_{env}_{s}.csv")
                e2 = pd.read_csv(d / "_e2" / f"trial_results_{method}_{env}_{s}.csv")
                n_id += int(np.array_equal(e1.current_optimal.values, e2.current_optimal.values))
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
