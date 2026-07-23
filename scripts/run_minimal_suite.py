"""MinimalBench suite runner (Experiments A-C).

Units:
    ParallelParent x {A0,A1,A2,A3} x {CBO, QCBO} x seeds          (Experiment A)
    FrontDoor      x {B0,B1}       x {CBO, QCBO} x seeds          (Experiment B)
    MediatedChain  x {C0}          x {CBO, QCBO, HQCBO} x seeds   (Experiment C)

CBO = identity partition (full-DAG CBO by the singleton-identity result);
QCBO = the declared coarse partition; HQCBO = QCBO + plateau-triggered
refinement (native two-phase, ccbo/minimal_suite.py). The misspecified
conditions inject ``assumed_graph_name`` only; SEM/objective/data always
come from the true graph.

Persists one CSV per unit:
    <outdir>/<scm>_<cond>_<arm>_seed<seed>.csv
    columns: trial, best_y, cum_cost, arm, x_values
(arm/x_values are '' on observe rows, 'init' on row 0), plus
``<outdir>/refine_info.json`` with per-seed HQCBO trigger/split records.

HQCBO units are scheduled after the QCBO wave and read the matching QCBO
CSV for the plateau-trigger time (falling back to computing it themselves).

Run (from repo root):
    PYTHONPATH=. python scripts/run_minimal_suite.py \
        --seeds 30 --trials 50 --mc-trials 60 --outdir results/minimal --jobs 12

Resumable: a unit whose CSV already exists with the expected number of rows
is skipped.
"""

import os

# Cap BLAS/OpenMP threads to 1 *before* numpy import so every worker is
# single-threaded and many units run in parallel (and fork stays safe).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import csv
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


PP_CONDITIONS = ("A0", "A1", "A2", "A3")
FD_CONDITIONS = ("B0", "B1")
MC_CONDITIONS = ("C0",)
PP_ARMS = ("CBO", "QCBO")
FD_ARMS = ("CBO", "QCBO")
MC_ARMS = ("CBO", "QCBO", "HQCBO")


def _csv_path(outdir, scm, cond, arm, seed):
    return os.path.join(outdir, f"{scm}_{cond}_{arm}_seed{seed}.csv")


def _complete(csv_path, trials):
    if not os.path.exists(csv_path):
        return False
    try:
        return len(pd.read_csv(csv_path)) >= trials + 1
    except Exception:
        return False


def _write_rows(csv_path, rows):
    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "best_y", "cum_cost", "arm", "x_values"])
        for t, y, c, a, xv in rows:
            w.writerow([t, repr(float(y)), repr(float(c)), a, xv])
    os.replace(tmp, csv_path)


def run_unit(scm, cond, arm, seed, trials, num_interventions,
             initial_num_obs_samples, type_cost, outdir):
    t0 = time.time()
    csv_path = _csv_path(outdir, scm, cond, arm, seed)
    unit = dict(scm=scm, cond=cond, arm=arm, seed=seed, csv=csv_path)

    if _complete(csv_path, trials):
        unit.update(ok=True, resumed=True, secs=0.0)
        return unit

    try:
        from ccbo.minimal_suite import run_plain_unit, run_refine_unit

        if arm == "HQCBO":
            coarse_traj = None
            qcbo_csv = _csv_path(outdir, scm, cond, "QCBO", seed)
            if _complete(qcbo_csv, trials):
                coarse_traj = pd.read_csv(qcbo_csv)["best_y"] \
                    .astype(float).tolist()
            rows, info = run_refine_unit(
                scm, cond, seed, trials, num_interventions,
                initial_num_obs_samples, type_cost, coarse_traj=coarse_traj)
            unit.update(refine_info=info)
        else:
            rows, info = run_plain_unit(
                scm, cond, arm, seed, trials, num_interventions,
                initial_num_obs_samples, type_cost)
            unit.update(uninformative_arms=info["uninformative_arms"],
                        es=info["es"])

        _write_rows(csv_path, rows)
        unit.update(ok=True, resumed=False, secs=time.time() - t0,
                    final_y=rows[-1][1], final_cost=rows[-1][2])
        return unit
    except Exception as e:
        import traceback
        unit.update(ok=False, resumed=False, secs=time.time() - t0,
                    err=f"{type(e).__name__}: {e}\n"
                        f"{traceback.format_exc()[-800:]}")
        return unit


def main():
    ap = argparse.ArgumentParser(description="MinimalBench suite runner")
    ap.add_argument("--scms", default="ParallelParent,FrontDoor,MediatedChain")
    ap.add_argument("--seeds", type=int, default=30,
                    help="Number of seeds (runs seeds 0..N-1)")
    ap.add_argument("--trials", type=int, default=50,
                    help="Interventional budget for ParallelParent units")
    ap.add_argument("--mc-trials", type=int, default=60,
                    help="Interventional budget for MediatedChain units")
    # 3 initial interventional points per arm: on these 1-2D arms a larger
    # initial design already contains near-optimal points, hiding the role
    # of the (correct or corrupted) causal prior that Experiment B isolates.
    ap.add_argument("--num-interventions", type=int, default=3)
    ap.add_argument("--initial-obs", type=int, default=100)
    ap.add_argument("--type-cost", type=int, default=1)
    ap.add_argument("--outdir", default="results/minimal")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    scms = [s for s in args.scms.split(",") if s]
    os.makedirs(args.outdir, exist_ok=True)

    def _units(scm):
        if scm == "ParallelParent":
            return [(scm, c, a, s, args.trials)
                    for c in PP_CONDITIONS for a in PP_ARMS
                    for s in range(args.seeds)]
        if scm == "FrontDoor":
            return [(scm, c, a, s, args.trials)
                    for c in FD_CONDITIONS for a in FD_ARMS
                    for s in range(args.seeds)]
        if scm == "MediatedChain":
            return [(scm, c, a, s, args.mc_trials)
                    for c in MC_CONDITIONS for a in MC_ARMS
                    for s in range(args.seeds)]
        raise SystemExit(f"Unknown SCM {scm!r}")

    units = [u for scm in scms for u in _units(scm)]
    # HQCBO after everything else so the QCBO CSVs exist for trigger reuse.
    wave1 = [u for u in units if u[2] != "HQCBO"]
    wave2 = [u for u in units if u[2] == "HQCBO"]

    print(f"minimal suite: {len(units)} units "
          f"({len(wave1)} plain + {len(wave2)} refine) | scms={scms} "
          f"seeds={args.seeds} T={args.trials}/{args.mc_trials} "
          f"jobs={args.jobs}", flush=True)

    results, done = [], 0

    def _report(r):
        nonlocal done
        done += 1
        tag = f"{r['scm']}/{r['cond']}/{r['arm']}/seed{r['seed']}"
        if not r["ok"]:
            print(f"[{done:3d}/{len(units)}] FAIL {tag} :: "
                  f"{r['err'].splitlines()[0]}", flush=True)
        elif r.get("resumed"):
            print(f"[{done:3d}/{len(units)}] SKIP {tag}", flush=True)
        else:
            extra = ""
            if "refine_info" in r:
                ri = r["refine_info"]
                extra = (f" trigger={ri['trigger_step']} "
                         f"split={ri['split_clusters']}"
                         + (" REFUSED" if ri["refused"] else ""))
            uninf = r.get("uninformative_arms") or []
            if uninf:
                extra += f" [uninformative: {','.join(uninf)}]"
            print(f"[{done:3d}/{len(units)}] OK   {tag} "
                  f"finalY={r['final_y']:+.3f} cost={r['final_cost']:.0f} "
                  f"({r['secs']:.0f}s){extra}", flush=True)

    def _run_wave(wave):
        payload = [(scm, c, a, s, t, args.num_interventions,
                    args.initial_obs, args.type_cost, args.outdir)
                   for (scm, c, a, s, t) in wave]
        if args.jobs <= 1:
            for u in payload:
                r = run_unit(*u)
                results.append(r)
                _report(r)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(run_unit, *u) for u in payload]
                for f in as_completed(futs):
                    r = f.result()
                    results.append(r)
                    _report(r)

    _run_wave(wave1)
    _run_wave(wave2)

    # Persist HQCBO trigger/split records (merged over relaunches).
    info_path = os.path.join(args.outdir, "refine_info.json")
    merged = {}
    if os.path.exists(info_path):
        with open(info_path) as fh:
            merged = json.load(fh)
    for r in results:
        if r.get("ok") and "refine_info" in r:
            key = f"{r['scm']}_{r['cond']}_seed{r['seed']}"
            merged[key] = r["refine_info"]
    if merged:
        with open(info_path, "w") as fh:
            json.dump(merged, fh, indent=2, sort_keys=True)
        print(f"refine info -> {info_path}", flush=True)

    failures = [r for r in results if not r["ok"]]
    print(f"\ndone: {len(results) - len(failures)}/{len(results)} units OK; "
          f"CSVs in {args.outdir}", flush=True)
    if failures:
        for r in failures:
            print(f"  FAILED {r['scm']}/{r['cond']}/{r['arm']}/"
                  f"seed{r['seed']}: {r['err'].splitlines()[0]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
