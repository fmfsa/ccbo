"""Parallel driver for the ClusterBench10 misspecification stress test.

Same protocol/output as ``scripts/run_misspec_fullfield.py`` (writes
``results/clusterbench10_misspec.json`` consumed by ``emit_misspec_table.py`` /
``plot_misspec_figures.py``), but fans each ``(method, perturbation, seed)`` unit
out to its own single-threaded process so the grid packs across cores instead of
one BLAS-saturated GP fit hogging every core.

BO is non-causal and therefore *invariant* to the structural perturbation, so we
run it only at ``P0`` (once per seed) and replicate that trajectory to every
perturbation (Delta == 0 by construction), avoiding redundant BO runs.

Run:
    PYTHONPATH=. python scripts/run_misspec_parallel.py \
        --seeds 10 --trials 100 --jobs 30
"""

import os

# Cap BLAS/OpenMP threads to 1 *before* numpy import so every worker is
# single-threaded and many units run in parallel (and fork stays safe).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import json
import time
import shutil
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Reuse the method runners + scoring from the serial driver (same directory).
import run_misspec_fullfield as drv
from ccbo import benchmark, clusterbench10 as cb

DS = drv.DS
RUNDIR = drv.RUNDIR
OUTDIR = drv.OUTDIR


def run_unit(method, pid, seed, trials, ninit):
    """One (method, perturbation, seed) run; returns its best-so-far trajectory.

    Resumable: if a complete per-run CSV already exists (>= trials points), it is
    reused instead of recomputed, so a relaunch only fills the missing units.
    """
    cb.register_variants()  # each worker registers the perturbation variants
    t0 = time.time()
    csv = os.path.join(RUNDIR, f"{method}_{pid}_seed{seed}.csv")
    if os.path.exists(csv):
        try:
            traj = pd.read_csv(csv)["current_optimal"].tolist()
            if len(traj) >= trials:
                return dict(method=method, pid=pid, seed=seed, ok=True, traj=traj,
                            secs=0.0, resumed=True)
        except Exception:
            pass
    try:
        traj = drv.METHODS[method](pid, seed, trials, ninit)
        return dict(method=method, pid=pid, seed=seed, ok=True, traj=traj,
                    secs=time.time() - t0)
    except Exception as e:
        import traceback
        return dict(method=method, pid=pid, seed=seed, ok=False,
                    err=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
                    secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--ninit", type=int, default=5)
    ap.add_argument("--methods", default=",".join(drv.METHODS))
    ap.add_argument("--perturbations",
                    default="P0,P1,P2,P3,Pic,P5,P6,P7,S1,S2,S3")
    ap.add_argument("--jobs", type=int,
                    default=min(30, (os.cpu_count() or 8) - 2))
    args = ap.parse_args()

    cb.register_variants()
    ystar = benchmark.theoretical_best(DS)
    task = benchmark.load_config(DS)["task"]
    methods = [m for m in args.methods.split(",") if m in drv.METHODS]
    pids = args.perturbations.split(",")
    pmeta = {p["id"]: p for p in cb.PERTURBATIONS}

    # Build the unit list. BO is invariant -> only run it at P0 (per seed).
    # S1 == P1 by construction -> scored from P1's runs (see drv.aggregate),
    # so it is never run as its own unit.
    units = []
    for m in methods:
        run_pids = ["P0"] if m == "BO" else [p for p in pids if p != "S1"]
        for pid in run_pids:
            for s in range(args.seeds):
                units.append((m, pid, s, args.trials, args.ninit))

    print(f"parallel misspec: {len(units)} units | methods={methods} "
          f"T={args.trials} seeds={args.seeds} jobs={args.jobs} "
          f"y*={ystar:.3f}", flush=True)

    traj = {m: {p: {} for p in pids} for m in methods}
    failures, done = [], 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_unit, *u) for u in units]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r["ok"]:
                traj[r["method"]][r["pid"]][r["seed"]] = r["traj"]
                print(f"[{done:3d}/{len(units)}] OK   {r['method']:12s} {r['pid']:4s} "
                      f"seed{r['seed']} finalY={r['traj'][-1]:+.3f} ({r['secs']:.0f}s)",
                      flush=True)
            else:
                failures.append(r)
                print(f"[{done:3d}/{len(units)}] FAIL {r['method']:12s} {r['pid']:4s} "
                      f"seed{r['seed']} :: {r['err'].splitlines()[0]}", flush=True)

    # BO is invariant: replicate its P0 trajectory to every perturbation and
    # mirror the per-run CSV so the plotting script finds BO_<pid>_seed*.csv.
    if "BO" in methods:
        for s, t in traj["BO"]["P0"].items():
            src = os.path.join(RUNDIR, f"BO_P0_seed{s}.csv")
            for pid in pids:
                traj["BO"][pid][s] = t
                dst = os.path.join(RUNDIR, f"BO_{pid}_seed{s}.csv")
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copyfile(src, dst)

    out = drv.aggregate(traj, methods, pids, pmeta, ystar, task, args)
    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, "clusterbench10_misspec.json")
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {outpath}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for r in failures:
            print(f"  {r['method']}/{r['pid']}/seed{r['seed']}: "
                  f"{r['err'].splitlines()[0]}")
    print("=== PARALLEL MISSPEC DONE ===", flush=True)


if __name__ == "__main__":
    main()
