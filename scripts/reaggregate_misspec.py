"""Re-aggregate the ClusterBench10 misspec JSON from the per-run CSV checkpoints.

Read-only: it does NOT run any optimization, so it is safe to run alongside an
active driver. It loads every ``{method}_{pid}_seed{seed}.csv`` in the run dir,
rebuilds the trajectory table, and calls ``run_misspec_fullfield.aggregate`` (the
single source of truth for scoring + the tolerance-based invariance check),
writing ``results/clusterbench10_misspec.json``.

Run:  PYTHONPATH=. python scripts/reaggregate_misspec.py [--seeds 10 --trials 100]
"""

import os
import glob
import json
import types
import argparse

import pandas as pd

import run_misspec_fullfield as drv
from ccbo import benchmark, clusterbench10 as cb

RUNDIR = drv.RUNDIR
OUTDIR = drv.OUTDIR
METHODS = ["BO", "CBO", "CEO", "QCBO-finest", "QCBO-coarse"]
PIDS = ["P0", "P1", "P2", "P3", "Pic", "P5", "P6", "P7", "S1", "S2", "S3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--ninit", type=int, default=5)
    args = ap.parse_args()

    cb.register_variants()
    ystar = benchmark.theoretical_best(drv.DS)
    task = benchmark.load_config(drv.DS)["task"]
    pmeta = {p["id"]: p for p in cb.PERTURBATIONS}

    traj = {m: {p: {} for p in PIDS} for m in METHODS}
    n_units = 0
    for m in METHODS:
        for pid in PIDS:
            for s in range(args.seeds):
                f = os.path.join(RUNDIR, f"{m}_{pid}_seed{s}.csv")
                if not os.path.exists(f):
                    continue
                try:
                    t = pd.read_csv(f)["current_optimal"].tolist()
                except Exception:
                    continue
                if len(t) >= args.trials:
                    traj[m][pid][s] = t
                    n_units += 1
    print(f"loaded {n_units} complete units from {RUNDIR}")
    # report per-method/pid seed coverage
    for m in METHODS:
        cov = {p: len(traj[m][p]) for p in PIDS}
        print(f"  {m:12s} " + " ".join(f"{p}:{cov[p]}" for p in PIDS))

    out = drv.aggregate(traj, METHODS, PIDS, pmeta, ystar, task, args)
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "clusterbench10_misspec.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
