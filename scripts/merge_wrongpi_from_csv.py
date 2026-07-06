"""Aggregate the 10 completed QCBO-wrongpi P0 trajectories from disk and merge
the scored cells into results/clusterbench10_misspec.json.

Bypasses the live driver: the wrong-partition arm is perturbation-invariant (one
fixed acyclic wrong C-DAG), so its trajectory for every perturbation id is the
same completed P0 run. We read those 10 CSVs and reuse the shared aggregate().
"""
import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import run_misspec_fullfield as R
from ccbo import benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--cap", type=int, default=5)
    args = ap.parse_args()

    R.cb.register_variants()
    ystar = benchmark.theoretical_best(R.DS)
    task = benchmark.load_config(R.DS)["task"]
    pids = ["P0", "P1", "P2", "P3", "Pic", "P5", "P6", "S1", "S2", "S3"]
    pmeta = {p["id"]: p for p in R.cb.PERTURBATIONS}
    m = "QCBO-wrongpi"

    # Read the completed per-seed trajectories.
    seed_traj = {}
    for s in range(args.seeds):
        csv = os.path.join(R.RUNDIR, f"{m}_P0_seed{s}.csv")
        tr = pd.read_csv(csv)["current_optimal"].tolist()
        assert len(tr) >= args.trials + 1, f"seed{s} incomplete: {len(tr)} rows"
        seed_traj[s] = tr
        print(f"  seed{s}: {len(tr)-1} trials, finalY={tr[args.trials]:+.4f}")

    # Perturbation-invariant: same trajectory under every perturbation id.
    traj = {m: {p: {s: seed_traj[s] for s in range(args.seeds)} for p in pids}}

    scored = R.aggregate(traj, [m], pids, pmeta, ystar, task, args)

    outpath = os.path.join(R.OUTDIR, "clusterbench10_misspec.json")
    with open(outpath) as f:
        full = json.load(f)
    full["methods"][m] = scored["methods"][m]
    with open(outpath, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nMerged {m} ({args.seeds} seeds) into {outpath}")


if __name__ == "__main__":
    main()
