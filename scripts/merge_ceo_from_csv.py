"""Aggregate completed CEO trajectories and merge the scored cells into
results/clusterbench10_misspec.json.

Reads ``CEO_{pid}_seed{s}.csv`` from the study RUNDIR for the 10 non-S1 pids
(P7 CSVs are copies of P0 -- the runner's alias; S1 is scored from P1 inside
the shared aggregate(), as for every method) and reuses run_misspec_fullfield's
aggregate() so CEO's cells are computed by exactly the same code path as the
existing methods'. Existing methods' cells are untouched.
"""
import os
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import run_misspec_fullfield as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--label", default="CEO")
    args = ap.parse_args()

    R.cb.register_variants()
    ystar = R.benchmark.theoretical_best(R.DS)
    task = R.benchmark.load_config(R.DS)["task"]
    pids = ["P0", "P1", "P2", "P3", "Pic", "P5", "P6", "P7", "S1", "S2", "S3"]
    pmeta = {p["id"]: p for p in R.cb.PERTURBATIONS}
    m = args.label

    traj = {m: {p: {} for p in pids}}
    for pid in pids:
        if pid == "S1":
            continue  # aliased to P1 by aggregate()
        for s in range(args.seeds):
            csv = os.path.join(R.RUNDIR, f"{m}_{pid}_seed{s}.csv")
            if not os.path.exists(csv):
                print(f"  MISSING {os.path.basename(csv)}")
                continue
            tr = pd.read_csv(csv)["current_optimal"].tolist()
            assert len(tr) >= args.trials + 1, \
                f"{os.path.basename(csv)} incomplete: {len(tr)} rows"
            traj[m][pid][s] = tr

    n_done = sum(len(v) for p, v in traj[m].items() if p != "S1")
    print(f"{m}: {n_done}/{(len(pids) - 1) * args.seeds} trajectories present")

    scored = R.aggregate(traj, [m], pids, pmeta, ystar, task, args)

    outpath = os.path.join(R.OUTDIR, "clusterbench10_misspec.json")
    with open(outpath) as f:
        full = json.load(f)
    full["methods"][m] = scored["methods"][m]
    with open(outpath, "w") as f:
        json.dump(full, f, indent=2)
    print(f"Merged {m} into {outpath}")


if __name__ == "__main__":
    main()
