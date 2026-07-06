"""Re-run ONLY the QCBO-wrongpi method (fixed acyclic wrong C-DAG) and merge its
scored cells into results/clusterbench10_misspec.json without disturbing the
other four methods.

Rationale: QCBO's contract is C-DAG-only, so the wrong-partition arm asserts one
fixed acyclic wrong C-DAG (base structure, wrong partition) and is scored against
the true SEM. It never re-derives a quotient from a perturbed fine DAG, so it
never "refuses" -- it just pays a constant suboptimality. See _wrongpi_runner in
run_misspec_fullfield.py. Cheap: one QCBO run per seed (perturbation-invariant).

Run:
  PYTHONPATH=.:scripts conda run -n ccbo python scripts/run_wrongpi_fixed.py \
      --seeds 10 --trials 100
"""
import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import run_misspec_fullfield as R
from ccbo import benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--ninit", type=int, default=5)
    ap.add_argument("--cap", type=int, default=5)
    args = ap.parse_args()

    R.cb.register_variants()
    ystar = benchmark.theoretical_best(R.DS)
    task = benchmark.load_config(R.DS)["task"]
    pids = ["P0", "P1", "P2", "P3", "Pic", "P5", "P6", "S1", "S2", "S3"]
    pmeta = {p["id"]: p for p in R.cb.PERTURBATIONS}

    m = "QCBO-wrongpi"
    runner = R.METHODS[m]
    traj = {m: {p: {} for p in pids}}
    for pid in pids:
        if pid == "S1":
            continue
        for s in range(args.seeds):
            traj[m][pid][s] = runner(pid, s, args.cap, args.trials, args.ninit)
            print(f"  {m}/{pid}/seed{s} finalY={traj[m][pid][s][-1]:+.3f}",
                  flush=True)

    scored = R.aggregate(traj, [m], pids, pmeta, ystar, task, args)

    outpath = os.path.join(R.OUTDIR, "clusterbench10_misspec.json")
    with open(outpath) as f:
        full = json.load(f)
    full["methods"][m] = scored["methods"][m]
    with open(outpath, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nMerged {m} into {outpath}")


if __name__ == "__main__":
    main()
