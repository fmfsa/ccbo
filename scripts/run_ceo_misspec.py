"""Run CEO (authors' stack) under the ClusterBench10 misspecification protocol.

Writes ``CEO_{pid}_seed{s}.csv`` (trial_number, current_optimal) into the
study's RUNDIR plus a ``.meta.json`` sidecar (pool composition, ES mode, wall
time, final graph posterior). Merge into results/clusterbench10_misspec.json
with scripts/merge_ceo_from_csv.py.

Aliases (see ceo_adapter docstring): P7 has the same observable edge set as
P0, so its CSVs are copies of the completed P0 run; S1 is never run anywhere
(aggregate() scores it from P1).

Run (one unit):    PYTHONPATH=. python scripts/run_ceo_misspec.py --pid P1 --seed 3
Run (local sweep): PYTHONPATH=. python scripts/run_ceo_misspec.py --pid all --seeds 0-9 --jobs 4
LSF array:         scripts/lsf/submit_ceo_misspec.sh
"""

import os
import sys
import json
import shutil
import argparse

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ccbo import benchmark, clusterbench10 as cb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ceo_adapter

RUNDIR = os.path.join(benchmark.BENCH_ROOT, "results", "_misspec")

# 9 pids actually run; P7 <- P0 (identical observable edges for CEO),
# S1 <- P1 inside aggregate() as for every method.
RUN_PIDS = ["P0", "P1", "P2", "P3", "Pic", "P5", "P6", "S2", "S3"]


def csv_path(label, pid, seed):
    return os.path.join(RUNDIR, f"{label}_{pid}_seed{seed}.csv")


def complete(path, trials):
    if not os.path.exists(path):
        return False
    try:
        return len(pd.read_csv(path)["current_optimal"]) >= trials + 1
    except Exception:
        return False


def run_unit(pid, seed, args):
    out = csv_path(args.label, pid, seed)
    if complete(out, args.trials):
        print(f"[skip] {os.path.basename(out)} complete")
        return
    os.makedirs(RUNDIR, exist_ok=True)
    print(f"[run ] {args.label} pid={pid} seed={seed} pool={args.pool} "
          f"es={args.es} trials={args.trials}", flush=True)
    traj, meta = ceo_adapter.run_ceo_once(
        pid, seed, args.trials, args.ninit, pool=args.pool, es_mode=args.es,
        noise_scale=args.eval_noise)
    pd.DataFrame({"trial_number": list(range(len(traj))),
                  "current_optimal": traj}).to_csv(out, index=False)
    with open(out.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {os.path.basename(out)} finalY={traj[args.trials]:+.4f} "
          f"wall={meta['wall_seconds']:.0f}s posterior={meta['final_posterior']}",
          flush=True)


def alias_p7(seed, args):
    """P7 == P0 for CEO (observable edge sets coincide); copy the CSVs."""
    src = csv_path(args.label, "P0", seed)
    dst = csv_path(args.label, "P7", seed)
    if complete(src, args.trials) and not complete(dst, args.trials):
        shutil.copyfile(src, dst)
        m = src.replace(".csv", ".meta.json")
        if os.path.exists(m):
            shutil.copyfile(m, dst.replace(".csv", ".meta.json"))
        print(f"[alias] P7 <- P0 seed{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="all",
                    help="perturbation id, comma list, or 'all'")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="range A-B (inclusive)")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--ninit", type=int, default=5)
    ap.add_argument("--pool", choices=["hedge", "committed"], default="hedge")
    ap.add_argument("--es", choices=["full", "cluster", "cap2"],
                    default="cluster")
    ap.add_argument("--eval-noise", type=float, default=0.0)
    ap.add_argument("--label", default="CEO")
    ap.add_argument("--jobs", type=int, default=1,
                    help="local process-parallel fallback over (pid, seed)")
    args = ap.parse_args()

    cb.register_variants()
    if args.pid == "all":
        pids = list(RUN_PIDS)
    else:
        pids = [p for p in args.pid.split(",") if p]
        bad = set(pids) - set(RUN_PIDS)
        if bad:
            ap.error(f"pids {sorted(bad)} not runnable (P7/S1 are aliases); "
                     f"choose from {RUN_PIDS}")
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = list(range(10))

    units = [(p, s) for p in pids for s in seeds]
    if args.jobs > 1:
        # Spawn one subprocess per unit (CEO pollutes global RNG state, so
        # in-process parallelism would break per-unit determinism).
        from concurrent.futures import ProcessPoolExecutor
        import subprocess

        def spawn(unit):
            p, s = unit
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--pid", p, "--seed", str(s), "--trials", str(args.trials),
                   "--ninit", str(args.ninit), "--pool", args.pool,
                   "--es", args.es, "--eval-noise", str(args.eval_noise),
                   "--label", args.label]
            return subprocess.call(cmd, cwd=ceo_adapter.REPO_ROOT)

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rcs = list(ex.map(spawn, units))
        if any(rcs):
            sys.exit(1)
    else:
        for p, s in units:
            run_unit(p, s, args)

    for s in seeds:
        alias_p7(s, args)


if __name__ == "__main__":
    main()
