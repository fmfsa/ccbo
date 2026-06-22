"""Run the CausalBO_Benchmark head-to-head baselines on the curated subset.

For each (method, dataset) pair we run the benchmark's own shell runner once per
seed, then archive the seed-agnostic progress CSV
(`results/<DIR>/<ds>/<DIR>_<ds>_<T>-trials_progress.csv`) to a seed-tagged path
(`..._seed<S>_progress.csv`, mirroring the QCBO suite) so seeds are not
clobbered. Scoring + table emission is done separately by
`scripts/emit_baseline_table.py`.

Pairs run in parallel (each worker serialises its own seeds, so the shared
seed-agnostic CSV is never raced); BLAS threads are capped per worker.

Launch AFTER the primary 100-trial run lands (it frees ~22 cores). Example:

    PYTHONUNBUFFERED=1 python scripts/run_baselines_suite.py \
        --trials 100 --seeds 5 --jobs 6 2>&1 | tee results/baselines_suite.log

Smoke (fast):

    python scripts/run_baselines_suite.py --trials 5 --seeds 1 \
        --methods BO,CBO --datasets toyGraph --jobs 1
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "third_party", "CausalBO_Benchmark")
H = "scripts/hard_internvention"

# method -> (results dir name, argv builder given trials T, seed S, dataset ds,
#            extra env). The seed-agnostic progress CSV is always
#            results/<dir>/<ds>/<dir>_<ds>_<T>-trials_progress.csv.
METHODS = {
    "BO":     dict(dir="BO",     env={},
                   argv=lambda T, S, ds: ["sh", f"{H}/run_bo_cbo.sh", "BO", str(T), str(S), ds]),
    "CBO":    dict(dir="CBO",    env={},
                   argv=lambda T, S, ds: ["sh", f"{H}/run_bo_cbo.sh", "CBO", str(T), str(S), ds]),
    "CEO":    dict(dir="CEO",    env={},
                   argv=lambda T, S, ds: ["sh", f"{H}/run_ceo.sh", str(T), str(S), ds]),
    "CoCaBO": dict(dir="CoCaBO", env={},
                   argv=lambda T, S, ds: ["sh", f"{H}/run_cocabo.sh", str(T), str(S), ds]),
    "DCBO":   dict(dir="DCBO",   env={},
                   argv=lambda T, S, ds: ["sh", f"{H}/run_dcbo.sh", str(T), str(S), ds]),
    # MCBO's runner cd's into baselines/mcbo but never puts it on sys.path.
    "MCBO":   dict(dir="MCBO",
                   env={"PYTHONPATH": os.path.join(BENCH, "baselines", "mcbo"),
                        "WANDB_MODE": "offline"},
                   argv=lambda T, S, ds: ["sh", f"{H}/run_mcbo.sh", str(T), str(S), ds]),
}

CURATED = ["toyGraph", "synthetic_2", "synthetic", "healthcare", "epidemiology", "ecology"]


def progress_csv(dir_, ds, T, seed=None):
    base = f"results/{dir_}/{ds}/{dir_}_{ds}_{T}-trials"
    return os.path.join(BENCH, base + (f"_seed{seed}_progress.csv" if seed is not None else "_progress.csv"))


def run_pair(method, ds, trials, seeds, threads, timeout):
    spec = METHODS[method]
    out = []
    for S in seeds:
        env = os.environ.copy()
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = str(threads)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(spec["env"])
        argv = spec["argv"](trials, S, ds)
        t0 = time.time()
        try:
            p = subprocess.run(argv, cwd=BENCH, env=env, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            rc, tail = p.returncode, p.stdout.decode("utf-8", "replace")[-600:]
        except subprocess.TimeoutExpired:
            rc, tail = 124, "TIMEOUT"
        dt = time.time() - t0
        src = progress_csv(spec["dir"], ds, trials)
        dst = progress_csv(spec["dir"], ds, trials, S)
        archived = False
        if rc == 0 and os.path.exists(src):
            shutil.copyfile(src, dst)
            archived = os.path.exists(dst)
        status = "OK" if (rc == 0 and archived) else ("FAIL" if rc == 0 else f"FAIL(rc={rc})")
        rec = dict(method=method, dataset=ds, seed=S, rc=rc, archived=archived,
                   status=status, seconds=round(dt, 1), csv=os.path.relpath(dst, BENCH))
        out.append(rec)
        print(f"[{method:6s} {ds:13s} seed{S}] {status:10s} {dt:6.1f}s"
              + ("" if archived else f"  :: {tail.strip()[-200:]}"), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0..n-1)")
    ap.add_argument("--seed-list", type=str, default="", help="explicit comma seeds, overrides --seeds")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--datasets", type=str, default=",".join(CURATED))
    ap.add_argument("--jobs", type=int, default=6, help="parallel (method,dataset) workers")
    ap.add_argument("--threads", type=int, default=0, help="BLAS threads/worker (0=auto from jobs)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-run timeout (s)")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    datasets = [d for d in args.datasets.split(",") if d]
    seeds = ([int(x) for x in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))
    ncpu = os.cpu_count() or 8
    threads = args.threads or max(1, ncpu // max(1, args.jobs))

    for m in methods:
        if m not in METHODS:
            sys.exit(f"unknown method {m}; known: {list(METHODS)}")

    pairs = [(m, d) for m in methods for d in datasets]
    print(f"baselines suite: {len(pairs)} (method,dataset) pairs x {len(seeds)} seeds "
          f"= {len(pairs)*len(seeds)} runs | trials={args.trials} jobs={args.jobs} "
          f"threads/worker={threads}", flush=True)

    manifest = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_pair, m, d, args.trials, seeds, threads, args.timeout): (m, d)
                for (m, d) in pairs}
        for fut in cf.as_completed(futs):
            manifest.extend(fut.result())
            with open(os.path.join(REPO, "results", "baselines_run_manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

    ok = sum(1 for r in manifest if r["status"] == "OK")
    print(f"\nBASELINES SUITE DONE: {ok}/{len(manifest)} runs OK", flush=True)
    bad = [f"{r['method']}/{r['dataset']}/s{r['seed']}({r['status']})"
           for r in manifest if r["status"] != "OK"]
    if bad:
        print("FAILURES:", ", ".join(bad), flush=True)


if __name__ == "__main__":
    main()
