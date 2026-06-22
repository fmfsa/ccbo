"""Parallel re-run of the QCBO benchmark suite.

Same protocol and output format as ``scripts/run_qcbo_benchmark_suite.py``
(writes ``results/qcbo_benchmark_results.json`` consumed by
``scripts/emit_benchmark_table.py``), but each (dataset, partition, seed) unit
runs in its own single-threaded process so the 60 units pack across cores
instead of one BLAS-saturated run hogging every core per GP fit. With the
corrected exploration-set cap (min(5, |M|)) the sequential suite is heavy;
this finishes in a fraction of the wall-clock.

Run:
    PYTHONPATH=. python scripts/run_qcbo_benchmark_parallel.py \
        --trials 100 --seeds 5 --jobs 30
"""

import os

# Cap BLAS/OpenMP threads to 1 *before* numpy is imported, so every worker is
# single-threaded and many units run in parallel. (Also makes fork-based
# multiprocessing safe: no OpenBLAS threadpool exists in the parent to fork.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ccbo import benchmark

sys.path.insert(0, benchmark.BENCH_ROOT)
from metrics.GAP import GAP            # noqa: E402
from metrics.PA_GAP import PA_GAP      # noqa: E402

# Curated subset -> coarse manipulable clusters (besides identity). Mirrors
# scripts/run_qcbo_benchmark_suite.py exactly.
COARSE = {
    'toyGraph':     [['X', 'Z']],
    'synthetic_2':  [['X', 'Z']],
    'synthetic':    [['B'], ['D', 'E']],
    'healthcare':   [['Aspirin', 'Statin']],
    'epidemiology': [['L', 'B']],
    'ecology':      [['C', 'N', 'O'], ['D', 'T']],
}
OUT = 'results/qcbo_benchmark_results.json'


def _sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def score(traj, ystar, task, n):
    t = traj[:n + 1]
    g = GAP(ystar); g.calculate_GAP(t, task)
    p = PA_GAP(ystar); p.calculate_PA_GAP(t, task)
    return g.GAP_value, p.PA_GAP_value, t[-1]


def run_unit(ds, label, clusters, seed, trials, ninit):
    """One (dataset, partition, seed) run; returns scored metrics."""
    csv = (f"{benchmark.BENCH_ROOT}/results/{label}/{ds}/"
           f"{label}_{ds}_{trials}-trials_seed{seed}_progress.csv")
    t0 = time.time()
    try:
        _, out = benchmark.run_qcbo_benchmark(
            ds, coarse_clusters=clusters, seed=seed, num_trials=trials,
            num_interventions=ninit, out_csv=csv, method_label=label)
        traj = pd.read_csv(out)['current_optimal'].tolist()
        ystar = benchmark.theoretical_best(ds)
        task = benchmark.load_config(ds)['task']
        g100, p100, fin = score(traj, ystar, task, trials)
        g50, _, _ = score(traj, ystar, task, 50)
        g20, _, _ = score(traj, ystar, task, 20)
        return dict(ds=ds, label=label, seed=seed, ok=True, secs=time.time() - t0,
                    final=fin, gap100=g100, pagap100=p100, gap50=g50, gap20=g20)
    except Exception as e:
        import traceback
        return dict(ds=ds, label=label, seed=seed, ok=False, secs=time.time() - t0,
                    err=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--ninit', type=int, default=5)
    ap.add_argument('--jobs', type=int, default=min(30, (os.cpu_count() or 8) - 2))
    args = ap.parse_args()

    os.makedirs('results', exist_ok=True)
    units = []
    for ds, coarse in COARSE.items():
        for label, clusters in [('QCBO-finest', None), ('QCBO-coarse', coarse)]:
            for s in range(args.seeds):
                units.append((ds, label, clusters, s, args.trials, args.ninit))

    print(f"parallel QCBO suite: {len(units)} units | trials={args.trials} "
          f"seeds={args.seeds} ninit={args.ninit} jobs={args.jobs}", flush=True)

    per = defaultdict(lambda: defaultdict(list))
    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_unit, *u) for u in units]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r['ok']:
                key = (r['ds'], r['label'])
                for m in ('final', 'gap100', 'pagap100', 'gap50', 'gap20'):
                    per[key][m].append(r[m])
                print(f"[{done:2d}/{len(units)}] OK   {r['ds']:13s} {r['label']:12s} "
                      f"seed{r['seed']} finalY={r['final']:+.3f} GAP={r['gap100']:.3f} "
                      f"({r['secs']:.0f}s)", flush=True)
            else:
                failures.append(r)
                print(f"[{done:2d}/{len(units)}] FAIL {r['ds']:13s} {r['label']:12s} "
                      f"seed{r['seed']} :: {r['err'].splitlines()[0]}", flush=True)

    # Aggregate into the suite's JSON format.
    results = {}
    for ds, coarse in COARSE.items():
        ystar = benchmark.theoretical_best(ds)
        task = benchmark.load_config(ds)['task']
        results[ds] = {'y_star': ystar, 'task': task, 'methods': {}}
        for label in ('QCBO-finest', 'QCBO-coarse'):
            key = (ds, label)
            if per[key]['final']:
                results[ds]['methods'][label] = {
                    m: [float(np.mean(v)), _sem(v)] for m, v in per[key].items()}
    with open(OUT, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"\nSaved {OUT}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for r in failures:
            print(f"  {r['ds']}/{r['label']}/seed{r['seed']}: {r['err'].splitlines()[0]}")
    print("=== PARALLEL SUITE DONE ===", flush=True)


if __name__ == '__main__':
    main()
