"""QCBO across *all valid coarsenings* of each benchmark dataset's true DAG.

For every hard-intervention dataset we take the benchmark's true causal graph,
enumerate every valid coarsening of the manipulable set
(``coarsening.enumerate_valid_coarsenings_manip``), select up to ``--max-coarse``
of them (all if fewer; otherwise finest + coarsest + an even spread by fineness),
and run QCBO on each over ``--seeds`` seeds. Each best-so-far trajectory is scored
with the **benchmark's own** GAP / PA-GAP (so the numbers are directly comparable
to the survey's Table 4) at budgets 100/50/20.

We then report, per dataset, QCBO's **best / worst / average** metric across the
selected coarsenings — characterizing the whole method rather than one
hand-picked partition.

Output: ``results/qcbo_coarsening_sweep.json`` (consumed by
``scripts/emit_coarsening_sweep_table.py``).

Run:
    PYTHONPATH=. python scripts/run_qcbo_coarsening_sweep.py \
        --trials 100 --seeds 20 --max-coarse 10 --jobs 30
"""

import os

# Single-thread BLAS so many units pack across cores (see run_qcbo_benchmark_parallel).
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
from ccbo.coarsening import enumerate_valid_coarsenings_manip

sys.path.insert(0, benchmark.BENCH_ROOT)
from metrics.GAP import GAP            # noqa: E402
from metrics.PA_GAP import PA_GAP      # noqa: E402

# Survey's 8 hard-intervention datasets (benchmark config names).
DATASETS = ['toyGraph', 'synthetic', 'synthetic_2', 'chain',
            'ecology', 'protein', 'healthcare', 'epidemiology']
BUDGETS = (100, 50, 20)
OUT = 'results/qcbo_coarsening_sweep.json'


def _sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def score(traj, ystar, task, n):
    """GAP and PA-GAP at budget n (trajectory truncated to n post-init points)."""
    t = traj[:n + 1]
    g = GAP(ystar); g.calculate_GAP(t, task)
    p = PA_GAP(ystar); p.calculate_PA_GAP(t, task)
    return g.GAP_value, p.PA_GAP_value, t[-1]


def _partition_label(partition):
    manip = [p for p in partition if p != frozenset({'Y'})]
    return '|'.join('{' + ','.join(sorted(p)) + '}'
                    for p in sorted(manip, key=lambda c: sorted(c)))


def _clusters_of(partition):
    """Convert an enumerate partition (frozensets incl {Y}) to run_qcbo_benchmark's
    ``coarse_clusters`` arg; finest (all singletons) -> None."""
    manip = [sorted(p) for p in partition if p != frozenset({'Y'})]
    if all(len(c) == 1 for c in manip):
        return None      # identity == finest
    return manip


def select_coarsenings(cs, k):
    """All if <=k; else finest + coarsest + an even spread by fineness level."""
    if len(cs) <= k:
        return list(cs)
    cs_sorted = sorted(cs, key=lambda c: -c['num_parts'])  # finest first
    finest, coarsest, middle = cs_sorted[0], cs_sorted[-1], cs_sorted[1:-1]
    idx = sorted(set(np.linspace(0, len(middle) - 1, k - 2).round().astype(int).tolist()))
    return [finest] + [middle[i] for i in idx] + [coarsest]


def build_units(datasets, max_coarse, seeds, trials, ninit):
    """List of (ds, cidx, label, clusters, seed, trials, ninit) work units."""
    units, plan = [], {}
    for ds in datasets:
        benchmark.load_graph(ds, seed=0)          # registers the DAG
        cs = enumerate_valid_coarsenings_manip(ds)
        chosen = select_coarsenings(cs, max_coarse)
        plan[ds] = []
        for cidx, c in enumerate(chosen):
            label = _partition_label(c['partition'])
            clusters = _clusters_of(c['partition'])
            plan[ds].append({'cidx': cidx, 'label': label,
                             'is_finest': clusters is None})
            for s in range(seeds):
                units.append((ds, cidx, label, clusters, s, trials, ninit))
    return units, plan


def run_unit(ds, cidx, label, clusters, seed, trials, ninit):
    out_csv = (f"{benchmark.BENCH_ROOT}/results/QCBO_sweep/{ds}/"
               f"c{cidx}_seed{seed}_{trials}t.csv")
    t0 = time.time()
    try:
        _, out = benchmark.run_qcbo_benchmark(
            ds, coarse_clusters=clusters, seed=seed, num_trials=trials,
            num_interventions=ninit, out_csv=out_csv, method_label='QCBOsweep')
        traj = pd.read_csv(out)['current_optimal'].tolist()
        ystar = benchmark.theoretical_best(ds)
        task = benchmark.load_config(ds)['task']
        rec = dict(ds=ds, cidx=cidx, seed=seed, ok=True, secs=time.time() - t0)
        for n in BUDGETS:
            g, p, fin = score(traj, ystar, task, n)
            rec[f'gap{n}'] = g
            rec[f'pagap{n}'] = p
            if n == max(BUDGETS):
                rec['final'] = fin
        return rec
    except Exception as e:
        import traceback
        return dict(ds=ds, cidx=cidx, seed=seed, ok=False, secs=time.time() - t0,
                    err=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--ninit', type=int, default=5)
    ap.add_argument('--max-coarse', type=int, default=10)
    ap.add_argument('--datasets', nargs='*', default=DATASETS)
    ap.add_argument('--jobs', type=int, default=min(30, (os.cpu_count() or 8) - 2))
    args = ap.parse_args()

    os.makedirs('results', exist_ok=True)
    units, plan = build_units(args.datasets, args.max_coarse, args.seeds,
                              args.trials, args.ninit)
    n_coarse = sum(len(v) for v in plan.values())
    print(f"coarsening sweep: {len(args.datasets)} datasets, {n_coarse} coarsenings, "
          f"{len(units)} units | trials={args.trials} seeds={args.seeds} "
          f"jobs={args.jobs}", flush=True)
    for ds in args.datasets:
        print(f"  {ds:13s}: {len(plan[ds])} coarsenings -> "
              f"{[c['label'] for c in plan[ds]]}", flush=True)

    # metrics[(ds,cidx)][metric] = [per-seed values]
    metrics = defaultdict(lambda: defaultdict(list))
    failures, done = [], 0
    metric_names = [f'{m}{n}' for n in BUDGETS for m in ('gap', 'pagap')] + ['final']
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_unit, *u) for u in units]
        for f in as_completed(futs):
            r = f.result(); done += 1
            if r['ok']:
                for m in metric_names:
                    metrics[(r['ds'], r['cidx'])][m].append(r[m])
                print(f"[{done:3d}/{len(units)}] OK   {r['ds']:12s} c{r['cidx']} "
                      f"seed{r['seed']:<2d} GAP@100={r['gap100']:.3f} "
                      f"finalY={r['final']:+.3f} ({r['secs']:.0f}s)", flush=True)
            else:
                failures.append(r)
                print(f"[{done:3d}/{len(units)}] FAIL {r['ds']:12s} c{r['cidx']} "
                      f"seed{r['seed']} :: {r['err'].splitlines()[0]}", flush=True)

    # Aggregate: per-coarsening mean-over-seeds, then best/worst/avg over coarsenings.
    results = {}
    for ds in args.datasets:
        ystar = benchmark.theoretical_best(ds)
        task = benchmark.load_config(ds)['task']
        per_coarse = []
        for c in plan[ds]:
            key = (ds, c['cidx'])
            if not metrics[key]['gap100']:
                continue
            means = {m: float(np.mean(metrics[key][m])) for m in metric_names}
            means_sem = {m: _sem(metrics[key][m]) for m in metric_names}
            per_coarse.append({'cidx': c['cidx'], 'label': c['label'],
                               'is_finest': c['is_finest'],
                               'n_seeds': len(metrics[key]['gap100']),
                               'mean': means, 'sem': means_sem})
        agg = {}
        for m in metric_names:
            vals = [pc['mean'][m] for pc in per_coarse]
            if not vals:
                continue
            # GAP/PA-GAP: higher is better; 'final' depends on task.
            agg[m] = {'best': float(np.max(vals)), 'worst': float(np.min(vals)),
                      'avg': float(np.mean(vals)),
                      'best_label': per_coarse[int(np.argmax(vals))]['label'],
                      'worst_label': per_coarse[int(np.argmin(vals))]['label']}
        results[ds] = {'y_star': ystar, 'task': task,
                       'n_coarsenings': len(per_coarse),
                       'per_coarsening': per_coarse, 'aggregate': agg}
        with open(OUT, 'w') as fh:
            json.dump(results, fh, indent=2)
        if per_coarse:
            a = agg.get('gap100', {})
            print(f"  [{ds}] {len(per_coarse)} coarsenings  GAP@100 "
                  f"best={a.get('best', float('nan')):.3f} "
                  f"avg={a.get('avg', float('nan')):.3f} "
                  f"worst={a.get('worst', float('nan')):.3f}", flush=True)

    print(f"\nSaved {OUT}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for r in failures[:20]:
            print(f"  {r['ds']}/c{r['cidx']}/seed{r['seed']}: {r['err'].splitlines()[0]}")
    print("=== COARSENING SWEEP DONE ===", flush=True)


if __name__ == '__main__':
    main()
