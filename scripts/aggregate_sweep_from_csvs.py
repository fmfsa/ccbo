"""Aggregate the coarsening-sweep results directly from the per-run CSVs already
on disk (so we can emit a comparison before the full sweep's final JSON lands).

Scores every completed QCBO_sweep trajectory (102-line CSVs) with the benchmark's
own GAP/PA-GAP at budgets 100/50/20, reconstructs the cidx->coarsening map via the
sweep's own deterministic selection, and writes results/qcbo_coarsening_sweep.json
in the same format scripts/emit_coarsening_sweep_table.py expects.

Run:  PYTHONPATH=. python scripts/aggregate_sweep_from_csvs.py [--min-seeds 1]
"""

import os, sys, glob, json, argparse
import numpy as np
import pandas as pd

from ccbo import benchmark
from ccbo.coarsening import enumerate_valid_coarsenings_manip
from scripts.run_qcbo_coarsening_sweep import (
    select_coarsenings, _partition_label, _clusters_of, BUDGETS)

sys.path.insert(0, benchmark.BENCH_ROOT)
from metrics.GAP import GAP            # noqa: E402
from metrics.PA_GAP import PA_GAP      # noqa: E402

DATASETS = ['toyGraph', 'synthetic', 'synthetic_2', 'chain',
            'ecology', 'protein', 'healthcare', 'epidemiology']
SWEEP_DIR = os.path.join(benchmark.BENCH_ROOT, 'results', 'QCBO_sweep')
OUT = 'results/qcbo_coarsening_sweep.json'
METRIC_NAMES = [f'{m}{n}' for n in BUDGETS for m in ('gap', 'pagap')] + ['final']


def _sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def score(traj, ystar, task, n):
    t = traj[:n + 1]
    g = GAP(ystar); g.calculate_GAP(t, task)
    p = PA_GAP(ystar); p.calculate_PA_GAP(t, task)
    return g.GAP_value, p.PA_GAP_value, t[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-seeds', type=int, default=1,
                    help='skip coarsenings with fewer completed seeds than this')
    ap.add_argument('--max-coarse', type=int, default=10)
    args = ap.parse_args()

    results = {}
    for ds in DATASETS:
        benchmark.load_graph(ds, seed=0)
        chosen = select_coarsenings(enumerate_valid_coarsenings_manip(ds), args.max_coarse)
        ystar = benchmark.theoretical_best(ds)
        task = benchmark.load_config(ds)['task']
        per_coarse = []
        for cidx, c in enumerate(chosen):
            csvs = sorted(glob.glob(f"{SWEEP_DIR}/{ds}/c{cidx}_seed*_100t.csv"))
            per = {m: [] for m in METRIC_NAMES}
            for cf in csvs:
                df = pd.read_csv(cf)
                if len(df) < 101:       # incomplete run
                    continue
                traj = df['current_optimal'].tolist()
                for n in BUDGETS:
                    g, p, fin = score(traj, ystar, task, n)
                    per[f'gap{n}'].append(g); per[f'pagap{n}'].append(p)
                    if n == max(BUDGETS):
                        per['final'].append(fin)
            if len(per['gap100']) < args.min_seeds:
                continue
            per_coarse.append({
                'cidx': cidx, 'label': _partition_label(c['partition']),
                'is_finest': _clusters_of(c['partition']) is None,
                'n_seeds': len(per['gap100']),
                'mean': {m: float(np.mean(per[m])) for m in METRIC_NAMES},
                'sem': {m: _sem(per[m]) for m in METRIC_NAMES}})
        agg = {}
        for m in METRIC_NAMES:
            vals = [pc['mean'][m] for pc in per_coarse]
            if vals:
                agg[m] = {'best': float(np.max(vals)), 'worst': float(np.min(vals)),
                          'avg': float(np.mean(vals)),
                          'best_label': per_coarse[int(np.argmax(vals))]['label'],
                          'worst_label': per_coarse[int(np.argmin(vals))]['label']}
        if per_coarse:
            results[ds] = {'y_star': ystar, 'task': task,
                           'n_coarsenings': len(per_coarse),
                           'per_coarsening': per_coarse, 'aggregate': agg}
            n_seeds = max(pc['n_seeds'] for pc in per_coarse)
            a = agg.get('gap100', {})
            print(f"  [{ds:13s}] {len(per_coarse)} coarsenings, up to {n_seeds} seeds | "
                  f"GAP@100 best={a.get('best', float('nan')):.3f} "
                  f"avg={a.get('avg', float('nan')):.3f} "
                  f"worst={a.get('worst', float('nan')):.3f}")
        else:
            print(f"  [{ds:13s}] no completed runs yet")

    os.makedirs('results', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT} ({len(results)} datasets)")


if __name__ == '__main__':
    main()
