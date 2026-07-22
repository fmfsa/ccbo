"""Run QCBO (finest = CBO; and coarse) on the curated benchmark
datasets, scored by the benchmark's own GAP / PA-GAP against its shipped y*.

Writes results/qcbo_benchmark_results.json and prints a summary table.

Run:  PYTHONPATH=. python scripts/run_qcbo_benchmark_suite.py [--trials 100] [--seeds 5]
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd

from ccbo import benchmark

sys.path.insert(0, benchmark.BENCH_ROOT)
from metrics.GAP import GAP
from metrics.PA_GAP import PA_GAP

# Curated subset: dataset -> coarse manipulable clusters (besides identity).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--ninit', type=int, default=5)
    args = ap.parse_args()

    os.makedirs('results', exist_ok=True)
    results = {}
    for ds, coarse in COARSE.items():
        ystar = benchmark.theoretical_best(ds)
        task = benchmark.load_config(ds)['task']
        results[ds] = {'y_star': ystar, 'task': task, 'methods': {}}
        print(f"\n{'='*70}\n{ds}  y*={ystar:.4f}  task={task}")
        for label, clusters in [('QCBO-finest', None), ('QCBO-coarse', coarse)]:
            rows = {'final': [], 'gap100': [], 'pagap100': [],
                    'gap50': [], 'gap20': []}
            for s in range(args.seeds):
                csv = (f"{benchmark.BENCH_ROOT}/results/{label}/{ds}/"
                       f"{label}_{ds}_{args.trials}-trials_seed{s}_progress.csv")
                try:
                    _, out = benchmark.run_qcbo_benchmark(
                        ds, coarse_clusters=clusters, seed=s,
                        num_trials=args.trials, num_interventions=args.ninit,
                        out_csv=csv, method_label=label)
                    traj = pd.read_csv(out)['current_optimal'].tolist()
                    g100, p100, fin = score(traj, ystar, task, args.trials)
                    g50, _, _ = score(traj, ystar, task, 50)
                    g20, _, _ = score(traj, ystar, task, 20)
                    rows['final'].append(fin); rows['gap100'].append(g100)
                    rows['pagap100'].append(p100); rows['gap50'].append(g50)
                    rows['gap20'].append(g20)
                except Exception as e:
                    import traceback
                    print(f"  {label} seed{s} FAILED: {e}")
                    traceback.print_exc()
            if rows['final']:
                summary = {k: [float(np.mean(v)), _sem(v)] for k, v in rows.items()}
                results[ds]['methods'][label] = summary
                print(f"  {label:12s}  finalY={summary['final'][0]:+.3f}"
                      f"±{summary['final'][1]:.3f}  "
                      f"GAP@100={summary['gap100'][0]:.3f}±{summary['gap100'][1]:.3f}  "
                      f"PA-GAP@100={summary['pagap100'][0]:.3f}±{summary['pagap100'][1]:.3f}")
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")
    print("=== SUITE DONE ===")


if __name__ == '__main__':
    main()
