"""Re-score the benchmark's *shipped* baseline trajectories with the benchmark's
own GAP / PA-GAP scorer, at budgets 100/50/20.

These shipped CSVs are 5-seed runs covering 6 of the 8 hard-intervention datasets
and 6 of the 8 methods (no cCBO/HCBO; no chain/protein). They are NOT the survey's
20-seed Table 4 — we use this re-scoring only as an **exact cross-check** to catch
gross transcription errors in ``results/survey_table4.json`` (and as a reproducible
secondary reference).

Output: ``results/baseline_5seed_rescored.json``.

Run:  PYTHONPATH=. python scripts/rescore_shipped_baselines.py
"""

import os, sys, glob, json
import numpy as np
import pandas as pd

from ccbo import benchmark

sys.path.insert(0, benchmark.BENCH_ROOT)
from metrics.GAP import GAP            # noqa: E402
from metrics.PA_GAP import PA_GAP      # noqa: E402

METHODS = ['BO', 'CBO', 'CEO', 'CoCaBO', 'DCBO', 'MCBO']
DATASETS = ['toyGraph', 'synthetic', 'synthetic_2', 'chain',
            'ecology', 'protein', 'healthcare', 'epidemiology']
BUDGETS = (100, 50, 20)
RES = os.path.join(benchmark.BENCH_ROOT, 'results')
OUT = 'results/baseline_5seed_rescored.json'


def _sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def score(traj, ystar, task, n):
    t = traj[:n + 1]
    g = GAP(ystar); g.calculate_GAP(t, task)
    p = PA_GAP(ystar); p.calculate_PA_GAP(t, task)
    return g.GAP_value, p.PA_GAP_value


def main():
    out = {}
    for ds in DATASETS:
        try:
            ystar = benchmark.theoretical_best(ds)
            task = benchmark.load_config(ds)['task']
        except Exception as e:
            print(f"  skip {ds}: {e}")
            continue
        out[ds] = {'y_star': ystar, 'task': task, 'methods': {}}
        for m in METHODS:
            csvs = sorted(glob.glob(
                f"{RES}/{m}/{ds}/{m}_{ds}_100-trials_seed*_progress.csv"))
            if not csvs:
                continue
            per = {f'{k}{n}': [] for n in BUDGETS for k in ('gap', 'pagap')}
            for c in csvs:
                traj = pd.read_csv(c)['current_optimal'].tolist()
                for n in BUDGETS:
                    g, p = score(traj, ystar, task, n)
                    per[f'gap{n}'].append(g); per[f'pagap{n}'].append(p)
            out[ds]['methods'][m] = {
                'n_seeds': len(csvs),
                **{k: [float(np.mean(v)), _sem(v)] for k, v in per.items()}}
        print(f"{ds:13s} y*={ystar:+.3f} task={task:3s}  "
              f"methods={list(out[ds]['methods'])}")
        g = {m: out[ds]['methods'][m]['gap100'][0]
             for m in out[ds]['methods']}
        print("   GAP@100: " + "  ".join(f"{m}={v:.3f}" for m, v in g.items()))

    os.makedirs('results', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
