"""Validate QCBO applicability on the curated benchmark datasets.

For each dataset: load the benchmark graph (reusing DiscoveredGraph), build
CoarsenedGraph at the finest (=CBO) and the chosen coarse partition, print the
MIS exploration set with per-arm prior tiers, and sanity-check that our
(noise-free) SEM evaluation at the benchmark's best intervention matches the
benchmark's reference optimum y*.

Run:  PYTHONPATH=. python scripts/validate_benchmark_qcbo.py
"""

import json
import numpy as np

from ccbo import benchmark
from ccbo.coarsened_graph import CoarsenedGraph

# Curated subset + the coarse partition (besides identity). 'Y' is the target.
PARTITIONS = {
    'toyGraph':     [['X', 'Z']],
    'synthetic_2':  [['X', 'Z']],
    'synthetic':    [['B'], ['D', 'E']],
    'healthcare':   [['Aspirin', 'Statin']],
    'epidemiology': [['L', 'B']],
    'ecology':      [['C', 'N', 'O'], ['D', 'T']],
}


def _part(clusters):
    return [frozenset(c) for c in clusters] + [frozenset({'Y'})]


def _finest(manip):
    return [frozenset({v}) for v in manip] + [frozenset({'Y'})]


def main():
    for ds, coarse_clusters in PARTITIONS.items():
        print("=" * 72)
        print(f"DATASET: {ds}")
        try:
            dg, obs, config = benchmark.load_graph(ds)
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            continue
        manip = list(config['intervention'])
        ystar = benchmark.theoretical_best(ds)
        print(f"  target=Y  task={config['task']}  manip={manip}  "
              f"y*={ystar:.4f}  n_obs={len(obs)}")

        # Exploration set = MIS of the (C-)DAG; the finest partition is the
        # Prop. 1 CBO anchor (same MIS rule on the full DAG).
        for label, partition in [('finest(=CBO)', _finest(manip)),
                                 ('coarse', _part(coarse_clusters))]:
            try:
                cg = CoarsenedGraph(dg, partition, ds, obs,
                                    num_mc_samples=200,
                                    assumed_graph_name=ds)
                es = cg.get_exploration_set_description()
                cg.get_all_do()  # populates identifiability info / estimators
                ident = cg.get_identifiability_summary()
                n_ident = sum(1 for v in ident.values() if v['identifiable'])
                n_info = sum(1 for v in ident.values() if not v['uninformative'])
                print(f"  [{label:12s}] {cg.get_partition_description()}")
                print(f"       ES={es}  ({n_info}/{len(ident)} with informative "
                      f"causal prior)")
            except Exception as e:
                import traceback
                print(f"  [{label}] BUILD FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()

        # Sanity: our noise-free SEM eval at the benchmark's best intervention.
        try:
            with open(benchmark._paths(ds)['ybest']) as f:
                tb = json.load(f)
            bi = tb.get('best_intervention')
            if isinstance(bi, dict):
                vars_, vals = list(bi.keys()), list(bi.values())
                y = benchmark.true_do_mc(dg, vars_, vals, n=4000)
                print(f"  MC E[Y|do(best_intervention)]={y:.4f}  (y*={ystar:.4f})")
        except Exception as e:
            print(f"  (best_intervention check skipped: {type(e).__name__})")
    print("=" * 72)


if __name__ == '__main__':
    main()
