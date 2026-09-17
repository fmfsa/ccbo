"""Predeclared fixed-query integration sensitivity; never runs an optimizer."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from ccbo.qdcbo.runner import get_setup, sample_observations, FINEST, PARTITIONS
from ccbo.qdcbo.quotient_dbn import build_quotient_dbn
from ccbo.qdcbo.coherent import fit_conditional_mechanisms, integrate_predictive

INTEGRATION_SEEDS = (31337, 27183, 16180, 14142)
THRESHOLDS = dict(mean_normalized_rms=.05, mean_normalized_max=.15,
                  variance_relative_rms=.10, variance_relative_max=.30)

def check(setup, resolution, low=256, high=1024):
    start = time.monotonic()
    sem, graph, _, _, target, changes = get_setup(setup, 3)
    observations = sample_observations(sem, changes, 3, 100, 1001)
    partition = FINEST if resolution == 'fine' else PARTITIONS[setup]
    spec = build_quotient_dbn(graph, partition, target)
    models = fit_conditional_mechanisms(spec, observations)
    rows = []
    # Same fixed histories and candidate points at both model resolutions.
    # Null is a numerical diagnostic query, not a new eligible optimizer arm.
    for history in ('natural', 'joint-fixed'):
        for candidate in (None, (-2., -1.), (0., 0.), (1., 2.)):
            blanket = {v: [None]*3 for v in ('X', 'Z', 'Y')}
            if history == 'joint-fixed':
                blanket['X'][:2] = [-1., -1.]
                blanket['Z'][:2] = [.5, .5]
            if candidate is not None:
                blanket['X'][2], blanket['Z'][2] = candidate
            for seed in INTEGRATION_SEEDS:
                a = integrate_predictive(spec, models, 2, deepcopy(blanket), low, seed)
                b = integrate_predictive(spec, models, 2, deepcopy(blanket), high, seed)
                rows.append(dict(history=history, candidate=candidate, seed=seed,
                    low_mean=a[0], high_mean=b[0], low_variance=a[1], high_variance=b[1],
                    mean_error=(a[0]-b[0])/max(1., np.sqrt(b[1])),
                    variance_error=(a[1]-b[1])/max(1., b[1])))
    mean = np.array([r['mean_error'] for r in rows])
    variance = np.array([r['variance_error'] for r in rows])
    metrics = dict(mean_normalized_rms=float(np.sqrt(np.mean(mean**2))),
                   mean_normalized_max=float(np.max(np.abs(mean))),
                   variance_relative_rms=float(np.sqrt(np.mean(variance**2))),
                   variance_relative_max=float(np.max(np.abs(variance))))
    return dict(schema=1, setup=setup, resolution=resolution, observation_seed=1001,
        n_obs=100, low=low, high=high, integration_seeds=INTEGRATION_SEEDS,
        thresholds=THRESHOLDS, metrics=metrics,
        passed=all(np.isfinite(v) and v <= THRESHOLDS[k] for k,v in metrics.items()),
        interpretation='Finite fixed-grid sensitivity only; neither high-particle truth nor global convergence is certified.',
        rows=rows, wall_seconds=time.monotonic()-start)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--setup', required=True, choices=['stat','ind','nonstat_regularized'])
    p.add_argument('--resolution', required=True, choices=['fine','quotient'])
    p.add_argument('--low', type=int, default=256)
    p.add_argument('--high', type=int, default=1024)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.low < 2 or a.high <= a.low:
        p.error('Require 2 <= low < high')
    result = check(a.setup, a.resolution, a.low, a.high)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('setup','resolution','passed','metrics','wall_seconds')}))
    # A failed scientific tolerance is preserved as a complete diagnostic,
    # not confused with an execution failure. Manifest acceptance reads passed.
