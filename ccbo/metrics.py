"""Standardized efficiency metrics for CBO trajectories: GAP and PA-GAP.

These follow the definitions in the CBO survey of Anonymous (TMLR 2026),
"Causal Bayesian Optimization: Foundations, Methods, and Applications"
(GAP originally from DCBO, Aglietti et al. 2021), so our numbers are
comparable to that benchmark's common scoring protocol.

Both metrics are computed from a best-so-far trajectory ``traj`` (length
``T + 1``: one initial incumbent ``traj[0]`` followed by ``T`` post-trial
best-so-far values) and a reference optimum ``y_star`` (the offline oracle
optimum of the benchmark). Larger is better for both.

Definitions (minimization shown; maximization flips the improvement sign)
------------------------------------------------------------------------
Improvement ratio at trial ``t`` (clipped to [0, 1]):

    R_t = min( (y_init - traj[t]) / (y_init - y_star), 1 )

GAP combines final improvement with discovery speed (t* = first trial that
attains the final ratio R_T; t* = T if no improvement):

    GAP = ( R_T + (T - t*) / T ) / ( 1 + (T - 1) / T )       in [0, 1]

PA-GAP weights every trial's improvement by a decaying efficiency factor:

    PA-GAP = (1/T) * sum_{t=1..T} R_t * (T - (t - 1)) / T     in [0, (T+1)/(2T)]
"""

import numpy as np

# Offline oracle optima (reference y*) for benchmarks where we know them
# exactly (see scripts/*_arm_values.py). Benchmarks not listed fall back to
# the best value observed across the compared trajectories.
Y_STAR = {
    'CompleteGraph': -3.60,       # mu*({B,D}); see scripts/completegraph_arm_values.py
    'ConfoundedCluster': -4.50,   # mu*({A,B,C}); see scripts/confoundedcluster_arm_values.py
    'Tier1Graph': -4.50,          # mu*({B,D,E}); see scripts/tier1graph_arm_values.py
}

_EPS = 1e-12


def _improvement_ratios(traj, y_star, task='min'):
    traj = np.asarray(traj, dtype=float)
    T = len(traj) - 1
    y_init = traj[0]
    denom = (y_init - y_star) if task == 'min' else (y_star - y_init)
    if abs(denom) < _EPS:
        # Already at (or past) the reference optimum: full credit everywhere.
        return np.ones(T), T
    ratios = np.empty(T)
    for t in range(1, T + 1):
        num = (y_init - traj[t]) if task == 'min' else (traj[t] - y_init)
        ratios[t - 1] = min(num / denom, 1.0)
    return ratios, T


def gap(traj, y_star, task='min'):
    """Survey GAP in [0, 1] (final improvement + discovery speed)."""
    ratios, T = _improvement_ratios(traj, y_star, task)
    if T <= 0:
        return float('nan')
    R_T = ratios[-1]
    if R_T <= _EPS:
        t_star = T                       # no improvement
    else:
        # first trial (1-indexed) attaining the final ratio
        t_star = int(np.argmax(ratios >= R_T - 1e-9)) + 1
    return (R_T + (T - t_star) / T) / (1.0 + (T - 1) / T)


def pa_gap(traj, y_star, task='min'):
    """Survey Path-Aware GAP (raw, trajectory-weighted; in [0, (T+1)/(2T)])."""
    ratios, T = _improvement_ratios(traj, y_star, task)
    if T <= 0:
        return float('nan')
    weights = np.array([(T - (t - 1)) / T for t in range(1, T + 1)])
    return float(np.mean(ratios * weights))


def reference_optimum(benchmark, trajectories, task='min'):
    """Reference y* for a benchmark: the known oracle value, or (fallback)
    the best value observed across ``trajectories`` (a list of best-so-far
    sequences)."""
    if benchmark in Y_STAR:
        return Y_STAR[benchmark]
    finals = [v for traj in trajectories for v in traj]
    return min(finals) if task == 'min' else max(finals)
