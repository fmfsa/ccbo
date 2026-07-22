"""Standardized metrics for CBO trajectories: regret (primary) and GAP.

Primary metrics are final Y, simple regret, and cumulative regret — direct
functions of the trajectory and the oracle optimum. GAP (originally from
DCBO, Aglietti et al. 2021) is kept as a secondary diagnostic; it is
sensitive to the method-specific initial incumbent, so regret and final Y
carry the headline comparisons.

All metrics are computed from a best-so-far trajectory ``traj`` (length
``T + 1``: one initial incumbent ``traj[0]`` followed by ``T`` post-trial
best-so-far values) and a reference optimum ``y_star`` (the offline oracle
optimum of the benchmark).

GAP definition (minimization shown; maximization flips the sign)
----------------------------------------------------------------
Improvement ratio at trial ``t`` (clipped to [0, 1]):

    R_t = min( (y_init - traj[t]) / (y_init - y_star), 1 )

GAP combines final improvement with discovery speed (t* = first trial that
attains the final ratio R_T; t* = T if no improvement):

    GAP = ( R_T + (T - t*) / T ) / ( 1 + (T - 1) / T )       in [0, 1]
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


def simple_regret(traj, y_star, task='min'):
    """Per-trial simple regret of the incumbent: r_t = |traj[t] - y_star|.

    ``traj`` is the best-so-far trajectory (length T+1, including the initial
    incumbent traj[0]); the returned array has length T+1 so r[0] is the
    regret of the initial incumbent and r[-1] the final simple regret.
    Smaller is better; 0 means the oracle optimum was found.
    """
    traj = np.asarray(traj, dtype=float)
    r = (traj - y_star) if task == 'min' else (y_star - traj)
    return np.maximum(r, 0.0)


def cumulative_regret(traj, y_star, task='min'):
    """Cumulative regret of the incumbent sequence: sum_{t=1..T} r_t.

    NOTE: this is the *incumbent-based* cumulative regret (sum of best-so-far
    regrets over trials), not the textbook cumulative regret over per-trial
    proposals -- the pipeline logs only best-so-far values, so per-proposal
    regret is not reconstructible for existing runs. The initial incumbent
    r_0 is excluded (it reflects the shared initialization, not the method).
    Smaller is better.
    """
    return float(np.sum(simple_regret(traj, y_star, task)[1:]))


def reference_optimum(benchmark, trajectories, task='min'):
    """Reference y* for a benchmark: the known oracle value, or (fallback)
    the best value observed across ``trajectories`` (a list of best-so-far
    sequences)."""
    if benchmark in Y_STAR:
        return Y_STAR[benchmark]
    finals = [v for traj in trajectories for v in traj]
    return min(finals) if task == 'min' else max(finals)
