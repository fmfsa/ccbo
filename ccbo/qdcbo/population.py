"""Versioned, explicit intervention-history population-response protocol.

Training is a fixed-common-noise Monte Carlo oracle; scoring is independent
and post-run. Neither channel clamps past Y to an estimated expectation.
This is separate from both released zero-disturbance DCBO and single-row
noisy controlled experiments. The regularized companion is a NEW SCM.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib

import numpy as np


class RegularizedNonStationarySEM:
    """New integrable companion, formula frozen before observing outcomes.

    The denominator sqrt(1+Xprev²)>=1 removes the reciprocal-normal
    singularity. All mechanisms have at most linear growth, preserving finite
    second moments over every finite horizon under Gaussian disturbances.
    Do not label its results as original NonStationaryDependentSEM results.
    """
    def __init__(self, change_point):
        self.cp = change_point

    @staticmethod
    def static():
        return OrderedDict([
            ('X', lambda e, t, s: e),
            ('Z', lambda e, t, s: s['X'][t] + e),
            ('Y', lambda e, t, s: np.sqrt(np.abs(36 - (s['Z'][t] - 1)**2)) + 1 + e)])

    def dynamic(self):
        return OrderedDict([
            ('X', lambda e, t, s: s['X'][t-1] + e),
            ('Z', lambda e, t, s: (-s['X'][t] / np.sqrt(1 + s['X'][t-1]**2)
                                  if t == self.cp else s['X'][t]) + s['Z'][t-1] + e),
            ('Y', lambda e, t, s: (s['Z'][t] * np.cos(np.pi*s['Z'][t]) - s['Y'][t-1] + e
                                  if t == self.cp else np.abs(s['Z'][t]) - s['Y'][t-1] - s['Z'][t-1] + e))])


def regularized_graph(graph, change_point=1):
    """Add every cross-lag parent used by the new companion to its DAG."""
    out = graph.copy()
    # The actual graphviz constructor returns a bare MultiDiGraph; Root
    # normally attaches T later, after this transformation is required.
    times = sorted({int(node.rsplit('_', 1)[1]) for node in out.nodes()})
    if times != list(range(len(times))):
        raise ValueError('Temporal nodes must contain contiguous slices from zero')
    out.T = len(times)
    for t in range(1, out.T):
        if t == change_point:
            out.add_edge(f'X_{t-1}', f'Z_{t}')
        else:
            out.add_edge(f'Z_{t-1}', f'Y_{t}')
    return out


def named_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256(repr((int(seed), parts)).encode()).digest()[:4], 'little')


def simulate_prefix(static_sem, dynamic_sem, interventions, temporal_index,
                    samples, seed, purpose, epsilon=None):
    """Vectorized true-SCM prefix; all free past variables are re-simulated.

    Only the declared X/Z action history is fixed. Vectorization is supported
    by the three named NumPy SEMs and validated against the scalar sampler.
    """
    if samples < 2:
        raise ValueError('At least two Monte Carlo draws required')
    if any(v is not None for v in interventions.get('Y', [])):
        raise ValueError('Population policy must not clamp past target values')
    names = tuple(static_sem)
    count = temporal_index + 1
    state = OrderedDict((v, np.zeros((count, samples))) for v in names)
    if epsilon is None:
        epsilon = {v: np.random.RandomState(named_seed(seed, purpose, v)).standard_normal((count, samples))
                   for v in names}
    for t in range(count):
        functions = static_sem if t == 0 or dynamic_sem is None else dynamic_sem
        for variable, fn in functions.items():
            value = interventions.get(variable, [None]*count)[t]
            if value is not None:
                state[variable][t] = float(value)
            else:
                state[variable][t] = fn(epsilon[variable][t], t, state)
    if not all(np.isfinite(array).all() for array in state.values()):
        raise FloatingPointError('Nonfinite true-SCM prefix')
    return state


def population_response(static_sem, dynamic_sem, history, arm, values, t,
                        samples=2048, seed=0, purpose='train'):
    blanket = deepcopy(history)
    flat = np.asarray(values, dtype=float).reshape(-1)
    if len(flat) != len(arm) or len(set(arm)) != len(arm):
        raise ValueError('Intervention arm/value dimension or duplicate mismatch')
    if not np.isfinite(flat).all() or 'Y' in arm:
        raise ValueError('Invalid intervention value or target intervention')
    for variable, value in zip(arm, flat):
        blanket[variable][t] = float(value)
    simulated = simulate_prefix(static_sem, dynamic_sem, blanket, t,
                                samples, seed, purpose)
    y = simulated['Y'][t]
    return float(y.mean()), float(y.std(ddof=1)/np.sqrt(samples)), blanket


def make_training_target(model, arm):
    def evaluate(current_target, intervention_levels, assigned_blanket):
        _, tt = current_target.rsplit('_', 1)
        t = int(tt)
        mean, se, blanket = population_response(
            model.true_initial_sem, model.true_sem, assigned_blanket,
            arm, intervention_levels, t, model.feedback_samples, model.population_seed, 'train-v1')
        model.population_events.append(dict(
            event_id=len(model.population_events), t=t, arm=list(arm),
            x=np.asarray(intervention_levels).reshape(-1).tolist(), history=blanket,
            training_mean=mean, training_mc_se=se,
            training_samples=model.feedback_samples, cost=len(arm)))
        return mean
    return evaluate


def score_events(model, samples=100000):
    """After model.run only: score previously serialized decisions, not select them."""
    for event in model.population_events:
        mean, se, _ = population_response(
            model.true_initial_sem, model.true_sem, event['history'], (), (), event['t'],
            samples, model.population_seed, 'independent-score-v1')
        event.update(population_mean=mean, population_mc_se=se, scoring_samples=samples)
    for event in model.population_events:
        recommended = model.population_events[event['recommendation_event']]
        event['recommendation_population_mean'] = recommended['population_mean']
        event['recommendation_population_mc_se'] = recommended['population_mc_se']
    for commit in model.policy_commits:
        recommended = model.population_events[commit['recommendation_event']]
        commit['population_mean'] = recommended['population_mean']
        commit['population_mc_se'] = recommended['population_mc_se']
