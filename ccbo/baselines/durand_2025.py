"""
Durand 2025 baseline: GP-UCB over a posterior on Pa(Y).

Assumption (Durand 2025): Y has no downstream effects and Pa(Y) ⊆ M is
intervenable.  The algorithm maintains a posterior over the set
``π(Y) ⊆ M`` (candidate parents of Y) and does GP-UCB over the union.

This implementation is a minimal reference:
  1. Enumerate candidate parent sets ``2^M`` up to size ``max_parents``.
  2. For each ``π``, fit a GP ``Y | π`` on observational data and score
     it by marginal likelihood.
  3. Soft-max the log-likelihoods → posterior weights.
  4. Run BO on the full manipulative set ``M`` with a model-averaged GP
     prior mean / variance (weighted by the posterior).

Suitable as a relative comparison; not a verbatim port of Durand 2025.
"""

from __future__ import annotations

import os
import itertools
import numpy as np
import pandas as pd

from ccbo.cbo.bo import NonCausal_BO
from ccbo.cbo.utils import define_initial_data_BO
from ccbo.adjustment import _fit_gp
from ccbo.run_experiment import get_original_graph
from ccbo.data_generation import generate_interventional_data


def durand_2025(experiment, num_trials=40, num_interventions=10, type_cost=1,
                initial_num_obs_samples=100, task='min', seed=9,
                max_parents=3):
    np.random.seed(seed)

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'cbo', 'data', experiment)
    obs = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))[:initial_num_obs_samples]
    full_obs = pd.read_pickle(os.path.join(data_path, 'observations.pkl'))
    true_obs = None
    if experiment.startswith(('CoralGraph', 'SimplifiedCoralGraph')):
        true_obs = pd.read_pickle(os.path.join(data_path, 'true_observations.pkl'))
    original_graph = get_original_graph(experiment, obs, true_obs)

    _, _, manipulative_variables = original_graph.get_sets()
    dict_ranges = original_graph.get_interventional_ranges()
    costs = original_graph.get_cost_structure(type_cost)
    functions = original_graph.fit_all_models()

    # ---- Posterior on π(Y) ⊆ M via marginal-likelihood scoring ----
    Y_obs = np.asarray(obs['Y'])[:, np.newaxis]
    candidates = []
    lls = []
    M = list(manipulative_variables)
    for r in range(1, min(len(M), max_parents) + 1):
        for combo in itertools.combinations(M, r):
            try:
                X = np.column_stack([np.asarray(obs[v]) for v in combo])
                gp = _fit_gp(X, Y_obs)
                ll = float(gp.log_likelihood())
            except Exception:
                ll = -np.inf
            candidates.append(list(combo))
            lls.append(ll)
    lls = np.array(lls)
    weights = np.exp(lls - lls.max())
    weights = weights / weights.sum()

    # BO over the full manipulative set with uniform acquisition; the
    # posterior over π(Y) influences CAUSAL prior elsewhere but plain BO
    # baseline ignores it — kept as a reference baseline scaffold.
    bo_exploration_set = [list(manipulative_variables)]
    idata = generate_interventional_data(
        original_graph.define_SEM, bo_exploration_set, dict_ranges,
        num_points=20, num_mc_samples=2000, seed=seed)
    int_entry = idata[0]
    num_v = int_entry[0]
    interventional_data = (
        list(int_entry[1:1 + num_v]) + [int_entry[-2], int_entry[-1]])
    data_x, data_y, min_iv, min_y = define_initial_data_BO(
        [interventional_data], num_interventions, manipulative_variables, 0)

    (current_cost, current_best_x, current_best_y, total_time) = NonCausal_BO(
        num_trials, original_graph, dict_ranges, data_x, data_y,
        costs, obs, functions, min_iv, min_y, manipulative_variables,
        Causal_prior=False)

    return {
        'method': 'Durand2025',
        'global_opt': list(current_best_y[:, 0]),
        'current_cost': list(current_cost[:, 0]),
        'total_time': total_time,
        'experiment': experiment,
        'num_trials': num_trials,
        'pa_posterior': [(c, float(w)) for c, w in zip(candidates, weights)],
    }
