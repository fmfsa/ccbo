"""
GACBO baseline (Mukherjee 2024): graph-uncertain causal BO.

Uniform prior over the set of plausible DAGs consistent with the manipulable
set; exploration set = union of MIS across that ensemble; causal prior =
posterior-weighted average of each DAG's GP prediction.

This is a thin, self-contained reference implementation suitable for a
relative comparison against CCBO / RCCBO.  It is **not** a verbatim port of
the Mukherjee 2024 paper — the paper's specific MCMC / RJ-MCMC sampler over
DAGs is replaced with exhaustive enumeration over a user-supplied "plausible
graph family" (defaults to: the true graph, and small perturbations of it).
The plausible family is restricted so enumeration is tractable for the two
benchmarks (CompleteGraph / SimplifiedCoralGraph).

Returns the same result dict shape as :func:`ccbo.run_experiment.run_single_experiment`.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from ccbo.cbo.cbo import CBO
from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO
from ccbo.coarsening import (enumerate_valid_coarsenings_manip,
                              get_dag_edges_from_sem)
from ccbo.coarsened_graph import CoarsenedGraph
from ccbo.data_generation import generate_interventional_data
from ccbo.run_experiment import get_original_graph


def _plausible_family(experiment):
    """
    A small, user-configurable plausible-graph family for enumeration.

    Default: singleton — the true graph name plus any *_NoXY misspecified
    variants that already exist as named graphs in ``coarsening.py``.
    """
    family = [experiment]
    for variant in (f'{experiment}_NoBC', f'{experiment}_NoCD'):
        try:
            get_dag_edges_from_sem(variant)
            family.append(variant)
        except Exception:
            pass
    return family


def gacbo(experiment, num_trials=40, num_interventions=10, type_cost=1,
          initial_num_obs_samples=100, task='min', seed=9,
          num_mc_samples=2000):
    """
    Run GACBO on a benchmark.

    Strategy
    --------
    1. Enumerate the plausible graph family ``G_1, ..., G_K``.
    2. For each ``G_k``, build a CoarsenedGraph on the finest manipulable
       partition (= plain CBO on ``G_k``) with its own POMIS / identifier.
    3. Run CBO on the *union* exploration set with a causal prior computed
       by averaging the C-DAG do-functions across the family (uniform
       weights; equivalent to a marginal posterior under a flat prior).
    """
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

    family = _plausible_family(experiment)

    # Build one CoarsenedGraph per family member on the *finest* manip
    # partition so each has the most informative C-DAG available.
    cgraphs = []
    for gname in family:
        coarsenings = enumerate_valid_coarsenings_manip(gname)
        finest = coarsenings[0]['partition']  # sorted finest-first
        cg = CoarsenedGraph(original_graph, finest, gname, obs,
                             num_mc_samples=num_mc_samples,
                             assumed_graph_name=gname)
        cgraphs.append(cg)

    # Union exploration set across the family (de-duplicated)
    union_es_tuples = set()
    for cg in cgraphs:
        for e in cg.get_sets()[0]:
            union_es_tuples.add(tuple(sorted(e)))
    exploration_set = [list(e) for e in union_es_tuples]

    _, _, manipulative_variables = cgraphs[0].get_sets()
    dict_ranges = cgraphs[0].get_interventional_ranges()
    functions = cgraphs[0].fit_all_models()
    costs = cgraphs[0].get_cost_structure(type_cost)

    # Model-averaged do-dicts: one do_fn per ES entry, averaging across
    # the family's per-graph C-DAG identifiers.
    do_dicts = [cg.get_all_do() for cg in cgraphs]
    avg_do_dict = {}
    for es in exploration_set:
        name = 'compute_do_' + ''.join(es)
        fns = [d[name] for d in do_dicts if name in d]
        if not fns:
            continue

        def _avg(obs_, funcs_, value, _fns=fns):
            means, vars_ = [], []
            for fn in _fns:
                m, v = fn(obs_, funcs_, value)
                means.append(m)
                vars_.append(v)
            m = float(np.mean(means))
            v = float(np.mean(vars_) + np.var(means))
            return m, v
        avg_do_dict[name] = _avg

    class _Wrapper:
        """Minimal graph-interface wrapper exposing averaged do-dict."""
        def __init__(self, cg, do_dict):
            self._cg = cg
            self._do_dict = do_dict
            for attr in ('define_SEM', 'refit_models', 'get_cost_structure'):
                setattr(self, attr, getattr(cg, attr))

        def get_sets(self):
            return exploration_set, exploration_set, manipulative_variables

        def get_interventional_ranges(self):
            return dict_ranges

        def fit_all_models(self):
            return functions

        def get_all_do(self):
            return self._do_dict

    graph = _Wrapper(cgraphs[0], avg_do_dict)

    interventional_data = generate_interventional_data(
        graph.define_SEM, exploration_set, dict_ranges,
        num_points=20, num_mc_samples=2000, seed=seed)
    max_N = initial_num_obs_samples + 50
    _, _, coverage_total = compute_coverage(
        obs, manipulative_variables, dict_ranges)
    data_x_list, data_y_list, best_iv, opt_y, best_var = \
        define_initial_data_CBO(
            interventional_data, num_interventions, exploration_set, 0, task)

    (current_cost, current_best_x, current_best_y,
     global_opt, observed, total_time) = CBO(
        num_trials, exploration_set, manipulative_variables,
        data_x_list, data_y_list, best_iv, opt_y, best_var,
        dict_ranges, functions, obs, coverage_total, graph, 20, costs,
        full_obs, task, max_N, initial_num_obs_samples, num_interventions,
        Causal_prior=True)

    return {
        'method': 'GACBO',
        'global_opt': global_opt,
        'current_cost': current_cost,
        'total_time': total_time,
        'experiment': experiment,
        'num_trials': num_trials,
        'exploration_set': exploration_set,
        'family': family,
    }
