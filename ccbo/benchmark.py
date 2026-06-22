"""Adapter for the CausalBO_Benchmark (anonymous.4open.science/r/CausalBO_Benchmark).

Lets QCBO (our `CoarsenedGraph` + vendored CBO) run on the benchmark's
standardized hard-intervention datasets and be scored by the benchmark's own
GAP / PA-GAP pipeline.

Design
------
We reuse the benchmark's ``DiscoveredGraph`` as the *original* (fine-grained)
graph: it already exposes ``define_SEM()`` (an OrderedDict of structural
functions in exactly the format our pipeline expects), ``get_interventional_
ranges()``, ``fit_all_models()`` and ``get_cost_structure()``. We parse the DAG
(directed edges + shared-latent confounders) from the dataset's
``sem_equations.json`` and register it with :mod:`ccbo.coarsening`, so
``CoarsenedGraph`` can build the quotient C-DAG, gate the exploration set, and
fit cluster-level priors from the benchmark's observational CSV.

Consistency with the benchmark's scoring: target evaluations use the
benchmark's (noise-free) SEM, and the reference optimum y* is the benchmark's
``observational_datasets/<ds>_theoretical_best.json`` — so our trajectories are
comparable to the other methods scored by the same harness.
"""

import os
import sys
import json
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml

from ccbo import coarsening


# Repo root -> third_party/CausalBO_Benchmark
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_ROOT = os.path.join(_REPO_ROOT, 'third_party', 'CausalBO_Benchmark')


def _ensure_benchmark_on_path():
    if not os.path.isdir(BENCH_ROOT):
        raise FileNotFoundError(
            f"CausalBO_Benchmark not found at {BENCH_ROOT}. "
            f"Download it into third_party/ first (see plan J1).")
    if BENCH_ROOT not in sys.path:
        sys.path.insert(0, BENCH_ROOT)


def parse_dag(sem_equations):
    """Parse sem_equations.json into observable edges + shared-latent confounders.

    Returns ``(dag_edges, nodes, confounders)`` where ``confounders`` is a list
    of ``(latent_name, [observable_vars...])`` for latents shared by >= 2
    observables (these become bidirected edges in the C-DAG). Per-node latents
    (affecting a single observable) are ordinary noise and are ignored.
    """
    variables = sem_equations['variables']
    nodes = list(variables.keys())
    edges = []
    latent_to_vars = {}
    for v, info in variables.items():
        for parent in info.get('dependencies', []):
            edges.append((parent, v))
        for lat in info.get('relationship_params', {}).get('latent_variables', []):
            latent_to_vars.setdefault(lat, []).append(v)
    confounders = [(lat, vs) for lat, vs in latent_to_vars.items() if len(vs) >= 2]
    return edges, nodes, confounders


def _paths(ds):
    return {
        'config': os.path.join(BENCH_ROOT, 'configs', f'{ds}.yaml'),
        'obs': os.path.join(BENCH_ROOT, 'observational_datasets', f'{ds}.csv'),
        'sem_eq': os.path.join(BENCH_ROOT, 'sem', f'{ds}_sem_equations.json'),
        'sem_model': os.path.join(BENCH_ROOT, 'sem', f'{ds}_sem_model.pkl'),
        'feat': os.path.join(BENCH_ROOT, 'observational_datasets',
                             f'{ds}_feature_params.json'),
        'ybest': os.path.join(BENCH_ROOT, 'observational_datasets',
                              f'{ds}_theoretical_best.json'),
    }


def load_config(ds):
    with open(_paths(ds)['config']) as f:
        return yaml.safe_load(f)


def theoretical_best(ds):
    """Reference optimum y* from the benchmark (global_best_value)."""
    with open(_paths(ds)['ybest']) as f:
        return float(json.load(f)['global_best_value'])


def _rename_in_sem(sem_equations, old, new):
    """Rename a variable across a sem_equations dict (in place, on a copy)."""
    se = json.loads(json.dumps(sem_equations))  # deep copy
    if old in se['variables']:
        se['variables'][new] = se['variables'].pop(old)
    for info in se['variables'].values():
        info['dependencies'] = [new if d == old else d
                                for d in info.get('dependencies', [])]
        coefs = info.get('coefficients')
        if isinstance(coefs, dict) and old in coefs:
            coefs[new] = coefs.pop(old)
    se['causal_order'] = [new if v == old else v for v in se.get('causal_order', [])]
    return se


def _load_raw(ds, num_observations=None, seed=0):
    """Load + normalize a benchmark dataset so the target is named 'Y'.

    Returns ``(model_results, sem_equations, feature_params, obs_df, config)``
    with the target column/variable renamed to 'Y' everywhere (CoarsenedGraph
    assumes a 'Y' target). Manipulable variable names are unchanged.
    """
    p = _paths(ds)
    config = load_config(ds)
    target = config['target']
    with open(p['sem_eq']) as f:
        sem_equations = json.load(f)
    with open(p['feat']) as f:
        feature_params = json.load(f)
    with open(p['sem_model'], 'rb') as f:
        model_results = pickle.load(f)

    data = pd.read_csv(p['obs'])
    if data.columns[0] == 'Unnamed: 0':
        data = data.drop(columns=data.columns[0])

    if target != 'Y':
        sem_equations = _rename_in_sem(sem_equations, target, 'Y')
        if target in data.columns:
            data = data.rename(columns={target: 'Y'})
        if target in feature_params:
            feature_params['Y'] = feature_params.pop(target)
        for key in ('node_names', 'causal_order'):
            if key in model_results:
                model_results[key] = ['Y' if v == target else v
                                      for v in model_results[key]]
        config = dict(config)
        config['target'] = 'Y'

    n = num_observations or int(config.get('num_observations', len(data)))
    n = min(n, len(data))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(data), n, replace=False)
    obs = data.iloc[idx].reset_index(drop=True)
    return model_results, sem_equations, feature_params, obs, config


def register_dataset(ds):
    """Parse the (target-normalized) DAG and register it with ccbo.coarsening.

    Returns the parsed ``(dag_edges, nodes, confounders, manipulative)``.
    """
    _, sem_equations, _, _, config = _load_raw(ds)
    manip = list(config['intervention'])
    edges, nodes, confounders = parse_dag(sem_equations)
    coarsening.register_graph(
        ds, dag_edges=edges, nodes=nodes, hidden_nodes=[],
        manipulative_variables=manip, confounders=confounders)
    return edges, nodes, confounders, manip


def load_graph(ds, num_observations=None, seed=0):
    """Build the benchmark's DiscoveredGraph + observational sample for ``ds``.

    Returns ``(discovered_graph, observational_samples, config)`` with target
    normalized to 'Y'. Also registers the dataset's DAG with ccbo.coarsening.
    """
    _ensure_benchmark_on_path()
    from baselines.BO_CBO.graph import DiscoveredGraph  # noqa: E402

    model_results, sem_equations, feature_params, obs, config = _load_raw(
        ds, num_observations=num_observations, seed=seed)

    edges, nodes, confounders = parse_dag(sem_equations)
    coarsening.register_graph(
        ds, dag_edges=edges, nodes=nodes, hidden_nodes=[],
        manipulative_variables=list(config['intervention']),
        confounders=confounders)

    dg = DiscoveredGraph(model_results, sem_equations, obs,
                         config['target'], config, feature_params)
    return dg, obs, config


def true_do_value(dg, intervention_vars, values):
    """Noise-free E[Y | do(intervention_vars = values)] via the benchmark SEM.

    Mirrors DiscoveredGraph.intervention_function (epsilon = 0), so it matches
    the convention used to compute the benchmark's theoretical best and to
    score every other method.
    """
    model = dg.define_SEM()
    values = np.atleast_1d(np.asarray(values, dtype=float))
    fixed = {v: float(values[i]) for i, v in enumerate(intervention_vars)}
    result = {}
    epsilon = np.zeros(len(model))
    for i, (var, func) in enumerate(model.items()):
        if var in fixed:
            result[var] = fixed[var]
        else:
            result[var] = func(epsilon=epsilon[i:i + 1], **result)
    return float(result[dg.target])


def true_do_mc(dg, intervention_vars, values, n=3000, seed=0):
    """Monte-Carlo estimate of E[Y | do(...)] over the SEM's exogenous draws.

    ``true_do_value`` performs one forward pass; exogenous uniforms/noises are
    redrawn each call, so averaging integrates them out (matching how the
    benchmark's reference optimum is estimated).
    """
    np.random.seed(seed)
    ys = [true_do_value(dg, intervention_vars, values) for _ in range(n)]
    return float(np.mean(ys))


# ---------------------------------------------------------------------------
# Running QCBO inside the benchmark's CBO harness
# ---------------------------------------------------------------------------

def _finest_partition(manip):
    return [frozenset({v}) for v in manip] + [frozenset({'Y'})]


def _coarse_partition(clusters):
    return [frozenset(c) for c in clusters] + [frozenset({'Y'})]


def make_coarsened_graph(dg, ds, obs, partition, max_intervention_size=1,
                         num_mc_samples=2000):
    """CoarsenedGraph subclass compatible with the benchmark's BO_CBO/CBO.py.

    The benchmark CBO keys do-functions as ``compute_do_{"_".join(vars)}`` and
    calls ``graph.intervention_function`` / ``graph.create_combined_do_function``.
    Our base CoarsenedGraph concatenates names and lacks those methods, so we
    adapt here. Target evaluations delegate to the benchmark's DiscoveredGraph
    (the single-draw SEM objective every method is scored against).
    """
    from ccbo.coarsened_graph import CoarsenedGraph

    class BenchmarkCoarsenedGraph(CoarsenedGraph):
        def get_all_do(self):
            base = super().get_all_do()  # keys: 'compute_do_' + ''.join(entry)
            out = dict(base)
            entries = list(self._exploration_set) + [self._manipulative_variables]
            for entry in entries:
                cat = 'compute_do_' + ''.join(entry)
                und = 'compute_do_' + '_'.join(entry)
                if cat in base:
                    out[und] = base[cat]
            return out

        def intervention_function(self, intervention_dict):
            # Delegate true (single-draw) evaluation to the benchmark SEM.
            return self.original_graph.intervention_function(intervention_dict)

        def create_combined_do_function(self, variables):
            # Our cluster do-functions are already built in get_all_do for every
            # exploration-set entry and for the full manipulable set; nothing to do.
            return None

    return BenchmarkCoarsenedGraph(
        dg, partition, ds, obs, max_intervention_size=max_intervention_size,
        num_mc_samples=num_mc_samples, assumed_graph_name=ds)


def _initial_interventional_data(dg, ES, num_interventions, task, seed):
    """Sample initial interventional data per exploration-set entry via the
    benchmark's single-draw objective (matching its protocol)."""
    np.random.seed(seed)
    ranges = dg.get_interventional_ranges()
    data_x_list, data_y_list = [], []
    for entry in ES:
        idx_map = {v: i for i, v in enumerate(entry)}
        tfn, _ = dg.intervention_function(idx_map)
        cols = [np.random.uniform(ranges[v][0], ranges[v][1], num_interventions)
                for v in entry]
        X = np.column_stack(cols)
        Y = np.array([float(tfn(X[i:i + 1])[0, 0])
                      for i in range(num_interventions)])[:, None]
        data_x_list.append(X)
        data_y_list.append(Y)
    # Best arm / incumbent across the initial data.
    arm_opt = [(Y.min() if task == 'min' else Y.max()) for Y in data_y_list]
    best_s = int(np.argmin(arm_opt) if task == 'min' else np.argmax(arm_opt))
    opt_y = float(arm_opt[best_s])
    yb = data_y_list[best_s][:, 0]
    bi = int(np.argmin(yb) if task == 'min' else np.argmax(yb))
    best_intervention_value = data_x_list[best_s][bi]
    best_variable = ('_'.join(ES[best_s]) if len(ES[best_s]) > 1 else ES[best_s][0])
    return data_x_list, data_y_list, best_intervention_value, opt_y, best_variable


def run_qcbo_benchmark(ds, coarse_clusters=None, seed=0, num_trials=40,
                       num_interventions=10, max_intervention_size=None,
                       out_csv=None, method_label='QCBO'):
    """Run QCBO (finest if coarse_clusters is None, else coarse) on a benchmark
    dataset through the benchmark's CBO loop; write a progress CSV.

    Returns ``(global_opt, out_csv)``.

    ``max_intervention_size`` caps how many partition clusters may be jointly
    intervened on. When ``None`` (default) it is set to ``min(5, len(manip))``,
    which **exactly matches the benchmark CBO's own exploration set** —
    ``DiscoveredGraph.get_sets`` enumerates every variable subset up to size
    ``min(5, n_manip)`` regardless of the config's ``num_intervention`` field.
    At the finest (identity) partition this makes QCBO's exploration set
    byte-identical to the benchmark CBO's, which is the Prop. 1 anchor the
    paper claims. Hardcoding 1 here silently strips every joint-intervention
    arm (e.g. ``do(Aspirin, Statin)``) and makes QCBO-finest a strictly
    weaker method than CBO rather than its equivalent.
    """
    _ensure_benchmark_on_path()
    from baselines.BO_CBO.CBO import CBO
    from baselines.BO_CBO.utils import compute_coverage

    np.random.seed(seed)
    dg, obs, config = load_graph(ds, seed=seed)
    manip = list(config['intervention'])
    task = config['task']
    if max_intervention_size is None:
        max_intervention_size = min(5, len(manip))
    partition = (_finest_partition(manip) if coarse_clusters is None
                 else _coarse_partition(coarse_clusters))

    cg = make_coarsened_graph(dg, ds, obs, partition,
                              max_intervention_size=max_intervention_size)
    ES, _, manip_vars = cg.get_sets()
    ranges = dg.get_interventional_ranges()
    dict_ranges = {v: (ranges[v][0], ranges[v][1]) for v in manip}
    functions = cg.fit_all_models()
    costs = cg.get_cost_structure(1)

    (data_x_list, data_y_list, best_x, opt_y, best_variable) = \
        _initial_interventional_data(dg, ES, num_interventions, task, seed)

    _, _, coverage_total = compute_coverage(obs, manip_vars, dict_ranges)

    if out_csv is None:
        out_csv = os.path.join(BENCH_ROOT, 'results', method_label, ds,
                               f'{method_label}_{ds}_{num_trials}-trials_progress.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    result = CBO(
        num_trials, ES, manip_vars, data_x_list, data_y_list, best_x, opt_y,
        best_variable, dict_ranges, functions, obs, coverage_total, cg,
        20, costs, obs, task, len(obs) + 50, len(obs), num_interventions,
        Causal_prior=True, csv_log_file=out_csv)

    # CBO returns ... global_opt as last element (see run_cbo.py usage).
    global_opt = result[-1] if isinstance(result, (list, tuple)) else None
    return global_opt, out_csv
