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
                         num_mc_samples=2000, assumed_graph_name=None,
                         gating=True):
    """CoarsenedGraph subclass compatible with the benchmark's BO_CBO/CBO.py.

    The benchmark CBO keys do-functions as ``compute_do_{"_".join(vars)}`` and
    calls ``graph.intervention_function`` / ``graph.create_combined_do_function``.
    Our base CoarsenedGraph concatenates names and lacks those methods, so we
    adapt here. Target evaluations delegate to the benchmark's DiscoveredGraph
    (the single-draw SEM objective every method is scored against).

    ``assumed_graph_name`` overrides the structure used for the C-DAG /
    exploration set / priors (defaults to ``ds``, the correct graph). Pass a
    registered *misspecified* variant (e.g. ``ds + '_WrongX1X2'``) to run the
    DAG-misspecification stress test; target evaluations still use the true
    ``dg`` SEM, so trajectories under correct/misspecified structure are
    directly comparable.
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
        num_mc_samples=num_mc_samples,
        assumed_graph_name=assumed_graph_name or ds, gating=gating)


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
                       out_csv=None, method_label='QCBO',
                       assumed_graph_name=None, gating=True):
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

    ``gating=False`` disables the identifiability gate on the exploration
    set (see ``CoarsenedGraph``): with the identity partition this is the
    non-gating CBO baseline used in the misspecification stress test.
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
                              max_intervention_size=max_intervention_size,
                              assumed_graph_name=assumed_graph_name,
                              gating=gating)
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


# ---------------------------------------------------------------------------
# QCBO-refine: plateau-triggered partition refinement (hierarchical C-DAG)
# ---------------------------------------------------------------------------

def _plateau(traj, k, delta, task):
    """True when the best-so-far trajectory improved < ``delta`` over the last
    ``k`` steps."""
    if len(traj) < k + 1:
        return False
    gain = (traj[-1 - k] - traj[-1]) if task == 'min' else (traj[-1] - traj[-1 - k])
    return gain < delta


def _split_partition(partition, refine_map, incumbent_vars):
    """Split every refinable cluster that intersects the incumbent arm.

    ``refine_map`` maps a cluster (any iterable of fine vars) to its declared
    sub-partition (list of lists). Returns ``(new_partition, split_clusters)``;
    ``split_clusters`` is empty when nothing applies.
    """
    rmap = {frozenset(c): [frozenset(s) for s in subs]
            for c, subs in refine_map.items()}
    inc = set(incumbent_vars)
    new_partition, split = [], []
    for part in partition:
        if part in rmap and part & inc:
            subs = rmap[part]
            assert frozenset().union(*subs) == part, (
                f"refine_map for {set(part)} is not a partition of the cluster")
            new_partition.extend(subs)
            split.append(part)
        else:
            new_partition.append(part)
    return new_partition, split


def _union_and_carry(ES_old, data_x_list, data_y_list, ES_new, fresh_sampler):
    """Union phase-1 arms into the refined ES and carry data by arm key.

    Every arm of ``ES_old`` is kept even if the refined ES (built under a
    finite ``max_intervention_size``) dropped it — the Prop. 4 superset made
    operational. ``fresh_sampler(entries)`` returns ``(xs, ys)`` init data for
    the arms that are new at the refined level. Returns ``(ES, dxl, dyl)``.
    """
    old_data = {tuple(e): (data_x_list[i], data_y_list[i])
                for i, e in enumerate(ES_old)}
    ES = [list(e) for e in ES_new]
    for e in ES_old:
        if not any(tuple(e) == tuple(n) for n in ES):
            ES.append(list(e))
    fresh = [e for e in ES if tuple(e) not in old_data]
    fresh_data = {}
    if fresh:
        fx, fy = fresh_sampler(fresh)
        fresh_data = {tuple(e): (fx[i], fy[i]) for i, e in enumerate(fresh)}
    dxl = [old_data.get(tuple(e), fresh_data.get(tuple(e)))[0] for e in ES]
    dyl = [old_data.get(tuple(e), fresh_data.get(tuple(e)))[1] for e in ES]
    return ES, dxl, dyl


def run_qcbo_refine_benchmark(ds, coarse_clusters, refine_map, seed=0,
                              num_trials=100, num_interventions=10,
                              max_intervention_size=None, out_csv=None,
                              method_label='QCBO-refine',
                              assumed_graph_name=None, gating=True,
                              plateau_k=5, plateau_delta=1e-3,
                              refine_at=None, coarse_csv=None, verbose=False):
    """QCBO with one plateau-triggered partition-refinement step (two-phase).

    Phase 1 of QCBO-refine *is* QCBO-coarse: same seed, same inputs, same
    code path. The plateau monitor (improvement < ``plateau_delta`` over the
    last ``plateau_k`` steps, evaluated every step) is therefore a function of
    the coarse trajectory alone, and its firing time ``t_r`` is determined by
    scanning that trajectory — from ``coarse_csv`` when the baseline run for
    this seed exists, otherwise by running the coarse configuration once here
    (simulation compute; the method's interventional budget stays
    ``num_trials + 1`` steps). Phase 1 then executes as a single monolithic
    CBO call of exactly ``t_r + 1`` steps — byte-identical to the coarse
    baseline's prefix.

    On trigger, every ``refine_map`` cluster intersecting the incumbent arm is
    split one level (Prop. 3 validity is re-checked by the CoarsenedGraph
    constructor; an invalid split is refused and the run continues coarse).
    The refined exploration set is unioned with every phase-1 arm (Prop. 4
    superset also under a finite ``max_intervention_size``), all
    interventional data carries over by arm key, and only the new arms
    receive the standard ``num_interventions`` init points. Phase 2 runs the
    remaining budget in a single warm-started CBO call. (The benchmark CBO
    forces an observation step at the start of every call, so the phase
    boundary costs one observation trial — a bias *against* refine.)

    ``refine_at`` (a step index) replaces the plateau trigger with a fixed
    checkpoint (ablation).

    Returns ``(traj, out_csv, info)`` where ``info`` records the trigger step
    and the split clusters.
    """
    _ensure_benchmark_on_path()
    from baselines.BO_CBO.CBO import CBO
    from baselines.BO_CBO.utils import compute_coverage
    import tempfile

    np.random.seed(seed)
    dg, obs, config = load_graph(ds, seed=seed)
    manip = list(config['intervention'])
    task = config['task']
    if max_intervention_size is None:
        max_intervention_size = min(5, len(manip))
    partition = _coarse_partition(coarse_clusters)

    def _build(part):
        cg = make_coarsened_graph(dg, ds, obs, part,
                                  max_intervention_size=max_intervention_size,
                                  assumed_graph_name=assumed_graph_name,
                                  gating=gating)
        ES = [list(e) for e in cg.get_sets()[0]]
        return cg, ES

    cg, ES = _build(partition)
    ranges = dg.get_interventional_ranges()
    dict_ranges = {v: (ranges[v][0], ranges[v][1]) for v in manip}
    # Same construction order as run_qcbo_benchmark (models fitted BEFORE the
    # init data re-seeds the RNG) so phase 1 is byte-identical to the coarse
    # baseline run.
    functions = cg.fit_all_models()
    costs = cg.get_cost_structure(1)

    (data_x_list, data_y_list, best_x, opt_y, best_variable) = \
        _initial_interventional_data(dg, ES, num_interventions, task, seed)

    _, _, coverage_total = compute_coverage(obs, cg.get_sets()[2], dict_ranges)

    if out_csv is None:
        out_csv = os.path.join(BENCH_ROOT, 'results', method_label, ds,
                               f'{method_label}_{ds}_{num_trials}-trials_progress.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    total_steps = num_trials + 1
    steps_used = 0
    traj = []            # concatenated best-so-far, one entry per executed step
    segment_csvs = []
    info = {'trigger_step': None, 'split_clusters': [], 'refused': False}

    # ---- Trigger time from the coarse trajectory ---------------------------
    if refine_at is not None:
        t_fire = int(refine_at)
    else:
        if not (coarse_csv and os.path.exists(coarse_csv)):
            fd, coarse_csv = tempfile.mkstemp(suffix='.csv',
                                              prefix='refine_coarse_')
            os.close(fd)
            run_qcbo_benchmark(
                ds, coarse_clusters=coarse_clusters, seed=seed,
                num_trials=num_trials, num_interventions=num_interventions,
                max_intervention_size=max_intervention_size,
                out_csv=coarse_csv, method_label=method_label,
                assumed_graph_name=assumed_graph_name, gating=gating)
        coarse_traj = pd.read_csv(coarse_csv)['current_optimal'] \
            .astype(float).tolist()
        t_fire = next((t for t in range(len(coarse_traj))
                       if _plateau(coarse_traj[:t + 1], plateau_k,
                                   plateau_delta, task)), None)
    # Refinement needs at least one phase-2 step.
    if t_fire is not None and not (0 < t_fire < total_steps - 1):
        t_fire = None

    def _run_segment(n_steps):
        nonlocal best_x, opt_y, best_variable, steps_used, functions
        fd, seg_csv = tempfile.mkstemp(suffix='.csv', prefix='refine_seg_')
        os.close(fd)
        result = CBO(
            n_steps - 1, ES, cg.get_sets()[2], data_x_list, data_y_list,
            best_x, opt_y, best_variable, dict_ranges, functions,
            obs, coverage_total, cg, 20, costs, obs, task,
            len(obs) + 50, len(obs), num_interventions,
            Causal_prior=True, csv_log_file=seg_csv)
        _, best_intervention, _, global_opt, _, _ = result
        segment_csvs.append(seg_csv)
        # global_opt[0] repeats the entry incumbent; per-step values follow.
        traj.extend(global_opt[1:])
        steps_used += n_steps
        if isinstance(best_intervention, dict) and 'original_key' in best_intervention:
            best_variable = best_intervention['original_key']
            best_x = best_intervention['intervention_values']
            if isinstance(best_x, list):
                best_x = np.asarray(best_x, dtype=float)
        opt_y = traj[-1]

    def _incumbent_arm():
        """Arm (fine-var tuple) holding the best observed interventional Y."""
        per_arm = [(Y.min() if task == 'min' else Y.max()) for Y in data_y_list]
        s = int(np.argmin(per_arm) if task == 'min' else np.argmax(per_arm))
        return tuple(ES[s])

    # ---- Phase 1: one monolithic coarse call up to the trigger -------------
    if t_fire is None:
        _run_segment(total_steps)          # never refines: this IS coarse
    else:
        _run_segment(t_fire + 1)           # steps 0..t_fire, coarse prefix
        incumbent = _incumbent_arm()
        new_partition, split = _split_partition(partition, refine_map,
                                                incumbent)
        new_cg = None
        if split:
            try:
                new_cg, new_ES = _build(new_partition)
            except ValueError as e:        # Prop. 3: refuse invalid split
                info['refused'] = True
                if verbose:
                    print(f'refine refused: {e}', flush=True)
        if new_cg is not None:
            def _sampler(entries):
                fx, fy, _, _, _ = _initial_interventional_data(
                    dg, entries, num_interventions, task, seed + 1000)
                return fx, fy
            ES, data_x_list, data_y_list = _union_and_carry(
                ES, data_x_list, data_y_list, new_ES, _sampler)
            # Register unioned phase-1 arms with the refined graph (before
            # its memoized get_all_do first runs) so they get C-DAG priors
            # too; each old cluster is a union of its sub-clusters, so
            # identification stays valid (Prop. 4(ii)), and a failure falls
            # back to the uninformative prior.
            for e in ES:
                if list(e) not in new_cg._exploration_set:
                    new_cg._exploration_set.append(list(e))
            cg, partition = new_cg, new_partition
            functions = cg.fit_all_models()
            costs = cg.get_cost_structure(1)
            _, _, coverage_total = compute_coverage(
                obs, cg.get_sets()[2], dict_ranges)
            info['trigger_step'] = t_fire
            info['split_clusters'] = [sorted(c) for c in split]
            if verbose:
                print(f'refined at step {t_fire}: split '
                      f'{info["split_clusters"]}, |ES|={len(ES)}', flush=True)
        # ---- Phase 2: the remaining budget in a single call ----------------
        _run_segment(total_steps - steps_used)

    # Merge segment CSVs into one benchmark-format progress CSV.
    frames = []
    offset = 0
    for p in segment_csvs:
        f = pd.read_csv(p)
        f['trial_number'] = f['trial_number'] + offset
        offset += len(f)
        frames.append(f)
        os.remove(p)
    merged = pd.concat(frames, ignore_index=True)
    # Best-so-far must be monotone across segment boundaries by construction;
    # enforce defensively so scoring never regresses on a merge artifact.
    col = merged['current_optimal'].astype(float)
    merged['current_optimal'] = (np.minimum.accumulate(col) if task == 'min'
                                 else np.maximum.accumulate(col))
    merged.to_csv(out_csv, index=False)
    return traj, out_csv, info
