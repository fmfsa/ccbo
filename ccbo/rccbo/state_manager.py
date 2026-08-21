"""
State manager for RCCBO.

Handles patching the CBO state dict when the partition changes
after a RePaRe refinement step. Key responsibilities:

1. Map old exploration set entries to new ones (which persist, which are new)
2. Carry over interventional data for matching ES entries
3. Rebuild target functions and parameter spaces for new ES entries
4. Recompute do-function priors for all ES entries on the new C-DAG
"""

import numpy as np

from ccbo.cbo.utils import (
    initialise_dicts,
    Intervention_function,
    get_interventional_dict,
    list_interventional_ranges,
    update_all_do_functions,
)


def _es_key(es_entry):
    """Canonical key for an exploration set entry (sorted tuple of var names)."""
    return tuple(sorted(es_entry))


def patch_state_for_new_partition(old_state, old_graph, new_graph,
                                  new_exploration_set, task='min'):
    """
    Patch a CBO state dict after the partition has been refined.

    Parameters
    ----------
    old_state : dict
        CBO state from the previous phase (from CBO(..., return_state=True)).
    old_graph : CoarsenedGraph
        The CoarsenedGraph from the previous partition.
    new_graph : CoarsenedGraph
        The CoarsenedGraph for the refined partition.
    new_exploration_set : list of list of str
        The new exploration set from new_graph.get_sets()[0].
    task : str
        'min' or 'max'.

    Returns
    -------
    dict
        Patched state dict ready to pass to CBO(..., state=patched_state).
    """
    old_es = old_state.get('_exploration_set', [])
    old_data_x = old_state['data_x_list']
    old_data_y = old_state['data_y_list']

    # Build mapping: old ES key → (data_x, data_y)
    old_data_map = {}
    for idx, es_entry in enumerate(old_es):
        key = _es_key(es_entry)
        old_data_map[key] = (old_data_x[idx], old_data_y[idx])

    # Build new data lists
    new_n = len(new_exploration_set)
    new_data_x = [None] * new_n
    new_data_y = [None] * new_n

    for idx, es_entry in enumerate(new_exploration_set):
        key = _es_key(es_entry)
        if key in old_data_map:
            # Exact match: carry over data
            new_data_x[idx] = old_data_map[key][0]
            new_data_y[idx] = old_data_map[key][1]
            continue

        # No exact match. Look for a predecessor ES that is a strict
        # subset of the new ES — its intervention values can be lifted
        # to the new dimensionality by sampling the extra coordinates
        # and re-evaluating the SEM. This preserves data we already paid
        # for at the cost of one SEM eval per lifted point.
        new_set = set(es_entry)
        best_subset_key = None
        best_overlap = 0
        for old_key in old_data_map:
            old_set = set(old_key)
            if old_set < new_set and len(old_set) > best_overlap:
                best_subset_key = old_key
                best_overlap = len(old_set)

        if best_subset_key is not None:
            new_data_x[idx], new_data_y[idx] = _lift_subset_data(
                old_data_map[best_subset_key], best_subset_key, es_entry,
                new_graph,
            )
        else:
            # No usable predecessor; generate fresh initial data.
            new_data_x[idx], new_data_y[idx] = _generate_initial_data(
                es_entry, new_graph, num_points=5
            )

    # Rebuild target functions and parameter spaces for ALL entries
    target_function_list = [None] * new_n
    space_list = [None] * new_n
    for s in range(new_n):
        target_function_list[s], space_list[s] = Intervention_function(
            get_interventional_dict(new_exploration_set[s]),
            model=new_graph.define_SEM(),
            target_variable='Y',
            min_intervention=list_interventional_ranges(
                new_graph.get_interventional_ranges(), new_exploration_set[s])[0],
            max_intervention=list_interventional_ranges(
                new_graph.get_interventional_ranges(), new_exploration_set[s])[1],
        )

    # Recompute do-function priors on the new C-DAG
    functions = new_graph.fit_all_models()
    new_best_x, new_best_y, x_dict_mean, x_dict_var, dict_interventions = \
        initialise_dicts(new_exploration_set, task)

    mean_functions_list, var_functions_list = update_all_do_functions(
        new_graph, new_exploration_set, functions, dict_interventions,
        old_state['observational_samples'], x_dict_mean, x_dict_var,
    )

    # Carry over the global tracking from old state
    # (current_best_x/y need to be re-keyed for new dict_interventions)
    patched_best_x, patched_best_y = _carry_over_best(
        old_state['current_best_x'], old_state['current_best_y'],
        new_best_x, new_best_y, dict_interventions,
    )

    # Ensure global_opt never regresses: seed at least one ES entry with
    # the overall best Y from the old phase so find_current_global preserves it.
    prev_best_y = old_state['global_opt'][-1] if old_state['global_opt'] else None
    if prev_best_y is not None:
        _ensure_global_best_preserved(patched_best_y, prev_best_y, task)

    new_state = {
        'current_cost': old_state['current_cost'],
        'global_opt': old_state['global_opt'],
        'current_best_x': patched_best_x,
        'current_best_y': patched_best_y,
        'x_dict_mean': x_dict_mean,
        'x_dict_var': x_dict_var,
        'dict_interventions': dict_interventions,
        'observed': old_state['observed'],
        'trial_intervened': old_state['trial_intervened'],
        'cumulative_cost': old_state['cumulative_cost'],
        'target_function_list': target_function_list,
        'space_list': space_list,
        'model_list': [None] * new_n,  # will be rebuilt on first use
        'type_trial': list(old_state['type_trial']),
        # Force CBO to rebuild every GP on the first intervention after
        # refinement so stale priors from the old partition are discarded.
        'force_rebuild_all': True,
        'mean_functions_list': mean_functions_list,
        'var_functions_list': var_functions_list,
        'functions': functions,
        'observational_samples': old_state['observational_samples'],
        # Observation-pool state is *persistent run state*: it survives a
        # partition change, so the refined phase never re-reveals rows the
        # previous phase already consumed.
        'obs_cursor': old_state.get('obs_cursor'),
        'initial_obs_cursor': old_state.get('initial_obs_cursor'),
        'obs_cap': old_state.get('obs_cap'),
        'num_observations_collected': old_state.get(
            'num_observations_collected', 0),
        'trial_log': list(old_state.get('trial_log', [])),
        # Every model_list entry is None below and force_rebuild_all is set,
        # so no arm GP is fresh.
        'models_fresh': False,
        'index': 0,
        'data_x_list': new_data_x,
        'data_y_list': new_data_y,
        # Private: used by state_manager on next refinement
        '_exploration_set': new_exploration_set,
    }

    return new_state


def _lift_subset_data(old_data, old_key, new_es_entry, graph):
    """
    Lift interventional data from a subset ES to the new (larger) ES.

    For each point in `old_data`, we keep its intervention values for the
    shared variables and sample new values for the extra variables from
    their interventional ranges, then re-evaluate the SEM to get a fresh Y.
    """
    from ccbo.generic_do import sample_from_model, intervene

    old_x, old_y = old_data
    n = old_x.shape[0]
    old_vars = list(old_key)
    new_vars = list(new_es_entry)
    extra_vars = [v for v in new_vars if v not in set(old_vars)]
    dict_ranges = graph.get_interventional_ranges()

    rng = np.random.RandomState(0)
    data_x = np.zeros((n, len(new_vars)))
    for j, var in enumerate(new_vars):
        if var in old_vars:
            data_x[:, j] = old_x[:, old_vars.index(var)]
        else:
            lo, hi = dict_ranges[var]
            data_x[:, j] = rng.uniform(lo, hi, n)

    data_y = np.zeros((n, 1))
    sem_fn = graph.define_SEM
    for i in range(n):
        intervention_dict = {var: data_x[i, j]
                             for j, var in enumerate(new_vars)}
        model = sem_fn()
        mutilated = intervene(intervention_dict, model)
        np.random.seed(i)
        samples = [sample_from_model(mutilated) for _ in range(1000)]
        data_y[i, 0] = np.mean([s['Y'] for s in samples])

    return data_x, data_y


def _generate_initial_data(es_entry, graph, num_points=5):
    """
    Generate initial interventional data for a new exploration set entry.

    Uses the true SEM to sample intervention outcomes.
    """
    from ccbo.generic_do import sample_from_model, intervene

    dict_ranges = graph.get_interventional_ranges()
    sem_fn = graph.define_SEM
    num_vars = len(es_entry)

    rng = np.random.RandomState(42)
    data_x = np.zeros((num_points, num_vars))
    for j, var in enumerate(es_entry):
        lo, hi = dict_ranges[var]
        data_x[:, j] = rng.uniform(lo, hi, num_points)

    data_y = np.zeros((num_points, 1))
    for i in range(num_points):
        intervention_dict = {var: data_x[i, j]
                             for j, var in enumerate(es_entry)}
        model = sem_fn()
        mutilated = intervene(intervention_dict, model)
        np.random.seed(42 + i)
        samples = [sample_from_model(mutilated) for _ in range(10000)]
        data_y[i, 0] = np.mean([s['Y'] for s in samples])

    return data_x, data_y


def _ensure_global_best_preserved(patched_best_y, prev_best_y, task='min'):
    """
    Ensure at least one ES entry in patched_best_y contains the previous
    global best, so that find_current_global never regresses.

    After refinement, new ES entries are initialized with [np.inf] (for min task).
    If no old entry's key matches a new key exactly, the previous global best
    is lost. This function injects it into the first non-trivial entry.
    """
    import numpy as np

    # Check if any entry already contains the previous best
    for key, vals in patched_best_y.items():
        if vals and len(vals) > 0:
            if task == 'min' and np.min(vals) <= prev_best_y:
                return  # already preserved
            elif task == 'max' and np.max(vals) >= prev_best_y:
                return  # already preserved

    # Not preserved — inject into the first entry
    first_key = next(iter(patched_best_y))
    patched_best_y[first_key].append(prev_best_y)


def _carry_over_best(old_best_x, old_best_y, new_best_x, new_best_y,
                     new_dict_interventions):
    """
    Carry over current-best tracking from old to new exploration set.

    Variables that exist in both old and new ES keep their best values.
    New ES entries start empty.
    """
    for new_key in new_best_x:
        # new_key is a string like 'B, D' or 'E'
        if new_key in old_best_x and old_best_x[new_key]:
            new_best_x[new_key] = old_best_x[new_key]
            new_best_y[new_key] = old_best_y[new_key]

    return new_best_x, new_best_y
