"""
RCCBO — Recursive Coarsened Causal Bayesian Optimization.

Main algorithm: start from pure BO (no causal knowledge), run phases of
optimization, use RePaRe to discover causal partition structure from
interventional data, switch to causal GP priors, and continue refining.

Phase 0 is plain BO on all manipulative variables (no partition, no C-DAG,
no causal prior). Once RePaRe discovers structure, RCCBO builds a C-DAG,
enables causal priors, and continues the CBO loop with progressive
refinement.
"""

import os
import warnings

import numpy as np
import pandas as pd

from ccbo.coarsened_graph import CoarsenedGraph
from ccbo.coarsening import get_hidden_confounders, get_dag_edges_from_sem
from ccbo.data_generation import generate_interventional_data

from ccbo.rccbo.partition_ops import (
    coarsest_valid_partition, partitions_equal, partition_diff, partition_description,
)
from ccbo.rccbo.repare_bridge import FullSampleStore, run_repare, sample_full_variables
from ccbo.rccbo.state_manager import patch_state_for_new_partition

from ccbo.cbo.cbo import CBO
from ccbo.cbo.utils import (
    compute_coverage, define_initial_data_CBO,
    get_interventional_dict,
)


def _find_closest_valid_partition(invalid_partition, experiment):
    """
    Find the closest valid partition to an invalid one suggested by RePaRe.

    Strategy: enumerate all valid coarsenings and find the one that is
    a refinement of the coarsest partition but closest to the RePaRe suggestion.
    "Closest" = most clusters in common (highest Rand index).

    Returns None if no closer-than-coarsest valid partition exists.
    """
    from ccbo.coarsening import enumerate_valid_coarsenings_manip

    valid = enumerate_valid_coarsenings_manip(experiment)
    coarsest = valid[-1]['partition']  # fewest parts = last (sorted finest-first)

    # Build element-to-cluster mapping for the invalid partition
    invalid_map = {}
    for cluster in invalid_partition:
        for elem in cluster:
            invalid_map[elem] = cluster

    best_score = -1
    best_partition = None
    for entry in valid:
        p = entry['partition']
        # Skip the coarsest (that's what we already have)
        if set(frozenset(c) for c in p) == set(frozenset(c) for c in coarsest):
            continue
        # Score: count element pairs that are in the same cluster in both partitions
        score = 0
        p_map = {}
        for cluster in p:
            for elem in cluster:
                p_map[elem] = cluster
        for elem in invalid_map:
            if elem in p_map:
                for elem2 in invalid_map:
                    if elem2 in p_map and elem < elem2:
                        if (invalid_map[elem] == invalid_map[elem2]) == (p_map[elem] == p_map[elem2]):
                            score += 1
        if score > best_score:
            best_score = score
            best_partition = p

    return best_partition


def rccbo(experiment, total_trials, *, seed, k_phase=10,
          num_interventions=10, type_cost=1,
          initial_num_obs_samples=100,
          task='min', num_mc_samples=2000,
          repare_alpha=0.05, repare_beta=0.05):
    """
    Run Recursive Coarsened Causal Bayesian Optimization.

    RCCBO starts as plain BO (no causal prior, no partition) and uses
    interventional data to progressively discover causal structure via
    RePaRe. Once a non-trivial partition is found, it switches to CCBO
    with causal GP priors and continues refining.

    Parameters
    ----------
    experiment : str
        Graph name (e.g. 'CompleteGraph').
    total_trials : int
        Total optimization budget (number of BO trials).
    k_phase : int
        Number of BO trials per phase before considering refinement.
    num_interventions : int
        Number of initial interventional data points per ES entry.
    type_cost : int
        Cost structure type.
    initial_num_obs_samples : int
        Number of initial observational samples.
    task : str
        'min' or 'max'.
    num_mc_samples : int
        MC samples for do-calculus.
    repare_alpha : float
        RePaRe significance level for partition refinement.
    repare_beta : float
        RePaRe significance level for adjacency testing.
    seed : int
        Random seed. Keyword-only and required so callers cannot accidentally
        share a default. Phases re-seed with seed + phase_idx; intervention
        callback samples use seed + phase_idx * 100000 + call_idx so each
        full-variable sample draws from an independent, reproducible stream.

    Returns
    -------
    dict with keys:
        'global_opt': list of best Y values per trial
        'current_cost': list of cumulative costs
        'partition_history': list of partitions used (None for BO phase)
        'refinement_trials': list of trial indices where refinement occurred
        'total_time': float
        'experiment': str
    """
    np.random.seed(seed)

    # --- Load data ---
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'cbo', 'data', experiment)
    observational_samples = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))[:initial_num_obs_samples]
    full_observational_samples = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))

    true_obs = None
    if experiment in ('CoralGraph', 'SimplifiedCoralGraph'):
        true_obs = pd.read_pickle(
            os.path.join(data_path, 'true_observations.pkl'))

    # --- Instantiate the original graph (needed for SEM, target function) ---
    from ccbo.run_experiment import get_original_graph
    original_graph = get_original_graph(experiment, observational_samples, true_obs)

    # --- Get manipulative variables and ranges from original graph ---
    MIS_orig, POMIS_orig, manipulative_variables = original_graph.get_sets()
    dict_ranges = original_graph.get_interventional_ranges()
    costs = original_graph.get_cost_structure(type_cost)
    functions = original_graph.fit_all_models()

    # --- Set up FullSampleStore for RePaRe ---
    # Use only observable variables (exclude hidden confounders U1, U2, etc.)
    _, obs_nodes, hidden_nodes, _ = get_dag_edges_from_sem(experiment)
    hidden_set = set(hidden_nodes)
    var_names = sorted([c for c in observational_samples.columns
                        if c not in hidden_set and not c.startswith('U')])
    # Use full observational samples for RePaRe's statistical baseline.
    # The observations.pkl must be generated from the same SEM as define_SEM()
    # to avoid distribution mismatches that break KS / Gaussian LRT tests.
    _repare_obs = full_observational_samples[var_names]
    sample_store = FullSampleStore(var_names, _repare_obs)

    # --- Phase 0 setup: uninformed CBO (no causal prior) ---
    # Start with singleton interventions only: do(B), do(D), do(E).
    # Each singleton isolates one variable's causal effect, giving RePaRe
    # the clearest signal per variable for partition discovery.  This is
    # much cheaper than the full power set (3 vs 7 entries for 3 variables).
    exploration_set = [[v] for v in sorted(manipulative_variables)]
    partition = None  # No partition in uninformed phase
    graph = original_graph  # Use original graph for SEM/target function
    use_causal_prior = False  # Start without causal priors

    # Generate initial interventional data for the single ES entry
    interventional_data = generate_interventional_data(
        graph.define_SEM, exploration_set, dict_ranges,
        num_points=20, num_mc_samples=2000, seed=seed)

    max_N = initial_num_obs_samples + 50
    alpha_coverage, hull_obs, coverage_total = compute_coverage(
        observational_samples, manipulative_variables, dict_ranges)

    data_x_list, data_y_list, best_intervention_value, opt_y, best_variable = \
        define_initial_data_CBO(interventional_data, num_interventions,
                                exploration_set, 0, task)

    # Per-call counter so each intervention sample uses a distinct,
    # reproducible seed derived from the user-provided seed + phase + call.
    _callback_state = {'phase_idx': 0, 'call_idx': 0}

    # --- Intervention callback: records full-variable samples for RePaRe ---
    def _intervention_callback(intervention_vars, x_new, y_new, sem_fn):
        """Called by CBO after each intervention to record full-variable sample."""
        intervention_dict = {var: float(x_new[0, j])
                             for j, var in enumerate(intervention_vars)}
        call_seed = (seed
                     + _callback_state['phase_idx'] * 100_000
                     + _callback_state['call_idx'])
        _callback_state['call_idx'] += 1
        try:
            full_sample = sample_full_variables(
                sem_fn, intervention_dict, num_samples=100, seed=call_seed)
            sample_store.record_intervention(intervention_vars, full_sample)
        except Exception as e:
            print(f"Warning: failed to record full sample: {e}")

    # --- Tracking ---
    partition_history = [None]  # None = BO phase (no partition)
    refinement_trials = []
    trials_completed = 0
    cbo_state = None

    import time as _time
    _rccbo_start = _time.perf_counter()

    print(f"RCCBO: Starting as pure BO (no causal prior, no partition)")
    print(f"RCCBO: Manipulative variables = {manipulative_variables}")
    print(f"RCCBO: Total budget = {total_trials}, phase length = {k_phase}")
    print(f"RCCBO: Exploration set = {exploration_set}")

    # --- Main loop: phases of BO/CBO + RePaRe refinement ---
    while trials_completed < total_trials:
        # Re-seed at each phase boundary so phase k always draws from the
        # same stream regardless of the trajectory taken to get there.
        np.random.seed(seed + _callback_state['phase_idx'])
        _callback_state['call_idx'] = 0

        trials_this_phase = min(k_phase, total_trials - trials_completed)

        phase_type = "BO" if partition is None else "CCBO"
        part_desc = "none (pure BO)" if partition is None else partition_description(partition)
        print(f"\n=== RCCBO Phase [{phase_type}] (trials {trials_completed}–{trials_completed + trials_this_phase - 1}) ===")
        print(f"    Partition: {part_desc}")
        print(f"    ES: {exploration_set}")
        print(f"    Causal prior: {use_causal_prior}")

        # Force full model rebuild on first intervention of each resumed
        # phase. Without this, CBO would reuse stale GP priors from the
        # previous phase and collapse to exploitation.
        if cbo_state is not None:
            cbo_state['force_rebuild_all'] = True

        # --- Run CBO phase ---
        results, cbo_state = CBO(
            trials_this_phase, exploration_set, manipulative_variables,
            data_x_list, data_y_list, best_intervention_value, opt_y,
            best_variable, dict_ranges, functions, observational_samples,
            coverage_total, graph, 20, costs,
            full_observational_samples, task, max_N,
            initial_num_obs_samples, num_interventions,
            Causal_prior=use_causal_prior,
            state=cbo_state, return_state=True,
            intervention_callback=_intervention_callback,
        )

        trials_completed += trials_this_phase

        # Each phase consumes one seed-offset slot, regardless of whether
        # the refinement that follows accepts a new partition or `continue`s.
        _callback_state['phase_idx'] += 1

        # Update references from state
        observational_samples = cbo_state['observational_samples']
        functions = cbo_state['functions']
        data_x_list = cbo_state['data_x_list']
        data_y_list = cbo_state['data_y_list']

        # Store the current ES in state for state_manager
        cbo_state['_exploration_set'] = exploration_set

        # Extract best so far
        opt_y = cbo_state['global_opt'][-1]

        print(f"    Phase complete. Best Y so far: {opt_y}")
        print(f"    Interventional samples for RePaRe: {sample_store.num_interventional_samples()}")

        # --- RePaRe refinement ---
        if trials_completed < total_trials and sample_store.num_interventional_samples() >= 10:
            print(f"\n--- RePaRe refinement attempt ---")
            try:
                new_partition, repare_model = run_repare(
                    sample_store, alpha=repare_alpha, beta=repare_beta,
                    seed=seed)

                print(f"    RePaRe suggests: {partition_description(new_partition)}")

                # Validate: (i) every non-singleton, non-Y cluster is
                # manipulable-only (Lee-2019 requirement), and (ii) the
                # induced C-DAG is acyclic.
                from ccbo.coarsening import (
                    project_out_hidden, build_coarsened_admg,
                    _admg_is_acyclic)
                _, _all_nodes, _hidden, _manip = get_dag_edges_from_sem(experiment)
                _manip_set = set(_manip)
                _hidden = set(_hidden or [])
                _nonmanip = set(_all_nodes) - _hidden - _manip_set - {'Y'}

                partition_is_manip_only = all(
                    (cluster == frozenset({'Y'})
                     or cluster.issubset(_manip_set)
                     or (len(cluster) == 1 and
                         next(iter(cluster)) in _nonmanip))
                    for cluster in new_partition
                )
                proj_val = project_out_hidden(experiment)
                test_cdag = build_coarsened_admg(
                    new_partition, proj_val, atomic_vertices=_nonmanip)
                if not partition_is_manip_only or not _admg_is_acyclic(test_cdag):
                    reason = ('mixed manipulable/non-manipulable cluster'
                              if not partition_is_manip_only
                              else 'cyclic C-DAG')
                    print(f"    ⚠ Suggested partition is INVALID ({reason}). Skipping.")
                    # Try to find the closest valid partition
                    new_partition = _find_closest_valid_partition(
                        new_partition, experiment)
                    if new_partition is None:
                        print(f"    No valid refinement found. Continuing with current partition.")
                        continue  # skip to next phase
                    print(f"    Closest valid partition: {partition_description(new_partition)}")

                # Determine the current effective partition for comparison
                if partition is None:
                    current_effective = coarsest_valid_partition(experiment)
                else:
                    current_effective = partition

                if not partitions_equal(new_partition, current_effective):
                    # Refinement occurred!
                    diff = partition_diff(current_effective, new_partition)
                    print(f"    Refinement detected! Splits: {diff}")

                    # Build everything in temporary variables first (atomic).
                    # Only commit to state if the entire process succeeds.
                    old_graph = graph
                    old_partition = partition

                    new_graph = CoarsenedGraph(
                        original_graph, new_partition, experiment,
                        observational_samples, num_mc_samples=num_mc_samples)

                    # Marginal-likelihood acceptance gate (Plan §6.1).
                    # Reject refinements that worsen the observational fit.
                    try:
                        new_mll = new_graph.marginal_log_likelihood()
                        if isinstance(graph, CoarsenedGraph):
                            cur_mll = graph.marginal_log_likelihood()
                        else:
                            cur_mll = float('-inf')  # Phase-0 BO has no C-DAG
                        print(f"    MLL: current={cur_mll:.3f}  new={new_mll:.3f}")
                        if new_mll < cur_mll:
                            print(f"    Rejecting refinement (MLL did not improve)")
                            continue
                    except Exception as _mll_err:
                        print(f"    MLL gate skipped ({_mll_err})")

                    new_functions = new_graph.fit_all_models()
                    MIS_new, _, new_manip = new_graph.get_sets()
                    new_exploration_set = MIS_new
                    new_ranges = new_graph.get_interventional_ranges()
                    new_costs = new_graph.get_cost_structure(type_cost)

                    print(f"    New ES: {new_exploration_set}")
                    print(f"    Identifiability: {new_graph.get_identifiability_summary()}")
                    if old_partition is None:
                        print(f"    >>> Transitioning from BO to CCBO with causal priors <<<")

                    # Patch CBO state for the new partition
                    new_cbo_state = patch_state_for_new_partition(
                        cbo_state, old_graph, new_graph,
                        new_exploration_set, task=task)

                    # Everything succeeded — now commit all state changes
                    partition = new_partition
                    partition_history.append(partition)
                    refinement_trials.append(trials_completed)
                    graph = new_graph
                    functions = new_functions
                    manipulative_variables = new_manip
                    dict_ranges = new_ranges
                    costs = new_costs
                    use_causal_prior = True
                    cbo_state = new_cbo_state
                    exploration_set = new_exploration_set
                    data_x_list = cbo_state['data_x_list']
                    data_y_list = cbo_state['data_y_list']

                    # Recompute coverage for new manipulative vars
                    alpha_coverage, hull_obs, coverage_total = compute_coverage(
                        observational_samples, manipulative_variables, dict_ranges)

                    # Update best for next CBO call — robust search over
                    # all ES entries for the actual best Y and corresponding X
                    best_y_val = np.inf if task == 'min' else -np.inf
                    best_variable = list(cbo_state['current_best_y'].keys())[0]
                    best_intervention_value = 0.
                    for var, ys in cbo_state['current_best_y'].items():
                        for idx_y, y_val in enumerate(ys):
                            if ((task == 'min' and y_val < best_y_val) or
                                    (task == 'max' and y_val > best_y_val)):
                                best_y_val = y_val
                                best_variable = var
                                xs = cbo_state['current_best_x'].get(var, [])
                                if idx_y < len(xs):
                                    best_intervention_value = xs[idx_y]

                else:
                    print(f"    No refinement needed (partition unchanged)")

            except Exception as e:
                # Catch broadly: RePaRe wraps a heterogeneous stack of
                # statistical tests (KS, Gaussian LRT, distance correlation)
                # whose failure modes are not enumerable. We log the full
                # traceback so the user can diagnose, and continue with the
                # current partition rather than aborting RCCBO.
                cur_desc = ("none (still in BO phase)" if partition is None
                            else partition_description(partition))
                warnings.warn(
                    f"RePaRe refinement failed at trial {trials_completed}: "
                    f"{type(e).__name__}: {e}. Keeping current partition "
                    f"({cur_desc}) and continuing.",
                    RuntimeWarning, stacklevel=2)
                import traceback
                traceback.print_exc()

    # --- Collect final results ---
    _rccbo_total_time = _time.perf_counter() - _rccbo_start

    # Enforce monotonicity: global_opt should never regress
    final_global_opt = list(cbo_state['global_opt'])
    if task == 'min':
        for i in range(1, len(final_global_opt)):
            final_global_opt[i] = min(final_global_opt[i], final_global_opt[i - 1])
    else:
        for i in range(1, len(final_global_opt)):
            final_global_opt[i] = max(final_global_opt[i], final_global_opt[i - 1])
    final_cost = cbo_state['current_cost']

    return {
        'global_opt': final_global_opt,
        'current_cost': final_cost,
        'partition_history': partition_history,
        'refinement_trials': refinement_trials,
        'total_time': _rccbo_total_time,
        'experiment': experiment,
        'method': 'RCCBO',
        'exploration_set': exploration_set,
        'final_partition': partition_description(partition) if partition else 'none (BO only)',
    }
