"""
Main experiment runner for Coarsened Causal Bayesian Optimization.

Runs CBO with different coarsenings and compares against standard BO
and original CBO.
"""

import os
import argparse
import pathlib
import time
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

from ccbo.cbo.cbo import CBO
from ccbo.cbo.bo import NonCausal_BO
from ccbo.cbo.utils import *
from ccbo.cbo.graphs import *

from .coarsening import (enumerate_valid_coarsenings_manip,
                          get_dag_edges_from_sem,
                          print_coarsenings, get_hidden_confounders)
from .coarsened_graph import CoarsenedGraph
from .data_generation import generate_interventional_data


def get_original_graph(experiment, observational_samples, true_obs=None):
    """Instantiate the original graph object."""
    if experiment == 'ToyGraph':
        return ToyGraph(observational_samples)
    elif experiment == 'CompleteGraph':
        return CompleteGraph(observational_samples)
    elif experiment == 'CoralGraph':
        return CoralGraph(observational_samples, true_obs)
    elif experiment == 'SimplifiedCoralGraph':
        return SimplifiedCoralGraph(observational_samples, true_obs)
    elif experiment in ('LightTunnel', 'LightTunnel_WrongRG'):
        return LightTunnel(observational_samples)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def run_single_experiment(experiment, coarsening_idx, num_trials, num_interventions,
                          type_cost, initial_num_obs_samples, causal_prior,
                          name_index, task='min', num_mc_samples=10000,
                          exploration_set_type='MIS', seed=9):
    """
    Run a single CCBO experiment with a specific coarsening.

    Parameters
    ----------
    experiment : str
        Graph name ('ToyGraph', 'CompleteGraph').
    coarsening_idx : int
        Index into valid coarsenings list. -1 = original CBO.
    num_trials : int
        Number of optimization trials.
    num_interventions : int
        Number of initial interventional data points.
    type_cost : int
        Cost structure type.
    initial_num_obs_samples : int
        Number of initial observational samples.
    causal_prior : bool
        Whether to use causal GP prior.
    name_index : int
        Index for data permutation.
    task : str
        'min' or 'max'.
    num_mc_samples : int
        MC samples for do-calculus.
    exploration_set_type : str
        'MIS' or 'POMIS'.

    Returns
    -------
    dict with results
    """
    np.random.seed(seed)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'cbo', 'data', experiment)
    observational_samples = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))[:initial_num_obs_samples]
    full_observational_samples = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))

    # Load true observations for coral graphs
    true_obs = None
    if experiment in ('CoralGraph', 'SimplifiedCoralGraph'):
        true_obs = pd.read_pickle(
            os.path.join(data_path, 'true_observations.pkl'))

    # Get original graph
    original_graph = get_original_graph(experiment, observational_samples, true_obs)

    # Get graph metadata
    dag_edges, nodes, hidden_nodes, fine_manip_vars = get_dag_edges_from_sem(experiment)

    if coarsening_idx == -1:
        # Run original CBO
        method_name = f'CBO-{exploration_set_type}'
        graph = original_graph
        functions = graph.fit_all_models()
        MIS, POMIS, manipulative_variables = graph.get_sets()
        exploration_set = MIS if exploration_set_type == 'MIS' else POMIS
        dict_ranges = graph.get_interventional_ranges()

        # Generate interventional data from SEM (seed-dependent)
        interventional_data = generate_interventional_data(
            graph.define_SEM, exploration_set, dict_ranges,
            num_points=20, num_mc_samples=2000, seed=seed)
    else:
        # Run CCBO with specific coarsening (manipulable-only; Lee-2019)
        coarsenings = enumerate_valid_coarsenings_manip(experiment)

        if coarsening_idx >= len(coarsenings):
            raise ValueError(
                f"Coarsening index {coarsening_idx} out of range "
                f"(only {len(coarsenings)} valid coarsenings)")

        partition = coarsenings[coarsening_idx]['partition']
        graph = CoarsenedGraph(
            original_graph, partition, experiment,
            observational_samples, num_mc_samples=num_mc_samples)

        method_name = f'CCBO-{coarsening_idx}'
        functions = graph.fit_all_models()
        MIS, POMIS, manipulative_variables = graph.get_sets()
        exploration_set = MIS  # For coarsened, MIS = POMIS
        dict_ranges = graph.get_interventional_ranges()

        # Generate interventional data for the coarsened exploration set
        print(f"Generating interventional data for coarsening {coarsening_idx}...")
        print(f"  Partition: {graph.get_partition_description()}")
        print(f"  Exploration set: {graph.get_exploration_set_description()}")
        interventional_data = generate_interventional_data(
            graph.define_SEM, exploration_set, dict_ranges,
            num_points=20, num_mc_samples=2000, seed=seed)

    # Set up CBO
    max_N = initial_num_obs_samples + 50
    alpha_coverage, hull_obs, coverage_total = compute_coverage(
        observational_samples, manipulative_variables, dict_ranges)

    data_x_list, data_y_list, best_intervention_value, opt_y, best_variable = \
        define_initial_data_CBO(interventional_data, num_interventions,
                                exploration_set, name_index, task)

    costs = graph.get_cost_structure(type_cost)

    print(f"\n--- Running {method_name} ---")
    print(f"  Exploration set: {exploration_set}")
    print(f"  Causal prior: {causal_prior}")
    print(f"  Num trials: {num_trials}")

    # Run CBO
    (current_cost, current_best_x, current_best_y,
     global_opt, observed, total_time) = CBO(
        num_trials, exploration_set, manipulative_variables,
        data_x_list, data_y_list, best_intervention_value, opt_y,
        best_variable, dict_ranges, functions, observational_samples,
        coverage_total, graph, 20, costs,
        full_observational_samples, task, max_N,
        initial_num_obs_samples, num_interventions,
        Causal_prior=causal_prior)

    result = {
        'method': method_name,
        'coarsening_idx': coarsening_idx,
        'global_opt': global_opt,
        'current_cost': current_cost,
        'total_time': total_time,
        'observed': observed,
        'experiment': experiment,
        'causal_prior': causal_prior,
        'exploration_set': exploration_set,
        'num_trials': num_trials,
    }

    if coarsening_idx >= 0:
        result['partition'] = graph.get_partition_description()
        result['es_description'] = graph.get_exploration_set_description()

    return result


def run_bo_baseline(experiment, num_trials, num_interventions, type_cost,
                    initial_num_obs_samples, causal_prior, name_index,
                    task='min', seed=9):
    """Run the standard BO baseline."""
    np.random.seed(seed)

    data_path = os.path.join(os.path.dirname(__file__), 'cbo', 'data', experiment)
    observational_samples = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))[:initial_num_obs_samples]
    full_observational_samples = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))

    true_obs = None
    if experiment in ('CoralGraph', 'SimplifiedCoralGraph'):
        true_obs = pd.read_pickle(
            os.path.join(data_path, 'true_observations.pkl'))

    original_graph = get_original_graph(experiment, observational_samples, true_obs)
    functions = original_graph.fit_all_models()

    MIS, POMIS, manipulative_variables = original_graph.get_sets()
    dict_ranges = original_graph.get_interventional_ranges()
    costs = original_graph.get_cost_structure(type_cost)

    # Generate interventional data from SEM (seed-dependent)
    # BO uses all manipulative variables as a single joint intervention
    bo_exploration_set = [list(manipulative_variables)]
    interventional_data_cbo = generate_interventional_data(
        original_graph.define_SEM, bo_exploration_set, dict_ranges,
        num_points=20, num_mc_samples=2000, seed=seed)

    # Convert CBO-format to BO format
    # CBO format: [num_vars, var1, ..., vark, data_x, data_y]
    # BO expects: [var1, ..., varN, data_x, data_y]
    int_entry = interventional_data_cbo[0]
    num_v = int_entry[0]
    interventional_data = list(int_entry[1:1+num_v]) + [int_entry[-2], int_entry[-1]]

    data_x, data_y, min_intervention_value, min_y = define_initial_data_BO(
        [interventional_data], num_interventions, manipulative_variables, name_index)

    print(f"\n--- Running BO baseline ---")
    print(f"  Variables: {manipulative_variables}")
    print(f"  Causal prior: {causal_prior}")

    (current_cost, current_best_x, current_best_y, total_time) = NonCausal_BO(
        num_trials, original_graph, dict_ranges, data_x, data_y,
        costs, observational_samples, functions,
        min_intervention_value, min_y, manipulative_variables,
        Causal_prior=causal_prior)

    return {
        'method': 'BO',
        'coarsening_idx': None,
        'global_opt': list(current_best_y[:, 0]),
        'current_cost': list(current_cost[:, 0]),
        'total_time': total_time,
        'experiment': experiment,
        'causal_prior': causal_prior,
        'num_trials': num_trials,
    }


def run_all(experiment, num_trials=40, num_interventions=10, type_cost=1,
            initial_num_obs_samples=100, causal_prior=True, name_index=0,
            task='min', num_mc_samples=10000):
    """
    Run all experiments for a given graph: BO, original CBO, and all CCBO coarsenings.

    Returns
    -------
    list of dict
        Results from each method.
    """
    results = []

    # 1. BO baseline
    print("=" * 60)
    print("Running BO baseline...")
    print("=" * 60)
    try:
        bo_result = run_bo_baseline(
            experiment, num_trials, num_interventions, type_cost,
            initial_num_obs_samples, False, name_index, task)
        results.append(bo_result)
    except Exception as e:
        print(f"BO baseline failed: {e}")

    # 2. Original CBO (MIS)
    print("\n" + "=" * 60)
    print("Running original CBO (MIS)...")
    print("=" * 60)
    try:
        cbo_result = run_single_experiment(
            experiment, -1, num_trials, num_interventions, type_cost,
            initial_num_obs_samples, causal_prior, name_index, task,
            num_mc_samples, 'MIS')
        results.append(cbo_result)
    except Exception as e:
        print(f"CBO-MIS failed: {e}")

    # 3. Original CBO (POMIS)
    print("\n" + "=" * 60)
    print("Running original CBO (POMIS)...")
    print("=" * 60)
    try:
        cbo_pomis_result = run_single_experiment(
            experiment, -1, num_trials, num_interventions, type_cost,
            initial_num_obs_samples, causal_prior, name_index, task,
            num_mc_samples, 'POMIS')
        results.append(cbo_pomis_result)
    except Exception as e:
        print(f"CBO-POMIS failed: {e}")

    # 4. All valid coarsenings (manipulable-only; Lee-2019)
    coarsenings = enumerate_valid_coarsenings_manip(experiment)

    print(f"\nFound {len(coarsenings)} valid coarsenings")
    print_coarsenings(experiment)

    # Skip the finest coarsening (identity = same as original CBO)
    # The finest is index 0 (sorted by -num_parts)
    for idx in range(len(coarsenings)):
        num_parts = coarsenings[idx]['num_parts']
        # Check if this is the identity partition (all singletons)
        all_singletons = all(len(p) == 1 for p in coarsenings[idx]['partition'])
        if all_singletons:
            print(f"\nSkipping coarsening {idx} (identity = original CBO)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running CCBO coarsening {idx} ({num_parts} parts)...")
        print(f"{'=' * 60}")
        try:
            ccbo_result = run_single_experiment(
                experiment, idx, num_trials, num_interventions, type_cost,
                initial_num_obs_samples, causal_prior, name_index, task,
                num_mc_samples)
            results.append(ccbo_result)
        except Exception as e:
            print(f"CCBO coarsening {idx} failed: {e}")
            import traceback
            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description='Coarsened Causal Bayesian Optimization')
    parser.add_argument('--experiment', default='CompleteGraph', type=str,
                        choices=['ToyGraph', 'CompleteGraph'],
                        help='Which graph to use')
    parser.add_argument('--num_trials', default=40, type=int,
                        help='Number of optimization trials')
    parser.add_argument('--num_interventions', default=10, type=int,
                        help='Number of initial interventional data points')
    parser.add_argument('--type_cost', default=1, type=int,
                        help='Cost structure type')
    parser.add_argument('--initial_num_obs_samples', default=100, type=int,
                        help='Number of initial observational samples')
    parser.add_argument('--causal_prior', action='store_true',
                        help='Use causal GP prior')
    parser.add_argument('--name_index', default=0, type=int,
                        help='Data permutation index')
    parser.add_argument('--task', default='min', type=str,
                        help='min or max')
    parser.add_argument('--num_mc_samples', default=10000, type=int,
                        help='MC samples for do-calculus')
    parser.add_argument('--list_coarsenings', action='store_true',
                        help='Just list valid coarsenings and exit')
    parser.add_argument('--output_dir', default='results', type=str,
                        help='Output directory for results')
    args = parser.parse_args()

    if args.list_coarsenings:
        print_coarsenings(args.experiment)
        return

    # Run all experiments
    results = run_all(
        args.experiment,
        num_trials=args.num_trials,
        num_interventions=args.num_interventions,
        type_cost=args.type_cost,
        initial_num_obs_samples=args.initial_num_obs_samples,
        causal_prior=args.causal_prior,
        name_index=args.name_index,
        task=args.task,
        num_mc_samples=args.num_mc_samples)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', args.output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir,
                               f'{args.experiment}_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_file}")

    # Plot
    from .visualize import plot_convergence_and_final, print_summary_table
    plot_convergence_and_final(
        results, args.experiment,
        os.path.join(output_dir, f'{args.experiment}_comparison.pdf'))
    print_summary_table(results, args.experiment)


if __name__ == '__main__':
    main()
