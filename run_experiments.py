"""
Unified experiment runner for Coarsened Causal Bayesian Optimization.

Replaces run_main_experiments.py, run_targeted.py, run_wrong_edge.py, and
run_multi_seed.py with a single parameterised entry point.

Usage examples
--------------
# Main: BO / CBO / CCBO (representative coarsenings) / RCCBO, 5 seeds, 40 trials
python run_experiments.py --benchmark CompleteGraph --condition main

# Wrong-edge robustness: two misspecification types × two partitions
python run_experiments.py --benchmark CompleteGraph --condition wrong_edge

# RCCBO teaser: RCCBO + baselines only
python run_experiments.py --benchmark CompleteGraph --condition rccbo_teaser

# Run on SimplifiedCoralGraph
python run_experiments.py --benchmark SimplifiedCoralGraph --condition main
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import pathlib

from ccbo.run_experiment import run_single_experiment, run_bo_baseline, get_original_graph
from ccbo.rccbo.rccbo import rccbo
from ccbo.coarsening import (
    enumerate_valid_coarsenings_manip, get_dag_edges_from_sem,
    get_hidden_confounders,
)
from ccbo.coarsened_graph import CoarsenedGraph
from ccbo.data_generation import generate_interventional_data
from ccbo.cbo.cbo import CBO
from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO


# ---------------------------------------------------------------------------
# Benchmark-specific representative coarsenings for 'main' condition
# ---------------------------------------------------------------------------
# All partitions are manipulable-only (Lee-2019 standard):
#   CompleteGraph M = {B, D, E}  →  2 valid coarsenings
#   SimplifiedCoralGraph M = {N, O, C, T, D}  →  47 valid coarsenings (3 representatives)

REPRESENTATIVE_COARSENINGS_COMPLETEGRAPH = [
    # Finest (identity on M): each manipulable variable is its own cluster
    ([frozenset({'B'}), frozenset({'D'}), frozenset({'E'}), frozenset({'Y'})],
     'CCBO-finest'),
    # Coarser: merge D and E into one cluster
    ([frozenset({'B'}), frozenset({'D', 'E'}), frozenset({'Y'})],
     'CCBO-{B},{DE}'),
]

# SimplifiedCoralGraph M = {N, O, C, T, D}
# POMIS for finest = {C, N, O};  for coarsest = {C, N, O, T, D}
REPRESENTATIVE_COARSENINGS_SIMPLIFIEDCORALGRAPH = [
    # Finest: all manipulable singletons — POMIS = {C,N,O}
    ([frozenset({'C'}), frozenset({'D'}), frozenset({'N'}),
      frozenset({'O'}), frozenset({'T'}), frozenset({'Y'})],
     'CCBO-finest'),
    # Natural merge: T and D are only reachable via non-manip mediators (P,CO) — POMIS = {C,N,O}
    ([frozenset({'C', 'N', 'O'}), frozenset({'D', 'T'}), frozenset({'Y'})],
     'CCBO-{CNO},{DT}'),
    # Coarsest: all manipulable in one cluster — POMIS = {C,N,O,T,D}
    ([frozenset({'C', 'D', 'N', 'O', 'T'}), frozenset({'Y'})],
     'CCBO-all-merged'),
]

# LightTunnel M = {R, G, B, P1, P2}
# Coarse partition {R,G,B} | {P1,P2} | {Y} — the partition under which Lee-2019
# latent projection drops the spurious R->G edge (see wrong_edge condition).
REPRESENTATIVE_COARSENINGS_LIGHTTUNNEL = [
    ([frozenset({'R', 'G', 'B'}), frozenset({'P1', 'P2'}), frozenset({'Y'})],
     'CCBO-{RGB},{P1P2}'),
]


def _get_representative_coarsenings(benchmark):
    if benchmark == 'CompleteGraph':
        return REPRESENTATIVE_COARSENINGS_COMPLETEGRAPH
    elif benchmark == 'SimplifiedCoralGraph':
        return REPRESENTATIVE_COARSENINGS_SIMPLIFIEDCORALGRAPH
    elif benchmark == 'LightTunnel':
        return REPRESENTATIVE_COARSENINGS_LIGHTTUNNEL
    else:
        return []


def _load_graph(benchmark, initial_num_obs_samples=100):
    data_path = os.path.join(
        os.path.dirname(__file__), 'ccbo', 'cbo', 'data', benchmark)
    obs = pd.read_pickle(os.path.join(data_path, 'observations.pkl'))
    full_obs = obs.copy()
    obs = obs[:initial_num_obs_samples]

    true_obs = None
    if benchmark in ('CoralGraph', 'SimplifiedCoralGraph'):
        true_obs = pd.read_pickle(os.path.join(data_path, 'true_observations.pkl'))

    graph = get_original_graph(benchmark, obs, true_obs)
    return graph, obs, full_obs


# ---------------------------------------------------------------------------
# Condition: main
# ---------------------------------------------------------------------------

def run_main_condition(benchmark, seeds, trials, k_phase, output_dir,
                       num_interventions=10, type_cost=1,
                       initial_num_obs_samples=100):
    """BO + CBO + representative CCBO variants + RCCBO, multi-seed."""
    results = {s: {} for s in range(seeds)}

    for seed in range(seeds):
        print(f"\n{'#'*60}  SEED {seed}  {'#'*60}")
        np.random.seed(seed)

        # BO (no causal prior)
        print(f"\n--- BO ---")
        r = run_bo_baseline(benchmark, trials, num_interventions, type_cost,
                            initial_num_obs_samples, False, 0, 'min', seed=seed)
        results[seed]['BO'] = {'global_opt': r['global_opt'], 'method': 'BO'}

        # CBO (full DAG)
        print(f"\n--- CBO ---")
        r = run_single_experiment(benchmark, -1, trials, num_interventions, type_cost,
                                  initial_num_obs_samples, True, 0, 'min', seed=seed)
        results[seed]['CBO'] = {'global_opt': r['global_opt'], 'method': 'CBO'}

        # Representative CCBO coarsenings
        graph, obs, full_obs = _load_graph(benchmark, initial_num_obs_samples)
        dag_edges, nodes, hidden_nodes, _ = get_dag_edges_from_sem(benchmark)
        hidden_conf = get_hidden_confounders(benchmark)

        # LightTunnel has 5 manipulable variables; capping the intervention size
        # at 1 keeps the exploration set to cluster singletons so each CBO trial
        # stays fast (mirrors the wrong_edge condition rationale below).
        max_size = 1 if benchmark == 'LightTunnel' else 3

        for partition, label in _get_representative_coarsenings(benchmark):
            print(f"\n--- {label} ---")
            try:
                cg = CoarsenedGraph(graph, partition, benchmark, obs,
                                    max_intervention_size=max_size,
                                    num_mc_samples=2000)
                functions = cg.fit_all_models()
                MIS, _, manip_vars = cg.get_sets()
                dict_ranges = cg.get_interventional_ranges()
                costs = cg.get_cost_structure(type_cost)

                np.random.seed(seed)
                int_data = generate_interventional_data(
                    cg.define_SEM, MIS, dict_ranges,
                    num_points=20, num_mc_samples=2000, seed=seed)

                from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO
                _, _, coverage = compute_coverage(obs, manip_vars, dict_ranges)
                x_list, y_list, best_x, opt_y, best_var = define_initial_data_CBO(
                    int_data, num_interventions, MIS, 0, 'min')

                (current_cost, _, _, global_opt, _, _) = CBO(
                    trials, MIS, manip_vars, x_list, y_list, best_x, opt_y,
                    best_var, dict_ranges, functions, obs, coverage, cg,
                    20, costs, full_obs, 'min',
                    initial_num_obs_samples + 50, initial_num_obs_samples,
                    num_interventions, Causal_prior=True)

                results[seed][label] = {
                    'global_opt': global_opt, 'method': label,
                    'partition': cg.get_partition_description(),
                    'es': cg.get_exploration_set_description(),
                }
                print(f"  Final Y: {global_opt[-1]:.4f}")
            except Exception as e:
                print(f"  {label} failed: {e}")
                import traceback; traceback.print_exc()

        # RCCBO
        print(f"\n--- RCCBO ---")
        r = rccbo(benchmark, trials, k_phase=k_phase,
                  num_interventions=num_interventions, type_cost=type_cost,
                  initial_num_obs_samples=initial_num_obs_samples,
                  task='min', num_mc_samples=2000, seed=seed)
        results[seed]['RCCBO'] = r
        print(f"  Final Y: {r['global_opt'][-1]:.4f}")
        print(f"  Final partition: {r.get('final_partition', 'none')}")

        # Checkpoint after each seed so a crash only loses the current seed
        ckpt = os.path.join(output_dir, f'{benchmark}_main_{seeds}seeds.pkl')
        with open(ckpt, 'wb') as f:
            pickle.dump(results, f)
        print(f"  [checkpoint saved: seed {seed} complete]")

    # Print summary
    print(f"\n{'='*70}\nMAIN SUMMARY — {benchmark}\n{'='*70}")
    all_methods = sorted({m for s in results for m in results[s]})
    for method in all_methods:
        finals = [results[s][method]['global_opt'][-1]
                  for s in results if method in results[s]]
        if finals:
            print(f"  {method:35s}: {np.mean(finals):.3f} ± {np.std(finals):.3f}")

    out = os.path.join(output_dir, f'{benchmark}_main_{seeds}seeds.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved {out}")
    return results


# ---------------------------------------------------------------------------
# Condition: wrong_edge
# ---------------------------------------------------------------------------

def _run_wrong_edge_condition(original_graph, obs, full_obs, benchmark,
                               partition, partition_label,
                               assumed_graph, misspec_label,
                               trials, num_interventions, type_cost,
                               initial_num_obs_samples, seed):
    np.random.seed(seed)
    label = f'{partition_label}/{misspec_label}/seed{seed}'
    # LightTunnel has 5 manipulable variables; the default max_intervention_size=3
    # would enumerate C(5,1)+C(5,2)+C(5,3)=25 exploration sets and make each
    # CBO trial ~25x slower than CompleteGraph's. Cap at 1 (singletons only)
    # so each condition completes in minutes rather than hours; the misspec
    # robustness claim is fully exercised by singleton interventions.
    max_size = 1 if benchmark == 'LightTunnel' else 3
    try:
        cg = CoarsenedGraph(original_graph, partition, benchmark, obs,
                            max_intervention_size=max_size,
                            num_mc_samples=2000,
                            assumed_graph_name=assumed_graph)
        functions = cg.fit_all_models()
        MIS, _, manip_vars = cg.get_sets()
        dict_ranges = cg.get_interventional_ranges()
        costs = cg.get_cost_structure(type_cost)

        int_data = generate_interventional_data(
            cg.define_SEM, MIS, dict_ranges,
            num_points=20, num_mc_samples=2000, seed=seed)

        _, _, coverage = compute_coverage(obs, manip_vars, dict_ranges)
        x_list, y_list, best_x, opt_y, best_var = define_initial_data_CBO(
            int_data, num_interventions, MIS, 0, 'min')

        (current_cost, _, _, global_opt, _, total_time) = CBO(
            trials, MIS, manip_vars, x_list, y_list, best_x, opt_y,
            best_var, dict_ranges, functions, obs, coverage, cg,
            20, costs, full_obs, 'min',
            initial_num_obs_samples + 50, initial_num_obs_samples,
            num_interventions, Causal_prior=True)

        return {
            'method': label,
            'partition_label': partition_label,
            'misspec': misspec_label,
            'assumed_graph': assumed_graph,
            'seed': seed,
            'global_opt': global_opt,
            'current_cost': current_cost,
            'total_time': total_time,
            'partition': cg.get_partition_description(),
            'es': cg.get_exploration_set_description(),
            'num_es': len(MIS),
        }
    except Exception as e:
        print(f"  {label} failed: {e}")
        import traceback; traceback.print_exc()
        return None


def run_wrong_edge_condition(benchmark, seeds, trials, output_dir,
                              num_interventions=10, type_cost=1,
                              initial_num_obs_samples=100):
    """
    Two misspec types × two partitions × seeds.

    CompleteGraph misspec types:
      - 'correct': no misspecification
      - 'NoBC': missing inter-cluster edge B→C (severe: removes B from all MIS
                 under identity; invisible inside {B,C,D} cluster)
      - 'NoCD': missing intra-cluster edge C→D (inside {B,C,D}; invisible there)

    Partitions:
      - identity: each variable singleton
      - {B,C,D},{E}: B→C and C→D are both internal
    """
    if benchmark not in ('CompleteGraph', 'LightTunnel'):
        print(f"Wrong-edge condition not implemented for {benchmark}")
        return {}

    graph, obs, full_obs = _load_graph(benchmark, initial_num_obs_samples)

    if benchmark == 'CompleteGraph':
        # Manipulable-only partitions for CompleteGraph (M = {B, D, E})
        finest = [frozenset({'B'}), frozenset({'D'}), frozenset({'E'}), frozenset({'Y'})]
        coarser = [frozenset({'B'}), frozenset({'D', 'E'}), frozenset({'Y'})]

        # (partition, partition_label, assumed_graph, misspec_label)
        # NoBC: removes edge B→C (B manip → C non-manip; changes causal paths to Y)
        # NoCD: removes edge C→D (C non-manip → D manip; changes adjustment sets)
        conditions = [
            (finest,  'finest',    'CompleteGraph',       'correct'),
            (finest,  'finest',    'CompleteGraph_NoBC',  'NoBC'),
            (finest,  'finest',    'CompleteGraph_NoCD',  'NoCD'),
            (coarser, '{B},{DE}',  'CompleteGraph',       'correct'),
            (coarser, '{B},{DE}',  'CompleteGraph_NoBC',  'NoBC'),
            (coarser, '{B},{DE}',  'CompleteGraph_NoCD',  'NoCD'),
        ]
    else:  # benchmark == 'LightTunnel'
        # Manipulable-only partitions for LightTunnel (M = {R, G, B, P1, P2}).
        finest = [frozenset({v}) for v in ('R', 'G', 'B', 'P1', 'P2')] + [frozenset({'Y'})]
        coarser = [
            frozenset({'R', 'G', 'B'}),
            frozenset({'P1', 'P2'}),
            frozenset({'Y'}),
        ]

        # WrongRG: spurious intra-cluster edge R -> G inside {R, G, B}.
        # CCBO under the coarser partition projects it out, so coarse/correct
        # and coarse/WrongRG should match within seed noise.
        conditions = [
            (finest,  'finest',        'LightTunnel',          'correct'),
            (finest,  'finest',        'LightTunnel_WrongRG',  'WrongRG'),
            (coarser, '{RGB},{P1P2}',  'LightTunnel',          'correct'),
            (coarser, '{RGB},{P1P2}',  'LightTunnel_WrongRG',  'WrongRG'),
        ]

    all_results = []
    for partition, part_label, assumed, misspec in conditions:
        for seed in range(seeds):
            print(f"\n--- {part_label} / {misspec} / seed={seed} ---")
            r = _run_wrong_edge_condition(
                graph, obs, full_obs, benchmark,
                partition, part_label, assumed, misspec,
                trials, num_interventions, type_cost,
                initial_num_obs_samples, seed)
            if r:
                all_results.append(r)
                print(f"  Final Y: {r['global_opt'][-1]:.4f}  |ES|={r['num_es']}")

    # Summary table
    print(f"\n{'='*70}\nWRONG-EDGE SUMMARY — {benchmark}\n{'='*70}")
    print(f"{'Partition':15} {'Misspec':8} {'Mean Y':>10} {'Std':>8} {'|ES|':>6}")
    from itertools import groupby
    for (pl, ms), grp in groupby(sorted(all_results, key=lambda r: (r['partition_label'], r['misspec'])),
                                  key=lambda r: (r['partition_label'], r['misspec'])):
        grp = list(grp)
        finals = [r['global_opt'][-1] for r in grp]
        n_es = grp[0]['num_es']
        print(f"  {pl:15} {ms:8} {np.mean(finals):10.3f} {np.std(finals):8.3f} {n_es:6d}")

    out = os.path.join(output_dir, f'{benchmark}_wrong_edge_{seeds}seeds.pkl')
    with open(out, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {out}")
    return all_results


# ---------------------------------------------------------------------------
# Condition: rccbo_teaser
# ---------------------------------------------------------------------------

def run_rccbo_teaser(benchmark, seeds, trials, k_phase, output_dir,
                     num_interventions=10, type_cost=1,
                     initial_num_obs_samples=100):
    """RCCBO + BO + CBO baselines, with per-seed partition evolution."""
    results = {s: {} for s in range(seeds)}

    for seed in range(seeds):
        print(f"\n{'#'*60}  SEED {seed}  {'#'*60}")

        print(f"\n--- BO ---")
        r = run_bo_baseline(benchmark, trials, num_interventions, type_cost,
                            initial_num_obs_samples, False, 0, 'min', seed=seed)
        results[seed]['BO'] = {'global_opt': r['global_opt'], 'method': 'BO'}

        print(f"\n--- CBO ---")
        r = run_single_experiment(benchmark, -1, trials, num_interventions, type_cost,
                                  initial_num_obs_samples, True, 0, 'min', seed=seed)
        results[seed]['CBO'] = {'global_opt': r['global_opt'], 'method': 'CBO'}

        print(f"\n--- RCCBO ---")
        r = rccbo(benchmark, trials, k_phase=k_phase,
                  num_interventions=num_interventions, type_cost=type_cost,
                  initial_num_obs_samples=initial_num_obs_samples,
                  task='min', num_mc_samples=2000, seed=seed)
        results[seed]['RCCBO'] = r
        print(f"  Final Y: {r['global_opt'][-1]:.4f}")
        print(f"  Partition history: {r.get('partition_history')}")
        print(f"  Refinement trials: {r.get('refinement_trials')}")

    # Summary
    print(f"\n{'='*70}\nRCCBO TEASER SUMMARY — {benchmark}\n{'='*70}")
    for method in ['BO', 'CBO', 'RCCBO']:
        finals = [results[s][method]['global_opt'][-1]
                  for s in results if method in results[s]]
        if finals:
            print(f"  {method:15s}: {np.mean(finals):.3f} ± {np.std(finals):.3f}")

    out = os.path.join(output_dir, f'{benchmark}_rccbo_teaser_{seeds}seeds.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved {out}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Unified CCBO experiment runner')
    parser.add_argument('--benchmark', default='CompleteGraph',
                        choices=['CompleteGraph', 'SimplifiedCoralGraph',
                                 'LightTunnel'],
                        help='Which benchmark to run')
    parser.add_argument('--condition', default='main',
                        choices=['main', 'wrong_edge', 'rccbo_teaser'],
                        help='Experiment condition')
    parser.add_argument('--seeds', default=5, type=int,
                        help='Number of random seeds')
    parser.add_argument('--trials', default=40, type=int,
                        help='Optimization trials per method per seed')
    parser.add_argument('--k-phase', default=10, type=int,
                        help='RCCBO phase length (trials before RePaRe)')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for results')
    args = parser.parse_args()

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Benchmark: {args.benchmark}")
    print(f"Condition: {args.condition}")
    print(f"Seeds: {args.seeds}   Trials: {args.trials}   k-phase: {args.k_phase}")
    print(f"Output dir: {args.output_dir}")

    if args.condition == 'main':
        run_main_condition(args.benchmark, args.seeds, args.trials, args.k_phase,
                           args.output_dir)
    elif args.condition == 'wrong_edge':
        run_wrong_edge_condition(args.benchmark, args.seeds, args.trials,
                                 args.output_dir)
    elif args.condition == 'rccbo_teaser':
        run_rccbo_teaser(args.benchmark, args.seeds, args.trials, args.k_phase,
                         args.output_dir)


if __name__ == '__main__':
    main()
