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

RCCBO is excluded from the paper experiment matrix (see KNOWN_ISSUES.md);
run_rccbo_teaser() is kept callable for follow-up work.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import pathlib

from ccbo.run_experiment import run_single_experiment, run_bo_baseline, get_original_graph
from ccbo.rccbo.rccbo import rccbo
from ccbo.coarsening import enumerate_valid_coarsenings_manip
from ccbo.coarsened_graph import CoarsenedGraph
from ccbo.data_generation import generate_interventional_data
from ccbo.cbo.cbo import CBO
from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO


# ---------------------------------------------------------------------------
# Unified experiment protocol
# ---------------------------------------------------------------------------
# One protocol per benchmark, applied to EVERY method (BO, CBO, all CCBO
# partitions). In particular `max_intervention_size` — the cap on how many
# C-DAG cluster vertices may be jointly intervened on — is a property of the
# benchmark, never of the method, so no method gets a larger search space
# than another. ConfoundedCluster uses cap 1 because it serves only as the
# Tier-2 invariance demonstration (paired correct-vs-misspecified comparisons
# within the same partition); it carries no cross-method performance claim.

PROTOCOL = {
    'CompleteGraph':        {'max_intervention_size': 3},
    'ConfoundedCluster':    {'max_intervention_size': 1},
}


def _max_intervention_size(benchmark):
    return PROTOCOL[benchmark]['max_intervention_size']


def check_fairness(results, context=''):
    """Assert all methods in a main-condition result dict are comparable.

    Enforces exactly the parity claimed in the paper (Sec. 6.1):
      (1) every seed contains the same method set;
      (2) every (seed, method) trajectory has the same length (same trial
          budget + same single initial incumbent);
      (3) every method used the same number of initial interventional points
          (``num_init`` recorded per result). This is the "same number of
          initial interventional points" claim — the budget parity that
          matters — and is what each method's ``define_initial_data_CBO`` is
          handed; it is *not* a claim that the sampled points coincide across
          methods (different partitions induce different exploration sets and
          therefore different interventional designs).
    """
    method_sets = {s: frozenset(results[s].keys()) for s in results}
    if len(set(method_sets.values())) > 1:
        raise AssertionError(
            f"[fairness{context}] method sets differ across seeds: "
            f"{method_sets}")
    lengths = {(s, m): len(results[s][m]['global_opt'])
               for s in results for m in results[s]}
    if len(set(lengths.values())) > 1:
        raise AssertionError(
            f"[fairness{context}] trajectory lengths differ: {lengths}")
    num_inits = {(s, m): results[s][m].get('num_init')
                 for s in results for m in results[s]
                 if results[s][m].get('num_init') is not None}
    if len(set(num_inits.values())) > 1:
        raise AssertionError(
            f"[fairness{context}] initial interventional budgets differ: "
            f"{num_inits}")


# ---------------------------------------------------------------------------
# Benchmark-specific representative coarsenings for 'main' condition
# ---------------------------------------------------------------------------
# All partitions are manipulable-only (Lee-2019 standard):
#   CompleteGraph M = {B, D, E}  →  2 valid coarsenings
#   SimplifiedCoralGraph M = {N, O, C, T, D}  →  47 valid coarsenings (3 representatives)

REPRESENTATIVE_COARSENINGS_COMPLETEGRAPH = [
    # Finest (identity on M): each manipulable variable is its own cluster
    ([frozenset({'B'}), frozenset({'D'}), frozenset({'E'}), frozenset({'Y'})],
     'QCBO-finest'),
    # Coarser: merge D and E into one cluster
    ([frozenset({'B'}), frozenset({'D', 'E'}), frozenset({'Y'})],
     'QCBO-{B},{DE}'),
]

# SimplifiedCoralGraph M = {N, O, C, T, D}
# POMIS for finest = {C, N, O};  for coarsest = {C, N, O, T, D}
REPRESENTATIVE_COARSENINGS_SIMPLIFIEDCORALGRAPH = [
    # Finest: all manipulable singletons — POMIS = {C,N,O}
    ([frozenset({'C'}), frozenset({'D'}), frozenset({'N'}),
      frozenset({'O'}), frozenset({'T'}), frozenset({'Y'})],
     'QCBO-finest'),
    # Natural merge: T and D are only reachable via non-manip mediators (P,CO) — POMIS = {C,N,O}
    ([frozenset({'C', 'N', 'O'}), frozenset({'D', 'T'}), frozenset({'Y'})],
     'QCBO-{CNO},{DT}'),
    # Coarsest: all manipulable in one cluster — POMIS = {C,N,O,T,D}
    ([frozenset({'C', 'D', 'N', 'O', 'T'}), frozenset({'Y'})],
     'QCBO-all-merged'),
]

# ConfoundedCluster M = {A, B, C}
# Coarse partition {A} | {B,C} | {Y} — the partition under which Lee-2019
# latent projection drops the spurious B->C edge (see wrong_edge condition).
REPRESENTATIVE_COARSENINGS_CONFOUNDEDCLUSTER = [
    # Finest (identity on M): each manipulable variable is its own cluster.
    ([frozenset({'A'}), frozenset({'B'}), frozenset({'C'}), frozenset({'Y'})],
     'QCBO-finest'),
    # Coarse: merge the confounded pair B, C into one cluster.
    ([frozenset({'A'}), frozenset({'B', 'C'}), frozenset({'Y'})],
     'QCBO-{A},{BC}'),
]


def _get_representative_coarsenings(benchmark):
    if benchmark == 'CompleteGraph':
        return REPRESENTATIVE_COARSENINGS_COMPLETEGRAPH
    elif benchmark == 'SimplifiedCoralGraph':
        return REPRESENTATIVE_COARSENINGS_SIMPLIFIEDCORALGRAPH
    elif benchmark == 'ConfoundedCluster':
        return REPRESENTATIVE_COARSENINGS_CONFOUNDEDCLUSTER
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

def _run_ccbo_partition(graph, obs, full_obs, benchmark, partition, label,
                        seed, trials, num_interventions, type_cost,
                        initial_num_obs_samples, max_size):
    """Run one CCBO partition through the unified complete-ID estimator
    pipeline (CoarsenedGraph + CBO).

    The identity partition routed through this path *is* CBO on the full DAG
    (Prop. 1): same exploration set, same per-query calibrated estimators. The
    main comparison therefore routes the full-DAG baseline here too, so there
    is no separate hand-coded CBO backend that could differ only in GP-fitting
    hygiene.
    """
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

    _, _, coverage = compute_coverage(obs, manip_vars, dict_ranges)
    x_list, y_list, best_x, opt_y, best_var = define_initial_data_CBO(
        int_data, num_interventions, MIS, 0, 'min')

    (current_cost, _, _, global_opt, _, _) = CBO(
        trials, MIS, manip_vars, x_list, y_list, best_x, opt_y,
        best_var, dict_ranges, functions, obs, coverage, cg,
        20, costs, full_obs, 'min',
        initial_num_obs_samples + 50, initial_num_obs_samples,
        num_interventions, Causal_prior=True)

    return {
        'global_opt': global_opt, 'method': label,
        'num_init': num_interventions,
        'partition': cg.get_partition_description(),
        'es': cg.get_exploration_set_description(),
        'num_es': len(MIS),
    }


def run_main_condition(benchmark, seeds, trials, k_phase, output_dir,
                       num_interventions=10, type_cost=1,
                       initial_num_obs_samples=100):
    """BO + CBO (= CCBO at the identity partition, Prop. 1) + coarser CCBO
    variants, multi-seed.

    The full-DAG CBO baseline and CCBO share a single estimator backend, so
    'CBO' and the finest CCBO coincide by construction; coarsening's value is
    read off the price-of-coarsening sweep and the robustness conditions, not
    a convergence gap at the identity partition.
    """
    results = {s: {} for s in range(seeds)}
    max_size = _max_intervention_size(benchmark)

    for seed in range(seeds):
        print(f"\n{'#'*60}  SEED {seed}  {'#'*60}")
        np.random.seed(seed)

        # BO (no causal prior)
        print(f"\n--- BO ---")
        r = run_bo_baseline(benchmark, trials, num_interventions, type_cost,
                            initial_num_obs_samples, False, 0, 'min', seed=seed)
        results[seed]['BO'] = {'global_opt': r['global_opt'], 'method': 'BO',
                               'num_init': num_interventions}

        # CBO (full DAG) + representative CCBO coarsenings — all through the
        # same CoarsenedGraph backend.
        graph, obs, full_obs = _load_graph(benchmark, initial_num_obs_samples)

        for partition, label in _get_representative_coarsenings(benchmark):
            # The identity partition is CBO on the full DAG (Prop. 1).
            is_finest = all(len(p) == 1 for p in partition)
            run_label = 'CBO' if is_finest else label
            print(f"\n--- {run_label} ---")
            try:
                res = _run_ccbo_partition(
                    graph, obs, full_obs, benchmark, partition, run_label,
                    seed, trials, num_interventions, type_cost,
                    initial_num_obs_samples, max_size)
                results[seed][run_label] = res
                print(f"  Final Y: {res['global_opt'][-1]:.4f}")
            except Exception as e:
                print(f"  {run_label} failed: {e}")
                import traceback; traceback.print_exc()

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
            print(f"  {method:35s}: {np.mean(finals):.3f} ± "
                  f"{np.std(finals, ddof=1) if len(finals) > 1 else 0.0:.3f}")

    check_fairness(results, context=f' {benchmark}/main')

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
    max_size = _max_intervention_size(benchmark)
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
    if benchmark not in ('CompleteGraph', 'ConfoundedCluster'):
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
    else:  # benchmark == 'ConfoundedCluster'
        # Manipulable-only partitions for ConfoundedCluster (M = {A, B, C}).
        finest = [frozenset({v}) for v in ('A', 'B', 'C')] + [frozenset({'Y'})]
        coarser = [
            frozenset({'A'}),
            frozenset({'B', 'C'}),
            frozenset({'Y'}),
        ]

        # WrongBC: spurious intra-cluster edge B -> C inside {B, C}. Together
        # with the U-induced bidirected B <-> C this is a bow, so do(B) is
        # non-identifiable and {B} (the optimal target) is dropped at the
        # fine level. CCBO under the coarser partition projects the edge out,
        # so coarse/correct and coarse/WrongBC are byte-for-byte identical.
        conditions = [
            (finest,  'finest',     'ConfoundedCluster',          'correct'),
            (finest,  'finest',     'ConfoundedCluster_WrongBC',  'WrongBC'),
            (coarser, '{A},{BC}',   'ConfoundedCluster',          'correct'),
            (coarser, '{A},{BC}',   'ConfoundedCluster_WrongBC',  'WrongBC'),
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
        std = np.std(finals, ddof=1) if len(finals) > 1 else 0.0
        print(f"  {pl:15} {ms:8} {np.mean(finals):10.3f} {std:8.3f} {n_es:6d}")

    out = os.path.join(output_dir, f'{benchmark}_wrong_edge_{seeds}seeds.pkl')
    with open(out, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {out}")
    return all_results


# ---------------------------------------------------------------------------
# Condition: sweep  (final Y vs. partition fineness across the lattice)
# ---------------------------------------------------------------------------

def _sweep_partitions(benchmark, per_level=2):
    """Structured sample of the valid-coarsening lattice.

    All valid coarsenings for small lattices; otherwise up to ``per_level``
    partitions per fineness level (num_parts), chosen deterministically
    (lexicographically first), always including the finest and the
    coarsest valid partition.
    """
    coarsenings = enumerate_valid_coarsenings_manip(benchmark)

    def _label(partition):
        manip = [p for p in partition if p != frozenset({'Y'})]
        return 'QCBO-' + '|'.join(
            '{' + ','.join(sorted(p)) + '}' for p in sorted(
                manip, key=lambda c: sorted(c)))

    if len(coarsenings) <= 8:
        return [(c['partition'], _label(c['partition']), c['num_parts'])
                for c in coarsenings]

    by_level = {}
    for c in coarsenings:
        by_level.setdefault(c['num_parts'], []).append(c)
    chosen = []
    levels = sorted(by_level, reverse=True)  # finest first
    for i, lvl in enumerate(levels):
        cands = sorted(by_level[lvl],
                       key=lambda c: _label(c['partition']))
        take = len(cands) if lvl in (levels[0], levels[-1]) else per_level
        chosen.extend(cands[:take])
    return [(c['partition'], _label(c['partition']), c['num_parts'])
            for c in chosen]


def run_sweep_condition(benchmark, seeds, trials, output_dir,
                        num_interventions=10, type_cost=1,
                        initial_num_obs_samples=100):
    """CCBO across the coarsening lattice → final-Y vs. fineness data."""
    partitions = _sweep_partitions(benchmark)
    print(f"Sweep over {len(partitions)} partitions:")
    for _, label, nparts in partitions:
        print(f"  [{nparts} parts] {label}")

    results = {s: {} for s in range(seeds)}
    max_size = _max_intervention_size(benchmark)

    for seed in range(seeds):
        print(f"\n{'#'*60}  SEED {seed}  {'#'*60}")
        graph, obs, full_obs = _load_graph(benchmark, initial_num_obs_samples)

        for partition, label, nparts in partitions:
            print(f"\n--- {label} ---")
            np.random.seed(seed)
            try:
                cg = CoarsenedGraph(graph, partition, benchmark, obs,
                                    max_intervention_size=max_size,
                                    num_mc_samples=2000)
                functions = cg.fit_all_models()
                MIS, _, manip_vars = cg.get_sets()
                dict_ranges = cg.get_interventional_ranges()
                costs = cg.get_cost_structure(type_cost)

                int_data = generate_interventional_data(
                    cg.define_SEM, MIS, dict_ranges,
                    num_points=20, num_mc_samples=2000, seed=seed)
                _, _, coverage = compute_coverage(obs, manip_vars, dict_ranges)
                x_list, y_list, best_x, opt_y, best_var = \
                    define_initial_data_CBO(
                        int_data, num_interventions, MIS, 0, 'min')

                (current_cost, _, _, global_opt, _, _) = CBO(
                    trials, MIS, manip_vars, x_list, y_list, best_x, opt_y,
                    best_var, dict_ranges, functions, obs, coverage, cg,
                    20, costs, full_obs, 'min',
                    initial_num_obs_samples + 50, initial_num_obs_samples,
                    num_interventions, Causal_prior=True)

                results[seed][label] = {
                    'global_opt': global_opt, 'method': label,
                    'num_parts': nparts,
                    'partition': cg.get_partition_description(),
                    'es': cg.get_exploration_set_description(),
                    'num_es': len(MIS),
                }
                print(f"  Final Y: {global_opt[-1]:.4f}  |ES|={len(MIS)}")
            except Exception as e:
                print(f"  {label} failed: {e}")
                import traceback; traceback.print_exc()

        ckpt = os.path.join(output_dir, f'{benchmark}_sweep_{seeds}seeds.pkl')
        with open(ckpt, 'wb') as f:
            pickle.dump(results, f)
        print(f"  [checkpoint saved: seed {seed} complete]")

    print(f"\n{'='*70}\nSWEEP SUMMARY — {benchmark}\n{'='*70}")
    labels = sorted({m for s in results for m in results[s]},
                    key=lambda m: -next(results[s][m]['num_parts']
                                        for s in results if m in results[s]))
    for label in labels:
        finals = [results[s][label]['global_opt'][-1]
                  for s in results if label in results[s]]
        nparts = next(results[s][label]['num_parts']
                      for s in results if label in results[s])
        if finals:
            std = np.std(finals, ddof=1) if len(finals) > 1 else 0.0
            print(f"  [{nparts} parts] {label:45s}: "
                  f"{np.mean(finals):.3f} ± {std:.3f}")

    out = os.path.join(output_dir, f'{benchmark}_sweep_{seeds}seeds.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved {out}")
    return results


# ---------------------------------------------------------------------------
# Condition: rccbo_teaser  (EXCLUDED from the paper experiment matrix —
# see KNOWN_ISSUES.md; kept callable for follow-up work)
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
                        choices=['CompleteGraph', 'ConfoundedCluster'],
                        help='Which benchmark to run')
    parser.add_argument('--condition', default='main',
                        choices=['main', 'wrong_edge', 'sweep'],
                        help='Experiment condition')
    parser.add_argument('--seeds', default=10, type=int,
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
    elif args.condition == 'sweep':
        run_sweep_condition(args.benchmark, args.seeds, args.trials,
                            args.output_dir)


if __name__ == '__main__':
    main()
