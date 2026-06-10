"""
Validate C-DAG do-function accuracy against true SEM Monte Carlo estimates.

For each benchmark and each valid manipulable-only coarsening, compares the
GP-based causal effect predictions from ``make_cdag_do_function`` (ananke
GID-PO gated) against ground-truth ``E[Y | do(X=x)]`` computed via the true
SEM (mutilated model + forward sampling).

Pipeline under test: Lee-2019 latent projection → coarsened ADMG → ananke
OneLineID identifiability gate → BD/FD/g-comp estimator on the expanded
DAG.  This is the full Plan §4 stack.
"""

import os
import numpy as np
import pandas as pd
import pytest

from ccbo.adjustment import make_cdag_do_function
from ccbo.coarsening import enumerate_valid_coarsenings_manip, compute_POMIS
from ccbo.generic_do import intervene, sample_from_model
from ccbo.run_experiment import get_original_graph


def true_do_mc(sem_fn, intervention_vars, values, num_samples=50000, seed=42):
    """Compute E[Y|do(intervention_vars=values)] via true SEM MC sampling."""
    model = sem_fn()
    intervention_dict = {var: float(val)
                         for var, val in zip(intervention_vars, values)}
    mutilated = intervene(intervention_dict, model)
    np.random.seed(seed)
    samples = [sample_from_model(mutilated) for _ in range(num_samples)]
    y_values = np.array([s['Y'] for s in samples])
    return float(np.mean(y_values)), float(np.var(y_values))


def evaluate_graph(graph_name, num_test_points=5, max_fine_vars=3, verbose=True):
    """
    Test all POMIS-based interventions under every valid manipulable-only
    coarsening of a benchmark.

    Returns a list of result dicts with RMSE and identification info.
    """
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'cbo', 'data', graph_name)
    observational_samples = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))[:200]

    true_obs = None
    true_obs_path = os.path.join(data_path, 'true_observations.pkl')
    if os.path.exists(true_obs_path):
        true_obs = pd.read_pickle(true_obs_path)

    original_graph = get_original_graph(graph_name, observational_samples, true_obs)
    sem_fn = original_graph.define_SEM

    coarsenings = enumerate_valid_coarsenings_manip(graph_name)
    dict_ranges = original_graph.get_interventional_ranges()

    results = []

    for c_idx, coarsening in enumerate(coarsenings):
        partition = coarsening['partition']
        admg = coarsening['coarsened_admg']
        manip_clusters = coarsening['manip_clusters']

        part_str = ' | '.join(
            '{' + ','.join(sorted(p)) + '}' for p in partition)

        # POMIS on the coarsened ADMG; each entry is a set of manipulable
        # cluster vertices to be jointly intervened on.
        target_node = frozenset({'Y'})
        pomis_sets = compute_POMIS(admg, manip_clusters, target_node)

        for pomis_entry in pomis_sets:
            # Flatten to fine-grained manipulative vars (joint-cluster do)
            fine_vars = []
            for cluster in pomis_entry:
                fine_vars.extend(sorted(cluster))
            fine_vars = sorted(fine_vars)
            if not fine_vars or len(fine_vars) > max_fine_vars:
                continue

            do_fn, ident = make_cdag_do_function(
                admg, fine_vars, partition, 'Y', observational_samples)
            method = ident.get('method', 'none')

            if do_fn is None:
                if verbose:
                    print(f"  [{part_str}] do({fine_vars}): "
                          f"NOT IDENTIFIABLE (ananke={ident.get('identifiable')})")
                results.append({
                    'graph': graph_name,
                    'coarsening_idx': c_idx,
                    'partition': part_str,
                    'intervention': fine_vars,
                    'method': 'none',
                    'rmse': None,
                    'identifiable': ident.get('identifiable', False),
                })
                continue

            np.random.seed(123)
            test_values = [
                [np.random.uniform(*dict_ranges[v]) for v in fine_vars]
                for _ in range(num_test_points)
            ]

            errors = []
            for vals in test_values:
                pred_mean, _ = do_fn(observational_samples, None, vals)
                true_mean, _ = true_do_mc(sem_fn, fine_vars, vals)
                errors.append((pred_mean - true_mean) ** 2)

            rmse = float(np.sqrt(np.mean(errors)))
            if verbose:
                status = "OK" if rmse < 0.5 else "HIGH"
                print(f"  [{part_str}] do({fine_vars}): {method} "
                      f"→ RMSE={rmse:.4f} [{status}]")

            results.append({
                'graph': graph_name,
                'coarsening_idx': c_idx,
                'partition': part_str,
                'intervention': fine_vars,
                'method': method,
                'rmse': rmse,
                'identifiable': True,
            })

    return results


@pytest.mark.slow
def test_completegraph_do_accuracy():
    """All identifiable C-DAG do-functions track the true SEM do-effect.

    RMSE < 0.5 on a target whose range spans several units; loose enough to
    absorb GP-fit noise, tight enough to catch a wrong adjustment formula.
    """
    results = evaluate_graph('CompleteGraph', num_test_points=3,
                             verbose=False)
    identifiable = [r for r in results if r['rmse'] is not None]
    assert identifiable, "No identifiable do-functions found on CompleteGraph"
    bad = [r for r in identifiable if r['rmse'] > 0.5]
    assert not bad, (
        "Do-functions with RMSE > 0.5 vs true SEM: "
        + "; ".join(f"[{r['partition']}] do({r['intervention']}) "
                    f"rmse={r['rmse']:.3f} ({r['method']})" for r in bad))


def main():
    print("=" * 60)
    print("Do-Function Accuracy Validation (Lee-2019 + ananke GID-PO)")
    print("=" * 60)

    benchmarks = ['CompleteGraph']
    sc_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'cbo', 'data', 'SimplifiedCoralGraph', 'observations.pkl')
    if os.path.exists(sc_path):
        benchmarks.append('SimplifiedCoralGraph')

    for graph_name in benchmarks:
        print(f"\n--- {graph_name} ---")
        results = evaluate_graph(graph_name, num_test_points=5)

        identifiable = [r for r in results if r['rmse'] is not None]
        non_ident = [r for r in results if r['rmse'] is None]
        if identifiable:
            rmses = [r['rmse'] for r in identifiable]
            print(f"\n  Summary: {len(identifiable)} identifiable, "
                  f"{len(non_ident)} non-identifiable")
            print(f"  RMSE: mean={np.mean(rmses):.4f}, "
                  f"max={np.max(rmses):.4f}, min={np.min(rmses):.4f}")
            high = [r for r in identifiable if r['rmse'] > 0.5]
            if high:
                print(f"  WARNING: {len(high)} with RMSE > 0.5:")
                for r in high:
                    print(f"    [{r['partition']}] do({r['intervention']}): "
                          f"RMSE={r['rmse']:.4f} [{r['method']}]")

    print("\n" + "=" * 60)
    print("Validation complete.")


if __name__ == '__main__':
    main()
