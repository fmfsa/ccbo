"""Identity anchor for Prop. 1 (identity recovery): at the all-singleton
partition, QCBO's MIS-based exploration set equals published CBO's.

Three layers of evidence, per graph:

  1. ``compute_MIS`` on the projected full DAG equals the hand-derived
     Lee & Bareinboim 2018 ground truth (X ⊆ M is a MIS iff X ⊆ An(Y) in
     G_X̄).
  2. It equals the vendored graphs' ``get_sets()`` MIS.  The ToyGraph list
     shipped by the CBO-2020 authors included the non-minimal {X, Z} and
     was corrected in this fork.
  3. POMIS ⊆ MIS (a POMIS X satisfies IB(G_X̄) ∩ M = X and IB members are
     ancestors of Y in G_X̄).

Plus the ToyGraph two-tier check: the coarse {X, Z} cluster forms a bow
with Y through the latent U, so its single MIS arm is non-identifiable
from the C-DAG and runs on the uninformative prior tier; at the identity
partition both arms are identifiable (front-door for do(X), backdoor for
do(Z)).
"""

import os

import pandas as pd
import pytest

from ccbo.coarsening import (get_dag_edges_from_sem, project_out_hidden,
                             build_coarsened_admg, compute_MIS,
                             compute_POMIS)
from ccbo.coarsened_graph import CoarsenedGraph
from ccbo.minibench import register_variants
from ccbo.scm_graphs import get_original_graph


DATA = os.path.join(os.path.dirname(__file__), '..', 'cbo', 'data')
register_variants()

# Hand-derived LB18 ground truth (sorted lists of sorted member lists).
EXPECTED_MIS = {
    # X -> Z -> Y, X <-> Y: {X, Z} is NOT minimal (mutilating Z orphans X).
    'ToyGraph': [['X'], ['Z']],
    # X1 -> Y <- X2, X1 <-> X2: both parents keep direct edges to Y.
    'ParallelParent': [['X1'], ['X1', 'X2'], ['X2']],
    # X1 -> M -> Y, X1 <-> Y: {X1, M} is NOT minimal (fixing M orphans X1).
    'FrontDoor': [['M'], ['X1']],
    # X1 -> X2 -> Y: {X1, X2} is NOT minimal.
    'MediatedChain': [['X1'], ['X2']],
}


def _computed_mis(name, partition=None):
    """MIS of the (quotient of the) projected DAG, flattened + sorted."""
    _, nodes, hidden, manip = get_dag_edges_from_sem(name)
    proj = project_out_hidden(name)
    hidden = set(hidden or [])
    nonmanip = set(nodes) - hidden - set(manip) - {'Y'}
    if partition is None:
        partition = [frozenset({v}) for v in manip] + [frozenset({'Y'})]
    admg = build_coarsened_admg(partition, proj, atomic_vertices=nonmanip)
    mnodes = [p for p in partition
              if p != frozenset({'Y'}) and p <= set(manip)]
    mis = compute_MIS(admg, mnodes, frozenset({'Y'}))
    pomis = compute_POMIS(admg, mnodes, frozenset({'Y'}))
    flat = sorted(sorted(v for c in m for v in c) for m in mis)
    return flat, mis, pomis


def _vendored_mis(name):
    """The vendored graph class's get_sets() MIS, sorted."""
    obs = pd.read_pickle(os.path.join(DATA, name, 'observations.pkl'))
    mis, _, _ = get_original_graph(name, obs).get_sets()
    return sorted(sorted(e) for e in mis)


@pytest.mark.parametrize('name', sorted(EXPECTED_MIS))
def test_identity_mis_matches_lb18_ground_truth(name):
    flat, _, _ = _computed_mis(name)
    assert flat == EXPECTED_MIS[name], (
        f'{name}: computed identity-partition MIS deviates from the '
        f'hand-derived LB18 ground truth.\n  computed: {flat}\n'
        f'  expected: {EXPECTED_MIS[name]}')


@pytest.mark.parametrize('name', sorted(EXPECTED_MIS))
def test_identity_mis_matches_vendored_cbo(name):
    flat, _, _ = _computed_mis(name)
    vendored = _vendored_mis(name)
    assert flat == vendored, (
        f'{name}: Prop. 1 anchor broken — computed identity-partition MIS '
        f'differs from the vendored CBO get_sets().\n  computed: {flat}\n'
        f'  vendored: {vendored}')


@pytest.mark.parametrize('name,partition', [
    ('ToyGraph', None),
    ('ToyGraph', [frozenset({'X', 'Z'}), frozenset({'Y'})]),
    ('ParallelParent', None),
    ('ParallelParent', [frozenset({'X1', 'X2'}), frozenset({'Y'})]),
    ('FrontDoor', None),
    ('FrontDoor', [frozenset({'X1', 'M'}), frozenset({'Y'})]),
    ('MediatedChain', None),
    ('MediatedChain', [frozenset({'X1', 'X2'}), frozenset({'Y'})]),
])
def test_pomis_subset_of_mis(name, partition):
    _, mis, pomis = _computed_mis(name, partition)
    non_empty = [p for p in pomis if p]
    assert all(p in mis for p in non_empty), (
        f'{name} ({partition}): POMIS ⊄ MIS — '
        f'POMIS={non_empty}, MIS={mis}')


def test_toygraph_two_tier():
    """Coarse Toy: one MIS arm {X, Z}, non-identifiable (bow) → uninformative
    tier. Identity Toy: {X} and {Z}, both identifiable."""
    obs = pd.read_pickle(os.path.join(DATA, 'ToyGraph', 'observations.pkl'))
    from ccbo.cbo.graphs import ToyGraph
    g = ToyGraph(obs)

    coarse = [frozenset({'X', 'Z'}), frozenset({'Y'})]
    cg_coarse = CoarsenedGraph(g, coarse, 'ToyGraph', obs,
                               num_mc_samples=100)
    assert cg_coarse._exploration_set == [['X', 'Z']]
    assert not cg_coarse._arm_identifiable[('X', 'Z')], (
        'do({X,Z}) must be non-identifiable from the C-DAG (the cluster '
        'forms a bow with Y through the latent U): uninformative tier')

    identity = [frozenset({'X'}), frozenset({'Z'}), frozenset({'Y'})]
    cg_fine = CoarsenedGraph(g, identity, 'ToyGraph', obs,
                             num_mc_samples=100)
    assert cg_fine._exploration_set == [['X'], ['Z']]
    assert cg_fine._arm_identifiable[('X',)], 'do(X) identifiable (front-door)'
    assert cg_fine._arm_identifiable[('Z',)], 'do(Z) identifiable (backdoor)'


if __name__ == '__main__':
    for n in sorted(EXPECTED_MIS):
        test_identity_mis_matches_lb18_ground_truth(n)
        test_identity_mis_matches_vendored_cbo(n)
        print(f'PASS: {n} identity MIS == LB18 == vendored CBO')
    test_toygraph_two_tier()
    print('PASS: ToyGraph two-tier priors')
