"""
Direct test of the headline LightTunnel claim:

Under the coarse partition {R,G,B} | {P1,P2} | {Y}, the spurious intra-cluster
edge R -> G in LightTunnel_WrongRG is dropped by Lee-2019 latent projection
and the resulting C-DAG is byte-identical to LightTunnel's. Therefore the
adjustment formulas CCBO derives from the assumed fine DAG are the same
whether the assumed DAG is correct or misspecified.

This test bypasses the (expensive) CBO loop and proves the structural
invariance directly. The full CBO regression is the multi-seed run launched
via ``run_experiments.py --benchmark LightTunnel --condition wrong_edge``;
this test runs in seconds and is the regression you actually want in CI.
"""

import os
import pickle

import pandas as pd

from ccbo.coarsened_graph import CoarsenedGraph
from ccbo.cbo.graphs import LightTunnel


COARSE_PARTITION = [
    frozenset({'R', 'G', 'B'}),
    frozenset({'P1', 'P2'}),
    frozenset({'Y'}),
]


def _admg_signature(admg):
    """Order-independent fingerprint of the CoarsenedGraph C-DAG."""
    di = frozenset(tuple(sorted(map(str, e))) for e in admg.get('di', set()))
    bi = frozenset(frozenset(map(str, e)) for e in admg.get('bi', set()))
    vs = frozenset(map(str, admg.get('vertices', set())))
    return (vs, di, bi)


def test_coarse_cdag_invariant_to_intra_cluster_edge():
    """The C-DAG over the coarse partition is identical for both fine DAGs."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'cbo', 'data',
                             'LightTunnel', 'observations.pkl')
    obs = pd.read_pickle(data_path)
    g = LightTunnel(obs)

    cg_true = CoarsenedGraph(
        g, COARSE_PARTITION, 'LightTunnel', obs,
        max_intervention_size=2, num_mc_samples=200,
        assumed_graph_name='LightTunnel',
    )
    cg_wrong = CoarsenedGraph(
        g, COARSE_PARTITION, 'LightTunnel', obs,
        max_intervention_size=2, num_mc_samples=200,
        assumed_graph_name='LightTunnel_WrongRG',
    )

    sig_true = _admg_signature(cg_true._coarsened_admg)
    sig_wrong = _admg_signature(cg_wrong._coarsened_admg)
    assert sig_true == sig_wrong, (
        f"Coarsening should hide the intra-cluster R->G edge but C-DAGs differ:\n"
        f"  true:  {sig_true}\n"
        f"  wrong: {sig_wrong}"
    )

    # Identification machinery should also agree on which exploration-set
    # entries are identifiable from the C-DAG.
    assert cg_true._exploration_set == cg_wrong._exploration_set, (
        f"Exploration sets differ:\n"
        f"  true:  {cg_true._exploration_set}\n"
        f"  wrong: {cg_wrong._exploration_set}"
    )


def test_finest_partition_records_wrong_edge():
    """Sanity: with the finest partition, the wrong edge IS visible in the C-DAG.

    This is the converse direction: if both partitions hid the misspec, the
    test above would be vacuous.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'cbo', 'data',
                             'LightTunnel', 'observations.pkl')
    obs = pd.read_pickle(data_path)
    g = LightTunnel(obs)
    finest = [frozenset({v}) for v in ('R', 'G', 'B', 'P1', 'P2', 'Y')]

    cg_true = CoarsenedGraph(
        g, finest, 'LightTunnel', obs,
        max_intervention_size=1, num_mc_samples=200,
        assumed_graph_name='LightTunnel',
    )
    cg_wrong = CoarsenedGraph(
        g, finest, 'LightTunnel', obs,
        max_intervention_size=1, num_mc_samples=200,
        assumed_graph_name='LightTunnel_WrongRG',
    )

    sig_true = _admg_signature(cg_true._coarsened_admg)
    sig_wrong = _admg_signature(cg_wrong._coarsened_admg)
    assert sig_true != sig_wrong, (
        f"Finest partition should see the wrong R->G edge but signatures match:\n"
        f"  {sig_true}"
    )


if __name__ == "__main__":
    test_coarse_cdag_invariant_to_intra_cluster_edge()
    print("PASS: coarse C-DAG invariant under R->G misspecification")
    test_finest_partition_records_wrong_edge()
    print("PASS: finest partition does see the R->G misspecification")
