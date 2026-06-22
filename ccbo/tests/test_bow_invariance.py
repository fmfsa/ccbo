"""
Direct test of the headline ConfoundedCluster (Tier-2) claim:

Under the coarse partition {A} | {B,C} | {Y}, the spurious intra-cluster edge
B -> C in ConfoundedCluster_WrongBC is dropped by Lee-2019 latent projection
and the resulting C-DAG is byte-identical to ConfoundedCluster's. Therefore
the exploration set and the adjustment formulas QCBO derives are the same
whether the assumed fine DAG is correct or misspecified.

This test bypasses the (expensive) CBO loop and proves the structural
invariance directly. The full CBO regression is the multi-seed run launched
via ``run_experiments.py --benchmark ConfoundedCluster --condition wrong_edge``;
this test runs in seconds and is the regression you actually want in CI.
"""

import os
import pandas as pd

from ccbo.coarsened_graph import CoarsenedGraph
from ccbo.cbo.graphs import ConfoundedCluster


COARSE_PARTITION = [
    frozenset({'A'}),
    frozenset({'B', 'C'}),
    frozenset({'Y'}),
]


def _admg_signature(admg):
    """Order-independent fingerprint of the CoarsenedGraph C-DAG."""
    di = frozenset(tuple(sorted(map(str, e))) for e in admg.get('di', set()))
    bi = frozenset(frozenset(map(str, e)) for e in admg.get('bi', set()))
    vs = frozenset(map(str, admg.get('vertices', set())))
    return (vs, di, bi)


def _load_obs():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'cbo', 'data',
                             'ConfoundedCluster', 'observations.pkl')
    return pd.read_pickle(data_path)


def test_coarse_cdag_invariant_to_intra_cluster_edge():
    """The C-DAG over the coarse partition is identical for both fine DAGs."""
    obs = _load_obs()
    g = ConfoundedCluster(obs)

    cg_true = CoarsenedGraph(
        g, COARSE_PARTITION, 'ConfoundedCluster', obs,
        max_intervention_size=1, num_mc_samples=200,
        assumed_graph_name='ConfoundedCluster',
    )
    cg_wrong = CoarsenedGraph(
        g, COARSE_PARTITION, 'ConfoundedCluster', obs,
        max_intervention_size=1, num_mc_samples=200,
        assumed_graph_name='ConfoundedCluster_WrongBC',
    )

    sig_true = _admg_signature(cg_true._coarsened_admg)
    sig_wrong = _admg_signature(cg_wrong._coarsened_admg)
    assert sig_true == sig_wrong, (
        f"Coarsening should hide the intra-cluster B->C edge but C-DAGs differ:\n"
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


def test_finest_partition_deletes_best_arm():
    """With the finest partition, the wrong edge IS visible and deletes {B}.

    This is the converse direction: if both partitions hid the misspec, the
    invariance test above would be vacuous. It also pins down the Tier-2
    cost: the deleted target {B} is the optimal singleton.
    """
    obs = _load_obs()
    g = ConfoundedCluster(obs)
    finest = [frozenset({v}) for v in ('A', 'B', 'C', 'Y')]

    cg_true = CoarsenedGraph(
        g, finest, 'ConfoundedCluster', obs,
        max_intervention_size=1, num_mc_samples=200,
        assumed_graph_name='ConfoundedCluster',
    )
    cg_wrong = CoarsenedGraph(
        g, finest, 'ConfoundedCluster', obs,
        max_intervention_size=1, num_mc_samples=200,
        assumed_graph_name='ConfoundedCluster_WrongBC',
    )

    sig_true = _admg_signature(cg_true._coarsened_admg)
    sig_wrong = _admg_signature(cg_wrong._coarsened_admg)
    assert sig_true != sig_wrong, (
        f"Finest partition should see the wrong B->C edge but signatures match:\n"
        f"  {sig_true}"
    )

    # Tier-2: the bow (B->C plus the U-induced B<->C) makes do(B) non-
    # identifiable, so {B} -- the best singleton -- is dropped at the fine
    # level but present under the correct DAG.
    assert ['B'] in cg_true._exploration_set, (
        f"{{B}} should be identifiable under the correct DAG; "
        f"ES={cg_true._exploration_set}")
    assert ['B'] not in cg_wrong._exploration_set, (
        f"{{B}} should be deleted under WrongBC (bow non-identifiability); "
        f"ES={cg_wrong._exploration_set}")


if __name__ == "__main__":
    test_coarse_cdag_invariant_to_intra_cluster_edge()
    print("PASS: coarse C-DAG invariant under B->C misspecification")
    test_finest_partition_deletes_best_arm()
    print("PASS: finest partition deletes the best arm {B} under WrongBC")
