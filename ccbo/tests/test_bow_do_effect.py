"""
Numerical demonstration of the ConfoundedCluster (Tier-2) misspecification gap.

What we show
------------
Under the coarse partition `{A} | {B,C} | {Y}`, the spurious B->C edge in
`ConfoundedCluster_WrongBC` is dropped by Lee-2019 latent projection. Two
observable consequences follow:

  1.  Exploration set: finest/correct includes `{B}`; finest/WrongBC drops it
      because the bow (B -> C plus the U-induced B <-> C bidirected edge)
      breaks do-calculus identification at the fine level. At the coarse level
      both DAGs yield the same exploration set.

  2.  Adjustment: at the coarse level the {B,C}-cluster intervention yields
      byte-identical do-effects under both DAGs (the C-DAG is invariant).

This is a structural numerical demo -- no BO loop. Runs in a few seconds.
"""

import os

import numpy as np
import pandas as pd
import pytest

from ccbo.cbo.graphs import ConfoundedCluster
from ccbo.coarsened_graph import CoarsenedGraph


CLUSTER_VALUES = (0.2, 0.5, 0.8)
COARSE_TOL = 1e-6


def _load_obs(n=200):
    """Subsample to keep C-DAG GP pre-fits fast; the misspec signal does
    not depend on the obs sample size."""
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'cbo', 'data',
        'ConfoundedCluster', 'observations.pkl',
    )
    return pd.read_pickle(data_path).iloc[:n].reset_index(drop=True)


def _build_cg(g, obs, partition, assumed, max_size):
    return CoarsenedGraph(
        g, partition, 'ConfoundedCluster', obs,
        max_intervention_size=max_size,
        num_mc_samples=200,
        assumed_graph_name=assumed,
    )


def _find_do(do_dict, var_set_pred):
    """Return the do-function whose name encodes the predicate, or None."""
    for key, fn in do_dict.items():
        if var_set_pred(key):
            return key, fn
    return None, None


@pytest.mark.slow
def test_confoundedcluster_misspec_gap():
    obs = _load_obs()
    g = ConfoundedCluster(obs)

    finest = [frozenset({v}) for v in ('A', 'B', 'C', 'Y')]
    coarse = [frozenset({'A'}), frozenset({'B', 'C'}), frozenset({'Y'})]

    cg_fine_true = _build_cg(g, obs, finest, 'ConfoundedCluster', 1)
    cg_fine_wrong = _build_cg(g, obs, finest, 'ConfoundedCluster_WrongBC', 1)
    cg_coarse_true = _build_cg(g, obs, coarse, 'ConfoundedCluster', 1)
    cg_coarse_wrong = _build_cg(g, obs, coarse, 'ConfoundedCluster_WrongBC', 1)

    print(f"corr(B, C) in observational data = "
          f"{obs['B'].corr(obs['C']):.3f}\n")

    # ----------------------------------------------------------------------
    # (a) Exploration-set effect of the misspec at the fine level
    # ----------------------------------------------------------------------
    es_fine_true = cg_fine_true._exploration_set
    es_fine_wrong = cg_fine_wrong._exploration_set
    es_coarse_true = cg_coarse_true._exploration_set
    es_coarse_wrong = cg_coarse_wrong._exploration_set

    print("Exploration set under each (partition, assumed DAG):")
    print(f"  finest / ConfoundedCluster        : {es_fine_true}")
    print(f"  finest / ConfoundedCluster_WrongBC: {es_fine_wrong}")
    print(f"  coarse / ConfoundedCluster        : {es_coarse_true}")
    print(f"  coarse / ConfoundedCluster_WrongBC: {es_coarse_wrong}")
    print()

    assert es_fine_true != es_fine_wrong, (
        "Expected the WrongBC misspec to remove at least one entry from "
        "the fine-level exploration set (B becomes non-identifiable via "
        "the B->C + B<->C bow)."
    )
    assert ['B'] in es_fine_true and ['B'] not in es_fine_wrong, (
        "The lost target must be {B} -- the optimal singleton -- for the "
        "misspecification to carry a performance cost."
    )
    assert es_coarse_true == es_coarse_wrong, (
        "Cluster invariance broken: coarse exploration sets differ between "
        "correct and WrongBC DAGs."
    )
    es_lost = {tuple(s) for s in es_fine_true} - {tuple(s) for s in es_fine_wrong}
    print(f"  Misspec cost (fine ES entries CBO can no longer intervene on): "
          f"{[list(e) for e in es_lost]}\n")

    # ----------------------------------------------------------------------
    # (b) Cluster-level do-effect: byte-identical across correct / WrongBC
    # ----------------------------------------------------------------------
    print("Coarse-level cluster intervention E[Y | do(B=C=v)]:")
    print(f"  {'v':>5s}  {'true':>12s}  {'WrongBC':>12s}  {'|Δ|':>10s}")
    print("  " + "-" * 45)

    def _bc_cluster(k):
        return ('B' in k) and ('C' in k) and ('A' not in k)

    coarse_diffs = []
    for v in CLUSTER_VALUES:
        key_t, fn_t = _find_do(cg_coarse_true.get_all_do(), _bc_cluster)
        key_w, fn_w = _find_do(cg_coarse_wrong.get_all_do(), _bc_cluster)
        assert key_t is not None and key_w is not None, (
            "{B,C}-cluster do-function missing in one of the coarse C-DAGs."
        )
        val = np.array([v, v])
        m_t, _ = fn_t(obs, None, val)
        m_w, _ = fn_w(obs, None, val)
        coarse_diffs.append(abs(m_t - m_w))
        print(f"  {v:5.2f}  {m_t:12.4f}  {m_w:12.4f}  {abs(m_t - m_w):10.2e}")
    print()

    assert max(coarse_diffs) < COARSE_TOL, (
        f"Cluster invariance violated: max |Δ| = {max(coarse_diffs):.2e} "
        f">= {COARSE_TOL:.2e}. Both fine DAGs should project to the same "
        f"C-DAG and yield identical adjustments."
    )

    print("PASS: WrongBC drops {B} from the fine ES; coarse ES is "
          "invariant; coarse adjustments match exactly.")


if __name__ == '__main__':
    test_confoundedcluster_misspec_gap()
