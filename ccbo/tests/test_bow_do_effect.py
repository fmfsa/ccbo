"""
Numerical demonstration of the ConfoundedCluster misspecification gap.

What we show
------------
Under the coarse partition `{A} | {B,C} | {Y}`, the spurious B->C edge in
`ConfoundedCluster_WrongBC` is dropped by Lee-2019 latent projection. Two
observable consequences follow:

  1.  Prior tier: the exploration set is the MIS of the assumed graph and
      does not change (membership is never gated), but at the fine level
      the WrongBC bow (B -> C plus the U-induced B <-> C bidirected edge)
      breaks do-calculus identification of `do(B)`, so the `{B}` arm falls
      from the do-calculus prior to the common uninformative prior. At the
      coarse level both DAGs yield the same C-DAG, hence identical tiers.

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


def _build_cg(g, obs, partition, assumed):
    return CoarsenedGraph(
        g, partition, 'ConfoundedCluster', obs,
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

    cg_fine_true = _build_cg(g, obs, finest, 'ConfoundedCluster')
    cg_fine_wrong = _build_cg(g, obs, finest, 'ConfoundedCluster_WrongBC')
    cg_coarse_true = _build_cg(g, obs, coarse, 'ConfoundedCluster')
    cg_coarse_wrong = _build_cg(g, obs, coarse, 'ConfoundedCluster_WrongBC')

    print(f"corr(B, C) in observational data = "
          f"{obs['B'].corr(obs['C']):.3f}\n")

    # ----------------------------------------------------------------------
    # (a) Prior-tier effect of the misspec at the fine level
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

    assert ['B'] in es_fine_true and ['B'] in es_fine_wrong, (
        "{B} must be a MIS arm under both assumed DAGs -- membership is "
        "never gated on identifiability."
    )
    assert cg_fine_true._arm_identifiable[('B',)], (
        "Under the correct DAG do(B) must be identifiable (do-calculus "
        "prior tier)."
    )
    assert not cg_fine_wrong._arm_identifiable[('B',)], (
        "Under WrongBC the B->C + B<->C bow must break identification of "
        "do(B): the arm stays but drops to the uninformative prior tier."
    )
    assert es_coarse_true == es_coarse_wrong, (
        "Cluster invariance broken: coarse exploration sets differ between "
        "correct and WrongBC DAGs."
    )
    assert (cg_coarse_true._arm_identifiable
            == cg_coarse_wrong._arm_identifiable), (
        "Cluster invariance broken: coarse prior tiers differ between "
        "correct and WrongBC DAGs."
    )
    tier_flips = {k for k in cg_fine_true._arm_identifiable
                  if cg_fine_true._arm_identifiable[k]
                  != cg_fine_wrong._arm_identifiable.get(k)}
    print(f"  Misspec cost (fine arms whose prior tier flips): "
          f"{sorted(tier_flips)}\n")

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

    print("PASS: WrongBC flips {B} to the uninformative prior tier; coarse "
          "ES, tiers, and adjustments are invariant.")


if __name__ == '__main__':
    test_confoundedcluster_misspec_gap()
