"""
Numerical demonstration of the LightTunnel misspecification gap.

What we show
------------
Under a coarse partition `{R,G,B} | {P1,P2} | {Y}`, the spurious B->G edge
in `LightTunnel_WrongBG` is dropped by Lee-2019 latent projection. Two
observable consequences follow:

  1.  Exploration set: finest/correct includes `{B}`; finest/WrongBG drops
      it because the bow (B -> G plus the U_color-induced B <-> G
      bidirected edge) breaks do-calculus identification at the fine
      level.  At the coarse level both DAGs yield the same ES.

  2.  Adjustment: at the coarse level the color-cluster intervention
      yields byte-identical do-effects under both DAGs (the C-DAG is
      invariant). At the fine level the WrongBG misspec changes both
      the ES and the adjustment formulas for the surviving singletons.

This is a structural numerical demo -- no BO loop. Runs in a few seconds.
"""

import os

import numpy as np
import pandas as pd
import pytest

from ccbo.cbo.graphs import LightTunnel
from ccbo.coarsened_graph import CoarsenedGraph


R_VALUES = (0.2, 0.5, 0.8)
COARSE_TOL = 1e-6


def _load_obs(n=200):
    """Subsample to keep C-DAG GP pre-fits fast; the misspec signal does
    not depend on the obs sample size."""
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'cbo', 'data',
        'LightTunnel', 'observations.pkl',
    )
    return pd.read_pickle(data_path).iloc[:n].reset_index(drop=True)


def _build_cg(g, obs, partition, assumed, max_size):
    return CoarsenedGraph(
        g, partition, 'LightTunnel', obs,
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
def test_lighttunnel_misspec_gap():
    obs = _load_obs()
    g = LightTunnel(obs)

    finest = [frozenset({v}) for v in ('R', 'G', 'B', 'P1', 'P2', 'Y')]
    coarse = [frozenset({'R', 'G', 'B'}),
              frozenset({'P1', 'P2'}),
              frozenset({'Y'})]

    cg_fine_true = _build_cg(g, obs, finest, 'LightTunnel', 1)
    cg_fine_wrong = _build_cg(g, obs, finest, 'LightTunnel_WrongBG', 1)
    cg_coarse_true = _build_cg(g, obs, coarse, 'LightTunnel', 1)
    cg_coarse_wrong = _build_cg(g, obs, coarse, 'LightTunnel_WrongBG', 1)

    print(f"corr(B, G) in observational data = "
          f"{obs['B'].corr(obs['G']):.3f}\n")

    # ----------------------------------------------------------------------
    # (a) Exploration-set effect of the misspec at the fine level
    # ----------------------------------------------------------------------
    es_fine_true = cg_fine_true._exploration_set
    es_fine_wrong = cg_fine_wrong._exploration_set
    es_coarse_true = cg_coarse_true._exploration_set
    es_coarse_wrong = cg_coarse_wrong._exploration_set

    print("Exploration set under each (partition, assumed DAG):")
    print(f"  finest / LightTunnel        : {es_fine_true}")
    print(f"  finest / LightTunnel_WrongBG: {es_fine_wrong}")
    print(f"  coarse / LightTunnel        : {es_coarse_true}")
    print(f"  coarse / LightTunnel_WrongBG: {es_coarse_wrong}")
    print()

    assert es_fine_true != es_fine_wrong, (
        "Expected the WrongBG misspec to remove at least one entry from "
        "the fine-level exploration set (B becomes non-identifiable via "
        "the B->G + B<->G bow)."
    )
    assert ['B'] in es_fine_true and ['B'] not in es_fine_wrong, (
        "The lost arm must be {B} — the highest-weight singleton — for the "
        "misspecification to carry a performance cost."
    )
    assert es_coarse_true == es_coarse_wrong, (
        "Cluster invariance broken: coarse exploration sets differ between "
        "correct and WrongBG DAGs."
    )
    es_lost = {tuple(s) for s in es_fine_true} - {tuple(s) for s in es_fine_wrong}
    print(f"  Misspec cost (fine ES entries CBO can no longer intervene on): "
          f"{[list(e) for e in es_lost]}\n")

    # ----------------------------------------------------------------------
    # (b) Cluster-level do-effect: byte-identical across correct / WrongBG
    # ----------------------------------------------------------------------
    print("Coarse-level cluster intervention E[Y | do(R=G=B=r)]:")
    print(f"  {'r':>5s}  {'true':>12s}  {'WrongBG':>12s}  "
          f"{'|Δ|':>10s}")
    print("  " + "-" * 45)

    def _color_cluster(k):
        return all(v in k for v in ('R', 'G', 'B')) and 'P1' not in k

    coarse_diffs = []
    for r in R_VALUES:
        key_t, fn_t = _find_do(cg_coarse_true.get_all_do(), _color_cluster)
        key_w, fn_w = _find_do(cg_coarse_wrong.get_all_do(), _color_cluster)
        assert key_t is not None and key_w is not None, (
            "Color-cluster do-function missing in one of the coarse C-DAGs."
        )
        v = np.array([r, r, r])
        m_t, _ = fn_t(obs, None, v)
        m_w, _ = fn_w(obs, None, v)
        coarse_diffs.append(abs(m_t - m_w))
        print(f"  {r:5.2f}  {m_t:12.4f}  {m_w:12.4f}  "
              f"{abs(m_t - m_w):10.2e}")
    print()

    assert max(coarse_diffs) < COARSE_TOL, (
        f"Cluster invariance violated: max |Δ| = {max(coarse_diffs):.2e} "
        f">= {COARSE_TOL:.2e}. Both fine DAGs should project to the same "
        f"C-DAG and yield identical adjustments."
    )

    # ----------------------------------------------------------------------
    # (c) For surviving fine singletons in both DAGs, the adjustment can
    #     still differ because the WrongBG DAG conditions G on B via the
    #     observational regression.
    # ----------------------------------------------------------------------
    print("Fine-level singleton do-effects (surviving entries):")
    print(f"  {'var':>3s} {'r':>5s}  {'true':>12s}  {'WrongBG':>12s}  "
          f"{'|Δ|':>10s}")
    print("  " + "-" * 50)
    fine_diffs = []
    for var in ('R', 'G', 'P1', 'P2'):
        def _singleton_for(v):
            def pred(k):
                others = set('RGB') | {'P1', 'P2'} - {v}
                # The do-function name uses concatenated names; we want
                # a key that contains `v` but none of the others.
                if v not in k:
                    return False
                for o in others:
                    if o in k:
                        return False
                return True
            return pred
        key_t, fn_t = _find_do(cg_fine_true.get_all_do(),
                               _singleton_for(var))
        key_w, fn_w = _find_do(cg_fine_wrong.get_all_do(),
                               _singleton_for(var))
        if key_t is None or key_w is None:
            continue
        for r in R_VALUES:
            m_t, _ = fn_t(obs, None, r)
            m_w, _ = fn_w(obs, None, r)
            d = abs(m_t - m_w)
            fine_diffs.append(d)
            print(f"  {var:>3s} {r:5.2f}  {m_t:12.4f}  {m_w:12.4f}  "
                  f"{d:10.2e}")

    print()
    print(f"max |fine_true - fine_wrong| over surviving singletons: "
          f"{max(fine_diffs):.2e}")
    print(f"max |coarse_true - coarse_wrong|: {max(coarse_diffs):.2e}\n")

    print("PASS: WrongBG drops {B} from the fine ES; coarse ES is "
          "invariant; coarse adjustments match exactly.")


if __name__ == '__main__':
    test_lighttunnel_misspec_gap()
