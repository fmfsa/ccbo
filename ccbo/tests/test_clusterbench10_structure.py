"""Structural validation of the ClusterBench10 misspecification benchmark.

These tests prove the experimental premise *before* any SCM coefficient tuning
or CBO runs, and need no dataset files: the coarsened C-DAG and the gated
exploration set are derived purely from the (registered) assumed structure +
partition (see CoarsenedGraph._build_coarsened_structure). We feed a stub
original graph that only has to expose get_interventional_ranges().

Claims checked (these ARE the conditions-characterization):
  1. Coarse partition {X1,X2,X3}|{X4,X5}: the C-DAG signature is IDENTICAL
     under the correct DAG and every *intra*-cluster perturbation (P1/P2/P3 and
     the sweep S1/S2/S3) -> QCBO-coarse is provably invariant (Delta==0).
  2. Coarse partition: the C-DAG signature DIFFERS under the *inter*-cluster
     controls (P4/P5) -> QCBO-coarse is NOT immune (Prop. 2's scope).
  3. Finest partition: the bow P1 (add X1->X2) is visible and deletes the
     best singleton arm do(X1) from the gated exploration set (Tier-2,
     unrecoverable) -- non-vacuity of claim (1).
"""

import numpy as np
import pandas as pd

from ccbo.coarsened_graph import CoarsenedGraph
from ccbo import clusterbench10 as cb


# Coarse + finest partitions (manipulable clusters + {Y}; non-manip observables
# are auto-lifted to atomic singletons by build_coarsened_admg).
COARSE = [frozenset(c) for c in cb.COARSE_CLUSTERS] + [frozenset({"Y"})]
FINEST = [frozenset({v}) for v in cb.MANIPULATIVE] + [frozenset({"Y"})]


class _StubGraph:
    """Minimal original-graph stand-in: only ranges are read at construction."""

    def get_interventional_ranges(self):
        from collections import OrderedDict
        return OrderedDict((v, [-3.0, 3.0]) for v in cb.MANIPULATIVE)


def _obs():
    # Columns must exist (stored as attributes); values are irrelevant to the
    # purely-structural C-DAG / exploration-set construction.
    return pd.DataFrame({n: np.zeros(4) for n in cb.NODES})


def _admg_signature(admg):
    """Order-independent fingerprint of the CoarsenedGraph C-DAG."""
    di = frozenset(tuple(sorted(map(str, e))) for e in admg.get("di", set()))
    bi = frozenset(frozenset(map(str, e)) for e in admg.get("bi", set()))
    vs = frozenset(map(str, admg.get("vertices", set())))
    return (vs, di, bi)


def _build(partition, perturbation_id, max_size):
    cb.register_variants()
    cg = CoarsenedGraph(
        _StubGraph(), partition, cb.NAME, _obs(),
        max_intervention_size=max_size, num_mc_samples=50,
        assumed_graph_name=cb.variant_name(perturbation_id),
    )
    return cg


def test_coarse_invariance_matches_predicted_protection():
    """The coarse C-DAG is invariant EXACTLY for the perturbations whose
    ``protected`` flag is True -- the three-way conditions characterization:
      * intra-cluster (P1/P2/P3, sweep)      -> invariant
      * inter & redundant (P6: X2->M2 alive) -> invariant
      * inter & non-redundant (P4/P5)        -> C-DAG changes
    """
    sig0 = _admg_signature(_build(COARSE, "P0", max_size=2)._coarsened_admg)
    for p in cb.PERTURBATIONS:
        if p["id"] == "P0":
            continue
        sig = _admg_signature(_build(COARSE, p["id"], max_size=2)._coarsened_admg)
        invariant = (sig == sig0)
        assert invariant == p["protected"], (
            f"{p['id']} ({p['locus']}): predicted protected={p['protected']} but "
            f"coarse C-DAG invariant={invariant}")


def test_intercluster_bow_deletes_cluster_arm_at_coarse():
    """The inter-cluster bow Pic (add X1->X5 on the X1<->X5 confounded pair)
    forms a cluster-level bow C1->C2 + C1<->C2 -> do(C1) is non-identifiable at
    the COARSE level and is dropped from the exploration set, while the *intra*
    bow P1 leaves the coarse ES untouched. This is the exact scope of Prop. 2:
    QCBO-coarse is protected iff the bow is abstracted away (intra), not when it
    spans clusters (inter)."""
    C1 = sorted(cb.COARSE_CLUSTERS[0])  # ['X1','X2','X3']
    es_p0 = _build(COARSE, "P0", max_size=2)._exploration_set
    es_intra = _build(COARSE, "P1", max_size=2)._exploration_set
    es_inter = _build(COARSE, "Pic", max_size=2)._exploration_set
    assert C1 in es_p0, f"do(C1) should be identifiable under the true DAG; ES={es_p0}"
    assert C1 in es_intra, (
        f"intra bow must NOT remove do(C1) from the coarse ES; ES={es_intra}")
    assert C1 not in es_inter, (
        f"inter-cluster bow must delete do(C1) from the coarse ES; ES={es_inter}")


def test_finest_bow_deletes_best_singleton():
    """At the finest partition the bow P1 deletes do(X1) from the gated ES."""
    cg_true = _build(FINEST, "P0", max_size=5)
    cg_bow = _build(FINEST, "P1", max_size=5)

    sig_true = _admg_signature(cg_true._coarsened_admg)
    sig_bow = _admg_signature(cg_bow._coarsened_admg)
    assert sig_true != sig_bow, "Finest partition should see the X1->X2 edge"

    assert ["X1"] in cg_true._exploration_set, (
        f"do(X1) should be identifiable under the correct DAG; "
        f"ES={cg_true._exploration_set}")
    assert ["X1"] not in cg_bow._exploration_set, (
        f"do(X1) should be deleted under the bow (X1->X2 + X1<->X2); "
        f"ES={cg_bow._exploration_set}")


if __name__ == "__main__":
    test_coarse_invariance_matches_predicted_protection()
    print("PASS: coarse C-DAG invariance matches predicted protection (3-way)")
    test_intercluster_bow_deletes_cluster_arm_at_coarse()
    print("PASS: inter-cluster bow deletes do(C1) at coarse; intra bow does not")
    test_finest_bow_deletes_best_singleton()
    print("PASS: finest bow deletes best singleton do(X1)")
