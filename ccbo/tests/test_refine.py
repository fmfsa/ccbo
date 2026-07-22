"""Unit tests for QCBO-refine (plateau-triggered partition refinement).

Dataset-free where possible (same stub-graph style as
test_clusterbench10_structure.py): the trigger, the split, the carryover, and
the refined-ES structural claims are all pure functions of the registered
structure + partition. ClusterBench10 shares ClusterBench10L's structure, so
the structural assertions (do(X1,X3) appears post-split; refined ES is a
superset) carry to the lossy demo.
"""

import numpy as np
import pandas as pd
import pytest

from ccbo import coarsening
from ccbo import clusterbench10 as cb
from ccbo.benchmark import _plateau, _split_partition, _union_and_carry
from ccbo.coarsened_graph import CoarsenedGraph

COARSE = [frozenset(c) for c in cb.COARSE_CLUSTERS] + [frozenset({"Y"})]
REFINED = ([frozenset({v}) for v in cb.COARSE_CLUSTERS[0]]
           + [frozenset(cb.COARSE_CLUSTERS[1])] + [frozenset({"Y"})])


class _StubGraph:
    def get_interventional_ranges(self):
        from collections import OrderedDict
        return OrderedDict((v, [-3.0, 3.0]) for v in cb.MANIPULATIVE)


def _obs(nodes=None):
    return pd.DataFrame({n: np.zeros(4) for n in (nodes or cb.NODES)})


def _build(partition):
    cb.register_variants()
    return CoarsenedGraph(
        _StubGraph(), partition, cb.NAME, _obs(),
        num_mc_samples=50,
        assumed_graph_name=cb.variant_name("P0"))


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

def test_plateau_trigger():
    k, d = 3, 1e-3
    # Not enough history: never fires.
    assert not _plateau([1.0, 1.0, 1.0], k, d, "min")
    # Still improving by more than delta over the window.
    assert not _plateau([2.0, 1.5, 1.2, 1.0], k, d, "min")
    # Flat (and noisy-flat) windows fire.
    assert _plateau([1.0, 1.0, 1.0, 1.0], k, d, "min")
    assert _plateau([1.0, 1.0, 1.0, 1.0 - 5e-4], k, d, "min")
    # Max task mirrors min.
    assert _plateau([1.0, 1.0, 1.0, 1.0], k, d, "max")
    assert not _plateau([1.0, 1.2, 1.5, 2.0], k, d, "max")


# ---------------------------------------------------------------------------
# Split policy
# ---------------------------------------------------------------------------

def test_split_partition_splits_only_incumbent_cluster():
    rmap = {("X1", "X2", "X3"): [["X1"], ["X2"], ["X3"]],
            ("X4", "X5"): [["X4"], ["X5"]]}
    # Incumbent touches C1 only -> only C1 splits.
    new, split = _split_partition(COARSE, rmap, ("X1", "X2", "X3"))
    assert split == [frozenset({"X1", "X2", "X3"})]
    assert frozenset({"X4", "X5"}) in new
    assert frozenset({"X1"}) in new and frozenset({"X2"}) in new
    # Incumbent touches no refinable cluster -> no-op.
    new, split = _split_partition(COARSE, {}, ("X1", "X2", "X3"))
    assert split == [] and new == list(COARSE)
    # A refine map that is not a partition of its cluster is rejected.
    bad = {("X1", "X2", "X3"): [["X1"], ["X2"]]}
    with pytest.raises(AssertionError):
        _split_partition(COARSE, bad, ("X1", "X2", "X3"))


# ---------------------------------------------------------------------------
# Warm start: data carryover + phase-1 arm union
# ---------------------------------------------------------------------------

def test_union_and_carry_by_arm_key():
    ES_old = [["X1", "X2", "X3"], ["X4", "X5"]]
    dxl = [np.arange(6, dtype=float).reshape(2, 3), np.ones((2, 2))]
    dyl = [np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])]
    # Refined ES drops the joint coarse arm (MIS membership is not
    # refinement-monotone in general) and adds two new singleton arms.
    ES_new = [["X1"], ["X3"], ["X4", "X5"]]
    sampled = []

    def sampler(entries):
        sampled.extend(tuple(e) for e in entries)
        return ([np.full((5, len(e)), 9.0) for e in entries],
                [np.full((5, 1), 9.0) for e in entries])

    ES, ndxl, ndyl = _union_and_carry(ES_old, dxl, dyl, ES_new, sampler)
    # Every phase-1 arm persists (the dropped coarse arm is unioned back).
    assert ["X1", "X2", "X3"] in ES and ["X4", "X5"] in ES
    # Carried arms keep their exact arrays; only genuinely new arms sampled.
    i_old = ES.index(["X1", "X2", "X3"])
    assert ndxl[i_old] is dxl[0] and ndyl[i_old] is dyl[0]
    i_c2 = ES.index(["X4", "X5"])
    assert ndxl[i_c2] is dxl[1]
    assert set(sampled) == {("X1",), ("X3",)}
    assert len(ES) == len(ndxl) == len(ndyl) == 4


# ---------------------------------------------------------------------------
# Structural claims on the ClusterBench10 family
# ---------------------------------------------------------------------------

def test_refined_es_superset_and_contains_optimum_arm():
    """Splitting C1 to singletons: on ClusterBench10 the refined MIS ES
    strictly contains the coarse one (every cluster member keeps a directed
    path to Y, so every coarse cluster-union stays minimal after the split)
    and gains do(X1,X3) — the ClusterBench10L optimum arm the coarse
    partition cannot express."""
    es_coarse = [tuple(e) for e in _build(COARSE)._exploration_set]
    es_refined = [tuple(e) for e in _build(REFINED)._exploration_set]
    assert set(es_coarse) <= set(es_refined)
    assert ("X1", "X3") in es_refined
    assert ("X1", "X3") not in es_coarse
    assert len(es_refined) > len(es_coarse)


def test_cyclic_refinement_refused():
    """Prop. 3 operationalized: a declared split whose quotient is cyclic is
    refused (CoarsenedGraph raises). Coarse cluster A={a1,a2,a3} with intra
    edges a1->a3->a2 is valid; the partial split {a1,a2}|{a3} creates the
    2-cycle {a1,a2} -> {a3} -> {a1,a2}."""
    name = "RefineCyclicToy"
    nodes = ["a1", "a2", "a3", "Y"]
    coarsening.register_graph(
        name, dag_edges=[("a1", "a3"), ("a3", "a2"), ("a1", "Y"),
                         ("a2", "Y"), ("a3", "Y")],
        nodes=nodes, hidden_nodes=[],
        manipulative_variables=["a1", "a2", "a3"], confounders=[])

    class _Stub3:
        def get_interventional_ranges(self):
            from collections import OrderedDict
            return OrderedDict((v, [-1.0, 1.0]) for v in ["a1", "a2", "a3"])

    valid = [frozenset({"a1", "a2", "a3"}), frozenset({"Y"})]
    CoarsenedGraph(_Stub3(), valid, name, _obs(nodes),
                   num_mc_samples=50)  # no raise

    cyclic = [frozenset({"a1", "a2"}), frozenset({"a3"}), frozenset({"Y"})]
    with pytest.raises(ValueError, match="cyclic"):
        CoarsenedGraph(_Stub3(), cyclic, name, _obs(nodes),
                       num_mc_samples=50)


if __name__ == "__main__":
    test_plateau_trigger()
    test_split_partition_splits_only_incumbent_cluster()
    test_union_and_carry_by_arm_key()
    test_refined_es_superset_and_contains_optimum_arm()
    test_cyclic_refinement_refused()
    print("PASS: all refine tests")
