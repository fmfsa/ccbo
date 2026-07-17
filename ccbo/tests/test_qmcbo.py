"""Unit tests for QMCBO's quotient transformation (ccbo/qmcbo/quotient.py).

Pure-python where possible; the torch-dependent target-lifting tests import
torch (present in the ccbo env — MCBO ran on it). The invariance test IS the
mechanism of the E2 result: an intra-cluster (or quotient-redundant) edit of
the fine DAG leaves the quotient view byte-identical.
"""

import pytest

from ccbo.qmcbo.quotient import (quotient_parent_nodes, perturb_parent_nodes,
                                 lift_targets)

# ToyGraph: X0 -> X1 -> X2(Y); partition {X0,X1} | {X2}.
TOY_PARENTS = [[], [0], [1]]
TOY_PART = [[0, 1], [2]]

# PSAGraph: age, bmi, A, S, cancer, Y; partition {A,S} + singletons.
PSA_PARENTS = [[], [0], [0, 1], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
PSA_PART = [[0], [1], [2, 3], [4], [5]]


def test_quotient_parents_toygraph():
    q = quotient_parent_nodes(TOY_PARENTS, TOY_PART)
    # Cluster {X0,X1} is a root cluster (its only edge is internal); Y sees
    # the whole cluster.
    assert q == [[], [], [0, 1]]


def test_quotient_parents_psagraph():
    q = quotient_parent_nodes(PSA_PARENTS, PSA_PART)
    # A and S both see {age, bmi}; cancer sees {age, bmi, A, S}; Y sees all.
    assert q[2] == [0, 1] and q[3] == [0, 1]
    assert q[4] == [0, 1, 2, 3]
    assert q[5] == [0, 1, 2, 3, 4]


def test_invariance_intra_cluster_edits():
    """Every intra-cluster edit of the fine DAG yields an IDENTICAL quotient
    view — the model-based Prop.-2 mirror, and the mechanism of E2."""
    base = quotient_parent_nodes(TOY_PARENTS, TOY_PART)
    for ops in ([("del", 0, 1)], [("rev", 0, 1)]):
        perturbed = perturb_parent_nodes(TOY_PARENTS, ops)
        assert quotient_parent_nodes(perturbed, TOY_PART) == base, ops
    # PSAGraph: add a spurious A->S inside {A,S}.
    base_psa = quotient_parent_nodes(PSA_PARENTS, PSA_PART)
    perturbed = perturb_parent_nodes(PSA_PARENTS, [("add", 2, 3)])
    assert quotient_parent_nodes(perturbed, PSA_PART) == base_psa


def test_quotient_redundant_edge_invisible():
    """Deleting one of two parallel fine edges between the same cluster pair
    leaves the quotient unchanged (inter-but-redundant, Prop.-2 scope)."""
    parents = [[], [], [0, 1], [2], [2, 3]]      # {0,1} -> 2; 2->3; {2,3}->4
    part = [[0, 1], [2], [3], [4]]
    base = quotient_parent_nodes(parents, part)
    perturbed = perturb_parent_nodes(parents, [("del", 1, 2)])
    assert quotient_parent_nodes(perturbed, part) == base


def test_cyclic_quotient_refused():
    """Partial split creating a 2-cycle at cluster level must raise (the
    Prop.-3 validity re-check): a1 -> b -> a2 with {a1,a2} clustered."""
    parents = [[], [0], [1], [2]]                 # a1 -> b -> a2 -> Y
    part = [[0, 2], [1], [3]]                     # cluster {a1, a2}
    with pytest.raises(ValueError, match="cyclic"):
        quotient_parent_nodes(parents, part)


def test_partition_validation():
    with pytest.raises(ValueError, match="two clusters"):
        quotient_parent_nodes(TOY_PARENTS, [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="misses"):
        quotient_parent_nodes(TOY_PARENTS, [[0, 1]])


def test_lift_targets():
    import torch
    targets = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 0]),
               torch.tensor([0, 0, 0])]
    lifted = lift_targets(targets, TOY_PART)
    # do(X0) and do(X1) both lift to do(cluster) and dedupe; null survives.
    assert len(lifted) == 2
    assert [int(x) for x in lifted[0]] == [1, 1, 0]
    assert [int(x) for x in lifted[1]] == [0, 0, 0]


if __name__ == "__main__":
    test_quotient_parents_toygraph()
    test_quotient_parents_psagraph()
    test_invariance_intra_cluster_edits()
    test_quotient_redundant_edge_invisible()
    test_cyclic_quotient_refused()
    test_partition_validation()
    test_lift_targets()
    print("PASS: all qmcbo quotient tests")
