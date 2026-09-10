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


# ---------------------------------------------------------------------------
# Joint cluster mechanisms (WP1)
# ---------------------------------------------------------------------------
# These need the MCBO authors' stack (scripts/fetch_mcbo.sh) AND a botorch
# contemporary with it (FixedNoiseGP / botorch.sampling.samplers — see the
# `mcbo` conda env); on a newer botorch they skip rather than error.

def _mcbo_available():
    import os as _os
    from ccbo.qmcbo.quotient import _MCBO_ROOT
    if not _os.path.isdir(_MCBO_ROOT):
        return False, "third_party/mcbo missing — run scripts/fetch_mcbo.sh"
    try:
        import botorch.sampling.samplers  # noqa: F401  (removed in >=0.8)
        from botorch.models import FixedNoiseGP  # noqa: F401
    except ImportError:
        return False, ("installed botorch too new for the MCBO authors' "
                       "stack — use the era-pinned `mcbo` env")
    return True, ""


_MCBO_OK, _MCBO_SKIP_REASON = _mcbo_available()
joint_net = pytest.mark.skipif(not _MCBO_OK, reason=_MCBO_SKIP_REASON)


def _toy_joint_net(parent_nodes, partition, n=60, seed=0, intra_coef=0.9):
    """Fit a JointQuotientGPNetwork on synthetic ToyGraph-shaped data where
    X1 = intra_coef * X0 + noise inside cluster {0,1} and Y = X1 + noise."""
    import torch
    from ccbo.qmcbo.quotient import ensure_mcbo_on_path
    ensure_mcbo_on_path()
    from mcbo.utils.dag import DAG
    from ccbo.qmcbo.qgp_network import JointQuotientGPNetwork
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    x0 = torch.randn(n)
    x1 = intra_coef * x0 + 0.3 * torch.randn(n)
    y = x1 + 0.1 * torch.randn(n)
    train_Y = torch.stack([x0, x1, y], dim=-1)
    train_X = torch.zeros(n, 6)          # do-flags all zero + values unused
    profile = {
        "dag": DAG(parent_nodes), "interventional": True,
        "valid_targets": [torch.tensor([0, 0, 0])],
        "additive_noise_dists": None, "input_dim": 6,
        "do_map": lambda X: X, "active_input_indices": [[]] * 6,
    }
    algo = {"algo": "MCBO", "beta": 1.0}
    return JointQuotientGPNetwork(train_X, train_Y, algo, profile, partition)


@joint_net
def test_joint_cluster_posterior_covariance():
    """The 2-member cluster mechanism must learn the within-cluster residual
    correlation (X1 = 0.9 X0 + small noise ⇒ predictive correlation near
    +0.95). At fixed cluster inputs this correlation is aleatoric, so it
    lives in the full-rank task NOISE — the predictive (observation-noise)
    posterior must carry it."""
    import torch
    net = _toy_joint_net(TOY_PARENTS, TOY_PART)
    gp = net.cluster_GPs[0]              # cluster {X0, X1}, root (const feat)
    post = gp.posterior(torch.zeros(1, 1), observation_noise=True)
    cov = post.mvn.covariance_matrix.reshape(2, 2)
    corr = cov[0, 1] / (cov[0, 0].sqrt() * cov[1, 1].sqrt())
    assert float(corr) > 0.5, f"residual correlation not learned: {float(corr)}"


@joint_net
def test_joint_singleton_reduces_to_per_node():
    """All-singleton partition: cluster inputs == fine parents and every
    mechanism is a stock single-output GP (Q-Identity at the model level)."""
    from botorch.models.gp_regression import SingleTaskGP
    net = _toy_joint_net(TOY_PARENTS, [[0], [1], [2]])
    assert net.cluster_inputs == [[], [0], [1]]     # == fine parents
    assert all(isinstance(g, SingleTaskGP) for g in net.cluster_GPs)


@joint_net
def test_joint_invariance_to_intra_cluster_edits():
    """The joint network's structure (cluster parents + inputs) is identical
    under intra-cluster edits of the fine DAG — Theorem-1 factoring for the
    joint formulation."""
    net_a = _toy_joint_net(TOY_PARENTS, TOY_PART)
    net_b = _toy_joint_net(perturb_parent_nodes(TOY_PARENTS, [("rev", 0, 1)]),
                           TOY_PART)
    assert net_a.cluster_parents == net_b.cluster_parents
    assert net_a.cluster_inputs == net_b.cluster_inputs
    assert net_a.cluster_order == net_b.cluster_order


if __name__ == "__main__":
    test_quotient_parents_toygraph()
    test_quotient_parents_psagraph()
    test_invariance_intra_cluster_edits()
    test_quotient_redundant_edge_invisible()
    test_cyclic_quotient_refused()
    test_partition_validation()
    test_lift_targets()
    print("PASS: all qmcbo quotient tests")



# ---------------------------------------------------------------------------
# Engine v3: zero-range normalisation guard (pure helper, no MCBO stack)
# ---------------------------------------------------------------------------

def test_normalize_columns_zero_range():
    torch = pytest.importorskip("torch")
    from ccbo.qmcbo.quotient import normalize_columns
    aux = torch.tensor([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]], dtype=torch.float64)
    lo = [aux[:, 0].min(), aux[:, 1].min()]
    hi = [aux[:, 0].max(), aux[:, 1].max()]
    out = normalize_columns(aux.clone(), lo, hi)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(out[:, 1], torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
