"""QDCBO contract tests (fast configs: stat DBN, T=2, small n_obs).

1. quotient structure on the stat DBN (nodes, emission/transition edges);
2. finest-partition identity: QDCBO == stock DCBO trajectory (same seeds);
3. E2 invariance: the intra-cluster edit of the ASSUMED graph leaves the
   QDCBO trajectory byte-identical;
4. singleton-cluster mechanisms == stock ``fit_arcs`` scalar fits
   (prediction-identical).

Both algorithms in a comparison run inside the SAME process, so the stock
code's hash-order-dependent set iteration (see runner docstring) cannot
differ between them.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from ccbo.qdcbo.quotient_dbn import (build_quotient_dbn, ensure_dcbo_on_path,
                                     perturb_temporal_edges)

ensure_dcbo_on_path()

FAST = dict(setup="stat", T=2, trials=5, seed=0, n_obs=20,
            num_anchor_points=25)
COARSE = [["X", "Z"]]
FINEST = [["X"], ["Z"]]


def _stat_graph(T=3):
    from dcbo.utils.dag_utils.graph_functions import make_graphical_model
    return make_graphical_model(0, T - 1, topology="dependent",
                                nodes=["X", "Z", "Y"])


def _edge_set(G):
    return sorted(set((u, v) for u, v in G.edges()))


# ---------------------------------------------------------------------------
# 1. Quotient graph structure
# ---------------------------------------------------------------------------

def test_quotient_structure_stat():
    spec = build_quotient_dbn(_stat_graph(T=3), COARSE, "Y")
    assert dict(spec.clusters) == {"C1": ("X", "Z"), "Y": ("Y",)}
    # Slice-major, causally ordered node insertion.
    assert list(spec.G_quotient.nodes) == [
        "C1_0", "Y_0", "C1_1", "Y_1", "C1_2", "Y_2"]
    assert _edge_set(spec.G_quotient) == [
        ("C1_0", "C1_1"), ("C1_0", "Y_0"),          # transition + emission
        ("C1_1", "C1_2"), ("C1_1", "Y_1"),
        ("C1_2", "Y_2"),
        ("Y_0", "Y_1"), ("Y_1", "Y_2"),             # target identity edges
    ]
    # Parent resolution: member-coordinate tuples, intra-cluster edges gone.
    assert spec.emit_input_nodes("C1", 0) == ()
    assert spec.emit_input_nodes("Y", 1) == ("X_1", "Z_1")
    assert spec.trans_input_nodes("C1", 1) == ("X_0", "Z_0")
    assert spec.trans_input_nodes("Y", 2) == ("Y_1",)
    # ES lifting: whole-cluster patterns in member coordinates, dedup'd.
    assert spec.lift_exploration_sets(
        [("X",), ("Z",), ("X", "Z")]) == [("X", "Z")]


def test_quotient_structure_finest_is_identity():
    G = _stat_graph(T=3)
    spec = build_quotient_dbn(G, FINEST, "Y")
    assert list(spec.G_quotient.nodes) == list(G.nodes)
    assert _edge_set(spec.G_quotient) == _edge_set(G)
    assert spec.lift_exploration_sets([("X",), ("Z",), ("X", "Z")]) == \
        [("X",), ("Z",), ("X", "Z")]


def test_e2_edit_is_invisible_to_the_quotient():
    G = _stat_graph(T=3)
    spec = build_quotient_dbn(G, COARSE, "Y")
    G_e2 = perturb_temporal_edges(G, [("del", "X", "Z")])
    spec_e2 = build_quotient_dbn(G_e2, COARSE, "Y")
    assert list(spec_e2.G_quotient.nodes) == list(spec.G_quotient.nodes)
    assert _edge_set(spec_e2.G_quotient) == _edge_set(spec.G_quotient)


# ---------------------------------------------------------------------------
# 2/3. Trajectory contracts (module-scoped runs, shared across asserts)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runs():
    from ccbo.qdcbo.runner import run_unit, trajectory
    out = {}
    out["dcbo"] = trajectory(run_unit(algo="DCBO", **FAST)[0])
    out["qdcbo_finest"] = trajectory(
        run_unit(algo="QDCBO", partition=FINEST, **FAST)[0])
    out["qdcbo_coarse"] = trajectory(
        run_unit(algo="QDCBO", partition=COARSE, **FAST)[0])
    out["qdcbo_coarse_e2"] = trajectory(
        run_unit(algo="QDCBO", partition=COARSE, misspec="e2", **FAST)[0])
    return out


def test_finest_partition_identity(runs):
    """QDCBO at the finest partition == stock DCBO, same seeds."""
    a, b = runs["dcbo"], runs["qdcbo_finest"]
    assert len(a) == len(b) == FAST["T"]
    for t, (ta, tb) in enumerate(zip(a, b)):
        assert len(ta) == len(tb) == FAST["trials"]
        assert ta == pytest.approx(tb, abs=1e-10), (t, ta, tb)


def test_e2_invariance(runs):
    """The intra-cluster edit changes nothing: byte-identical trajectory."""
    assert runs["qdcbo_coarse"] == runs["qdcbo_coarse_e2"]


def test_coarse_run_shape(runs):
    tr = runs["qdcbo_coarse"]
    assert len(tr) == FAST["T"]
    assert all(len(t) == FAST["trials"] for t in tr)
    # best-so-far is monotone for a minimisation task
    for t in tr:
        assert all(b <= a + 1e-12 for a, b in zip(t, t[1:]))


# ---------------------------------------------------------------------------
# 4. Singleton-cluster mechanisms == stock scalar fits
# ---------------------------------------------------------------------------

def test_singleton_mechanisms_match_stock_fit_arcs():
    from dcbo.utils.sem_utils.sem_estimate import fit_arcs
    from ccbo.qdcbo.qdcbo_base import fit_quotient_arcs
    from ccbo.qdcbo.runner import get_setup, sample_observations

    T = 2
    sem_class, G, *_ = get_setup("stat", T)
    data = sample_observations(sem_class, None, T, 20, seed=0)
    G.T = T

    stock_emit = fit_arcs(G, data, emissions=True)
    stock_trans = fit_arcs(G, data, emissions=False)

    spec = build_quotient_dbn(G, FINEST, "Y")
    q_emit = fit_quotient_arcs(spec, data, emissions=True)
    q_trans = fit_quotient_arcs(spec, data, emissions=False)

    xtest = np.linspace(-3.0, 3.0, 17).reshape(-1, 1)

    # Emissions: Z|X and Y|Z per slice.
    for t in range(T):
        for child, key in (("Z", (f"X_{t}",)), ("Y", (f"Z_{t}",))):
            ms, mq = stock_emit[t][key], q_emit[t][child]["model"]
            mu_s, va_s = ms.predict(xtest)
            mu_q, va_q = mq.predict(xtest)
            assert np.allclose(mu_s, mu_q, atol=1e-12), (t, child)
            assert np.allclose(va_s, va_q, atol=1e-12), (t, child)
        # Source marginal (X): same KDE family, identical scores.
        ks, kq = stock_emit[t][(None, f"X_{t}")], q_emit[t]["X"]["model"]
        assert np.allclose(ks.score_samples(xtest),
                           kq.score_samples(xtest), atol=1e-12)

    # Transitions at t=1: X|X, Z|Z, Y|Y (stock fits each scalar edge).
    for child in ("X", "Z", "Y"):
        ms = stock_trans[1][(f"{child}_0",)]
        mq = q_trans[1][child]["model"]
        mu_s, va_s = ms.predict(xtest)
        mu_q, va_q = mq.predict(xtest)
        assert np.allclose(mu_s, mu_q, atol=1e-12), child
        assert np.allclose(va_s, va_q, atol=1e-12), child


def test_joint_mechanism_is_multi_output():
    """Coarse C1={X,Z}: one ICM GP, inputs = parent member columns, outputs =
    both member columns, and both members' moments come from that single
    posterior."""
    from ccbo.qdcbo.qdcbo_base import JointClusterGP, fit_quotient_arcs
    from ccbo.qdcbo.runner import get_setup, sample_observations

    T = 2
    sem_class, G, *_ = get_setup("stat", T)
    data = sample_observations(sem_class, None, T, 20, seed=0)
    spec = build_quotient_dbn(G, COARSE, "Y")

    q_trans = fit_quotient_arcs(spec, data, emissions=False)
    m = q_trans[1]["C1"]["model"]
    assert isinstance(m, JointClusterGP)
    assert q_trans[1]["C1"]["inputs"] == ("X_0", "Z_0")
    mu, va = m.predict(np.zeros((3, 2)))
    assert mu.shape == (3, 2) and va.shape == (3, 2)
    assert np.all(np.isfinite(mu)) and np.all(va >= 0)

    q_emit = fit_quotient_arcs(spec, data, emissions=True)
    # C1 is a within-slice source: joint 2-D KDE marginal.
    kde = q_emit[0]["C1"]["model"]
    assert kde.sample().shape == (1, 2)
    # Y is singleton but conditions on the whole cluster.
    assert q_emit[0]["Y"]["inputs"] == ("X_0", "Z_0")
