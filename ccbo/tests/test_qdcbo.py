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

Requires the DCBO authors' stack in third_party/DCBO — run
``scripts/fetch_dcbo.sh`` first; otherwise this module is skipped.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from ccbo.qdcbo.quotient_dbn import (build_quotient_dbn, ensure_dcbo_on_path,
                                     perturb_temporal_edges, _DCBO_ROOT)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_DCBO_ROOT),
    reason="third_party/DCBO missing — run scripts/fetch_dcbo.sh")

if os.path.isdir(_DCBO_ROOT):
    ensure_dcbo_on_path()

FAST = dict(setup="stat", T=2, trials=5, seed=0, n_obs=20,
            num_anchor_points=25)
COARSE = [["X", "Z"]]
FINEST = [["X"], ["Z"]]


def _stat_graph(T=3):
    from ccbo.qdcbo.quotient_dbn import make_temporal_graph
    return make_temporal_graph(0, T - 1, topology="dependent",
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

@pytest.fixture(scope="module", params=[True, False], ids=["stock", "fixed"])
def runs(request):
    """Both semantics: ``stock`` reproduces the authors' backend, ``fixed``
    (engine v3 default) applies the corrected transition fits and
    is-not-None clamping to baseline and quotient alike."""
    from ccbo.qdcbo.runner import run_unit, trajectory
    q = request.param
    out = {"stock_quirks": q}
    out["dcbo"] = trajectory(run_unit(algo="DCBO", stock_quirks=q, **FAST)[0])
    out["qdcbo_finest"] = trajectory(
        run_unit(algo="QDCBO", partition=FINEST, stock_quirks=q, **FAST)[0])
    out["qdcbo_coarse"] = trajectory(
        run_unit(algo="QDCBO", partition=COARSE, stock_quirks=q, **FAST)[0])
    out["qdcbo_coarse_e2"] = trajectory(
        run_unit(algo="QDCBO", partition=COARSE, misspec="e2", stock_quirks=q,
                 **FAST)[0])
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


# ---------------------------------------------------------------------------
# 5. Engine v3: corrected stock semantics
# ---------------------------------------------------------------------------

def test_stock_fixes_apply_patches_every_binding():
    from ccbo.qdcbo import stock_fixes as sf
    import dcbo.bases.dcbo_base as db, dcbo.methods.cbo as cbo_mod
    import dcbo.utils.gp_utils as gu, dcbo.utils.sequential_sampling as ss
    import dcbo.utils.sem_utils.sem_estimate as se
    sf.apply(False)
    assert db.fit_arcs is sf.fixed_fit_arcs and cbo_mod.fit_arcs is sf.fixed_fit_arcs
    assert se.fit_arcs is sf.fixed_fit_arcs
    for m in (gu, ss, db):
        assert m.sequential_sample_from_SEM_hat is sf.fixed_sequential_sample_from_SEM_hat
    sf.apply(True)
    assert db.fit_arcs is sf._STOCK["fit_arcs"]
    assert ss.sequential_sample_from_SEM_hat is sf._STOCK["sampler"]
    sf.apply(False)


def test_clamp_zero_is_honoured_in_both_samplers():
    """do(X = 0.0) is an intervention: the corrected samplers clamp it, the
    stock semantics drop it (value 7 from the mechanism)."""
    from collections import OrderedDict
    from ccbo.qdcbo import stock_fixes as sf
    from ccbo.qdcbo.qdcbo_base import sequential_sample_from_qsem_hat
    # --- stock-shaped sampler: every variable is a source node returning 7
    static = OrderedDict((v, (lambda t, *a, **k: 7.0)) for v in ("X", "Z", "Y"))
    iv = {v: [0.0] for v in ("X", "Z", "Y")}
    stock = sf._STOCK["sampler"](static, None, 1, lambda V, t: (), interventions=iv)
    fixed = sf.fixed_sequential_sample_from_SEM_hat(static, None, 1, lambda V, t: (),
                                                    interventions=iv)
    assert all(stock[v][0] == 7.0 for v in iv)
    assert all(fixed[v][0] == 0.0 for v in iv)
    # --- quotient sampler on the finest stat partition
    spec = build_quotient_dbn(_stat_graph(T=2), FINEST, "Y")
    qstatic = OrderedDict((C, (lambda t, *a, **k: 7.0)) for C in spec.cluster_order)
    ivq = {v: [0.0, None] for v in spec.base_order}
    q_stock = sequential_sample_from_qsem_hat(qstatic, None, 1, spec, interventions=ivq,
                                              stock_quirks=True)
    q_fixed = sequential_sample_from_qsem_hat(qstatic, None, 1, spec, interventions=ivq,
                                              stock_quirks=False)
    assert all(q_stock[v][0] == 7.0 for v in spec.base_order)
    assert all(q_fixed[v][0] == 0.0 for v in spec.base_order)


def test_transition_fit_uses_the_lagged_parent():
    """With X_t = X_{t-1} + 1 exactly, the corrected transition mechanism
    predicts x + 1 from x; the stock quirk (regress X_1 on X_1) predicts x."""
    from ccbo.qdcbo.qdcbo_base import fit_quotient_arcs
    rng = np.random.default_rng(0)
    n = 40
    x0 = rng.uniform(0.0, 1.0, n)
    data = {"X": np.column_stack([x0, x0 + 1.0]),
            "Z": np.column_stack([x0 * 0.5, x0 * 0.5 + 1.0]),
            "Y": np.column_stack([x0 * 0.2, x0 * 0.2 + 1.0])}
    spec = build_quotient_dbn(_stat_graph(T=2), FINEST, "Y")
    fixed = fit_quotient_arcs(spec, data, emissions=False, stock_quirks=False)
    stock = fit_quotient_arcs(spec, data, emissions=False, stock_quirks=True)
    C = spec.cluster_of("X")
    xq = np.array([[0.5]])
    m_fixed = float(np.ravel(fixed[1][C]["model"].predict(xq)[0])[0])
    m_stock = float(np.ravel(stock[1][C]["model"].predict(xq)[0])[0])
    assert abs(m_fixed - 1.5) < 0.15, m_fixed
    assert abs(m_stock - 0.5) < 0.15, m_stock


def test_decisions_payload_records_every_trial(tmp_path):
    from ccbo.qdcbo.runner import run_unit, decisions_payload
    model, _ = run_unit(algo="DCBO", stock_quirks=False, **FAST)
    pl = decisions_payload(model, "DCBO", FAST["setup"], FAST["T"], FAST["trials"],
                           FAST["seed"], "", False)
    import json
    (tmp_path / "d.json").write_text(json.dumps(pl))
    back = json.loads((tmp_path / "d.json").read_text())
    assert len(back["per_t"]) == FAST["T"]
    for rec in back["per_t"]:
        assert len(rec["best_so_far"]) == FAST["trials"]
        assert len(rec["per_trial_cost"]) == FAST["trials"]
        assert rec["chosen_sets"] and all(isinstance(c, list) for c in rec["chosen_sets"])
    assert back["unit"]["stock_quirks"] is False
