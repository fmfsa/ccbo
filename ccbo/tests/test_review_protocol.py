"""Protocol v3 contracts: null recommendation, mixed-union refinement, ClusterChain.

Pure checks (no CBO backend): the recommendation rule, refinement arm
generation, ClusterChain closed forms against Monte Carlo, analytic prices,
and the code's arm families against the analysis' independent enumeration.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from ccbo import minibench as mb
from ccbo.matched_protocol import (Experiment, all_arms, recommend, sample_true, score_events,
                                   observational_data, regret_start, keyed_seed)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from matched_refinement import apply_stage, mixed_unions  # noqa: E402

_spec = importlib.util.spec_from_file_location("paper_analysis", ROOT / "experiments" / "analyze.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


def _event(i, arm, x, y, cost):
    return dict(event_id=i, phase="init", arm=arm, x=x, measured={"Y": y}, cost=len(arm), cum_cost=cost)


# ---------------------------------------------------------------------------
# Null intervention
# ---------------------------------------------------------------------------

def test_null_is_recommended_only_when_strictly_better():
    events = [_event(0, ["X1"], [0.], 5., 1), _event(1, ["X2"], [0.], 3., 2)]
    assert recommend(events) == 1
    assert recommend(events, null_estimate=3.) == 1          # ties keep the executed intervention
    assert recommend(events, null_estimate=2.9) is None


def test_null_optimal_scm_is_attainable_only_through_null():
    # The colleague's example: X = U, Y = (X - U)^2. Every do(X = x) has value
    # x^2 + Var(U) > 0 = E[Y], so only the null recommendation attains y*.
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, 200000)
    null_mean = np.mean((u - u) ** 2)
    do_values = [np.mean((x - u) ** 2) for x in np.linspace(-1, 1, 21)]
    assert null_mean == 0 and min(do_values) > 0.3
    events = [_event(i, ["X"], [x], float(v), i + 1) for i, (x, v) in enumerate(zip([0., .5], do_values[10:12]))]
    assert recommend(events, null_estimate=float(null_mean)) is None


def test_scoring_values_null_recommendation_by_population_mean():
    events = [_event(0, ["X1", "X2"], [0., 0.], 9., 2)]
    scores = score_events(mb.PP_NAME, events, 2, null_estimate=1.)
    assert scores["final"]["recommendation_event_id"] is None
    assert scores["final"]["recommendation_population"] == pytest.approx(mb.null_value(mb.PP_NAME))
    assert analyze.population(mb.PP_NAME, [], []) == pytest.approx(mb.null_value(mb.PP_NAME))


@pytest.mark.parametrize("scm", list(mb.NODES))
def test_null_value_matches_monte_carlo_and_analysis(scm):
    rng = np.random.RandomState(5)
    ys = np.array([sample_true(scm, {}, rng.randn(mb.NOISE_DIM[scm]))["Y"] for _ in range(60000)])
    assert abs(ys.mean() - mb.null_value(scm)) < 6 * ys.std() / np.sqrt(len(ys))
    assert analyze.population(scm, [], []) == pytest.approx(mb.null_value(scm))


def test_shared_seed_namespace_and_null_only_changes_recommendations():
    # Observational data and initial levels are keyed by the frozen v2 namespace.
    import ccbo.matched_protocol as mp
    assert mp.SEED_NAMESPACE == "matched-controlled-noisy-v2" != mp.PROTOCOL_ID
    a = Experiment(mb.PP_NAME, 1000, all_arms(mb.PP_NAME))
    b = Experiment(mb.PP_NAME, 1000, all_arms(mb.PP_NAME), null_estimate=-1e9)
    assert [e["measured"] for e in a.events] == [e["measured"] for e in b.events]
    assert all(e["recommendation_event_id"] is None for e in b.events)


# ---------------------------------------------------------------------------
# Mixed-union refinement
# ---------------------------------------------------------------------------

def test_two_cluster_split_exposes_mixed_unions():
    A, B, C, D = "A1", "A2", "D1", "D2"
    clusters = [frozenset({A, B}), frozenset({C, D})]
    stage = {clusters[0]: [frozenset({A}), frozenset({B})],
             clusters[1]: [frozenset({C}), frozenset({D})]}
    after = apply_stage(clusters, stage)
    arms = set(mixed_unions(after))
    assert ("A1", "D1") in arms and len(arms) == 15
    # Splitting one cluster still mixes its children with the other cluster.
    one = set(mixed_unions(apply_stage(clusters, {clusters[0]: stage[clusters[0]]})))
    assert ("A1", "D1", "D2") in one and len(one) == 7


def test_clusterchain_first_stage_exposes_the_fine_optimal_arm():
    clusters = [c for c in mb.partition(mb.CC_NAME, "pairs") if c != frozenset({"Y"})]
    after = apply_stage(clusters, mb.REFINE_STAGES[mb.CC_NAME][0])
    new = set(mixed_unions(after)) - set(mixed_unions(clusters))
    assert ("A1", "A2", "B2", "D1", "D2") in new and len(new) == 8
    assert mb.CC_N_INIT * sum(map(len, new)) <= mb.CC_BUDGET - 2 * 24


def test_mediatedchain_refinement_arms_unchanged():
    clusters = [frozenset({"X1", "X2"})]
    after = apply_stage(clusters, mb.REFINE_STAGES[mb.MC_NAME][0])
    assert set(mixed_unions(after)) - {("X1", "X2")} == {("X1",), ("X2",)}


# ---------------------------------------------------------------------------
# ClusterChain
# ---------------------------------------------------------------------------

def test_clusterchain_closed_forms_match_monte_carlo():
    rng = np.random.RandomState(11)
    arms = all_arms(mb.CC_NAME)
    for arm in [arms[i] for i in rng.choice(len(arms), 12, replace=False)] + [("A1",), ("A1", "A2", "B2", "D1", "D2")]:
        x = [rng.uniform(*mb.domain(mb.CC_NAME, v)) for v in arm]
        iv = dict(zip(arm, x))
        ys = np.array([sample_true(mb.CC_NAME, iv, rng.randn(9))["Y"] for _ in range(20000)])
        assert abs(ys.mean() - mb.population_do(mb.CC_NAME, arm, x)) < 6 * ys.std() / np.sqrt(len(ys)) + 1e-8
        assert analyze.population(mb.CC_NAME, arm, x) == pytest.approx(mb.population_do(mb.CC_NAME, arm, x))


def test_clusterchain_prices():
    o = mb.ORACLE[mb.CC_NAME]
    assert o["y_star"] == pytest.approx(0.04)
    assert o["v_pi"] == pytest.approx({"fine": 0.04, "alt": 0.04, "pairs": 1.29, "coarse": 4.0})
    assert mb.cc_do({"A1": 2, "A2": -2, "B2": 1, "D1": 1, "D2": -1}) == pytest.approx(0.04)
    assert mb.cc_mu_star(["A1", "A2", "B2", "D2"]) - o["y_star"] == pytest.approx(1.5)   # CBO's floor under K1
    assert regret_start(mb.CC_NAME) == 0 and regret_start(mb.PP_NAME) == 0


def test_clusterchain_observations_shape():
    obs = observational_data(mb.CC_NAME, 1000)
    assert list(obs.columns) == mb.CC_NODES and obs.shape == (100, 7)
    assert abs(obs["B1"].corr(obs["A1"])) > 0.8 and abs(obs["D1"].corr(obs["D2"])) > 0.2


@pytest.mark.parametrize("cond", ["K0", "K1", "K2", "K3"])
@pytest.mark.parametrize("pid", ["fine", "alt", "pairs", "coarse"])
def test_clusterchain_arms_match_independent_enumeration(cond, pid):
    from ccbo.tests.test_minibench_structure import _build
    cg = _build(mb.CC_NAME, mb.partition(mb.CC_NAME, pid), cond)
    code = {tuple(sorted(a)) for a in cg._exploration_set}
    assert code == analyze.cc_arms(cond, pid)


def test_clusterchain_quotient_protection_flags():
    from ccbo.tests.test_minibench_structure import _build, _admg_signature
    for pid in ("alt", "pairs", "coarse"):
        part = mb.partition(mb.CC_NAME, pid)
        base = _admg_signature(_build(mb.CC_NAME, part, "K0")._coarsened_admg)
        for pert in (p for p in mb.PERTURBATIONS if p["scm"] == mb.CC_NAME and p["id"] != "K0"):
            same = _admg_signature(_build(mb.CC_NAME, part, pert["id"])._coarsened_admg) == base
            assert same == pert["protected"], (pid, pert["id"])


def test_k3_keeps_fine_identification_but_changes_the_functional():
    from ccbo.tests.test_minibench_structure import _build
    fine = mb.fine_partition(mb.CC_NAME)
    for cond in ("K0", "K3"):
        cg = _build(mb.CC_NAME, fine, cond)
        assert cg._arm_gate_status[("A2",)] == "identified"
    true_adm = _build(mb.CC_NAME, fine, "K0")._coarsened_admg
    wrong_adm = _build(mb.CC_NAME, fine, "K3")._coarsened_admg
    assert len(true_adm.get("bi", ())) > len(wrong_adm.get("bi", ()))


# ---------------------------------------------------------------------------
# CBO-aligned protocol: free initial points, exact feedback, observation pool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scm", list(mb.NODES))
def test_initial_points_are_free_and_measurements_exact(scm):
    e = Experiment(scm, 1000, all_arms(scm))
    assert e.cost == 0 and all(x["cost"] == 0 and x["cum_cost"] == 0 and x["trial"] == 0 for x in e.events)
    assert sum(x["phase"] == "init" for x in e.events) == 3 * len(all_arms(scm))
    arm = all_arms(scm)[-1]
    e.purchase(arm, [mb.domain(scm, v)[0] for v in arm])
    last = e.events[-1]
    assert last["cost"] == len(arm) and last["trial"] == 1 and e.trial == 1
    for x in e.events:
        assert x["measured"]["Y"] == pytest.approx(mb.population_do(scm, x["arm"], x["x"]))
    e.observed(120)
    assert e.trial == 2 and e.observation_log == [dict(trial=2, cum_cost=len(arm), n_rows_after=120)]


def test_observation_pool_extends_initial_rows():
    small, pool = observational_data(mb.FD_NAME, 1000), observational_data(mb.FD_NAME, 1000, 150)
    assert len(small) == 100 and len(pool) == 150 and pool.iloc[:100].equals(small)


def test_prior_only_surrogate_matches_engine_prior_and_hands_over():
    from ccbo.cbo.utils.BO_functions import PriorOnlyModel, update_BO_models
    mean = lambda X: (X[:, :1] - 1.0) ** 2
    var = lambda X: np.full((X.shape[0], 1), 0.5)
    m = update_BO_models(mean, var, np.zeros((0, 1)), np.zeros((0, 1)), True)
    assert isinstance(m, PriorOnlyModel) and m.model is None
    mu, v = m.predict(np.array([[3.0]]))
    assert mu[0, 0] == pytest.approx(4.0) and v[0, 0] == pytest.approx(1.5, rel=1e-6)
    dmu, dv = m.get_prediction_gradients(np.array([[3.0]]))
    assert dmu[0, 0] == pytest.approx(4.0, rel=1e-4) and dv[0, 0] == pytest.approx(0.0, abs=1e-6)
    m.set_data(np.array([[3.0]]), np.array([[4.2]]))
    assert m.model is not None and m.predict(np.array([[3.0]]))[0][0, 0] == pytest.approx(4.2, abs=1e-3)
    plain = update_BO_models(None, None, np.zeros((0, 2)), np.zeros((0, 1)), False)
    assert plain.predict(np.zeros((1, 2)))[0][0, 0] == 0.0
