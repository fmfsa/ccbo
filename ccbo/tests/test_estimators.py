"""Engine v3 estimator contracts (ccbo/adjustment.py).

The front-door and g-computation functionals must *integrate* the fitted
intermediate conditionals.  The oracle-conditional tests inject the true
conditionals of the paper's own SCMs, so they isolate the estimator's
integration from GP regression error (the pure mean-substitution bias on
FrontDoor is 0.336 -> 0 at x1 = 0.5).
"""

import numpy as np
import pandas as pd
import pytest

from ccbo import adjustment as adj
from ccbo import minibench as mb


# --------------------------------------------------------------- front-door
def _fd_oracle_conditionals():
    """True FrontDoor conditionals: M | X1 ~ N(2 x1, 0.2^2);
    E[Y | M=m, X1=x'] = 2(1 - exp(-(m-1)^2 / (2 0.3^2))) + 0.64 x'
    (E[U | X1] = 0.64 X1), Var(Y | m, x') = 0.0676."""
    m_cond = lambda X: (2.0 * X[:, :1], np.full((X.shape[0], 1), 0.04))

    def y_cond(X):
        m, x1 = X[:, 0], X[:, 1]
        mu = 2.0 * (1.0 - np.exp(-(m - 1.0) ** 2 / (2 * 0.09))) + 0.64 * x1
        return mu[:, None], np.zeros((X.shape[0], 1)), 0.0676

    S_obs = np.linspace(-1.0, 1.0, 101)[:, None]   # symmetric: mean(0.64 x') = 0
    return m_cond, y_cond, S_obs


@pytest.mark.parametrize("x1, truth", [(0.5, mb.fd_do_x1(0.5)), (0.0, mb.fd_do_x1(0.0)),
                                       (1.0, mb.fd_do_x1(1.0)), (-1.0, mb.fd_do_x1(-1.0))])
def test_frontdoor_oracle_conditional_integration(x1, truth):
    m_cond, y_cond, S_obs = _fd_oracle_conditionals()
    r = adj.frontdoor_integrate([x1], [m_cond], y_cond, S_obs, n_nodes=20)
    assert abs(r["mean"] - truth) < 1e-4
    # Var[Y | do(x1)] = lik + spread of the conditional mean over the mixture
    assert r["v"] == pytest.approx(r["aleatoric"]) and r["v"] >= 0.0676


def test_frontdoor_mean_substitution_is_biased_on_the_papers_scm():
    """n_nodes=1 is exactly the legacy plug-in of E[M | x]: at x1 = 0.5 the
    mediator mean sits in the bottom of the well and the estimate is 0,
    whereas the integrated value is 0.336."""
    m_cond, y_cond, S_obs = _fd_oracle_conditionals()
    legacy = adj.frontdoor_integrate([0.5], [m_cond], y_cond, S_obs, n_nodes=1)
    fixed = adj.frontdoor_integrate([0.5], [m_cond], y_cond, S_obs, n_nodes=20)
    assert abs(legacy["mean"]) < 1e-9
    assert abs(fixed["mean"] - 0.33590) < 1e-4


def test_frontdoor_two_mediators_product_grid_matches_single_mediator_limit():
    """A second mediator that Y ignores must leave the value unchanged."""
    m_cond, y_cond, S_obs = _fd_oracle_conditionals()
    m2_cond = lambda X: (0.3 * X[:, 1:2], np.full((X.shape[0], 1), 0.5))   # M2 | x, m1

    def y_cond2(X):  # X = [M1, M2, X1']
        return y_cond(X[:, [0, 2]])

    r1 = adj.frontdoor_integrate([0.5], [m_cond], y_cond, S_obs, n_nodes=12)
    r2 = adj.frontdoor_integrate([0.5], [m_cond, m2_cond], y_cond2, S_obs, n_nodes=12)
    assert abs(r1["mean"] - r2["mean"]) < 1e-9


def test_frontdoor_many_mediators_uses_seeded_draws_deterministically():
    m_cond, y_cond, S_obs = _fd_oracle_conditionals()
    extra = lambda X: (np.zeros((X.shape[0], 1)), np.ones((X.shape[0], 1)))

    def y_cond4(X):
        return y_cond(X[:, [0, 3]])

    a = adj.frontdoor_integrate([0.5], [m_cond, extra, extra], y_cond4, S_obs, mc_draws=300)
    b = adj.frontdoor_integrate([0.5], [m_cond, extra, extra], y_cond4, S_obs, mc_draws=300)
    assert a["mean"] == b["mean"]                        # common random numbers
    assert abs(a["mean"] - 0.33590) < 0.08               # Monte-Carlo accuracy


# ------------------------------------------------------------ g-computation
def _chain_steps():
    n = 200
    base = {"X": np.zeros(n), "M": np.zeros(n)}
    steps = [("M", ["X"], lambda X: (X[:, :1], np.ones((X.shape[0], 1))))]  # M | x ~ N(x, 1)
    y_cond = lambda X: (X[:, :1] ** 2, np.zeros((X.shape[0], 1)), 0.0)        # Y = M^2
    Z = np.random.default_rng(0).standard_normal((4 * n, 1))
    return base, steps, y_cond, Z


@pytest.mark.parametrize("x", [0.0, 1.5])
def test_gcomp_oracle_nonlinear(x):
    base, steps, y_cond, Z = _chain_steps()
    drawn = adj.gcomp_propagate([x], ["X"], steps, y_cond, ["M"], base, Z)
    mean_prop = adj.gcomp_propagate([x], ["X"], steps, y_cond, ["M"], base, np.zeros_like(Z))
    assert abs(drawn["mean"] - (x * x + 1.0)) < 0.15       # E[M^2] = x^2 + Var
    assert abs(mean_prop["mean"] - x * x) < 1e-12           # mean propagation is biased


def test_gcomp_is_deterministic_and_continuous():
    base, steps, y_cond, Z = _chain_steps()
    f = lambda x: adj.gcomp_propagate([x], ["X"], steps, y_cond, ["M"], base, Z)["mean"]
    assert f(0.7) == f(0.7)
    assert abs(f(0.7 + 1e-6) - f(0.7)) < 1e-4


def test_gcomp_builder_uses_chain_rule_within_clusters():
    """Members of a propagated cluster are conditioned on their parents AND
    on the earlier members of the same cluster (joint law preserved)."""
    rng = np.random.default_rng(1)
    n = 60
    X = rng.normal(size=n)
    M1 = X + 0.3 * rng.normal(size=n)
    M2 = M1 + 0.1 * rng.normal(size=n)
    Y = M1 * M2 + 0.05 * rng.normal(size=n)
    df = pd.DataFrame({"X": X, "M1": M1, "M2": M2, "Y": Y})
    cl = frozenset({"M1", "M2"})
    plan = {"propagation_order": [(cl, [frozenset({"X"})])],
            "conditioning_vars": [cl]}
    pred = adj._build_cdag_gcomputation_predictor_admg(
        ["X"], plan, "Y", df, set(df.columns), reps=2)
    assert pred.steps == [("M1", ["X"]), ("M2", ["X", "M1"])]
    m = pred.moments([0.5])
    assert np.isfinite(m["mean"]) and m["v"] >= 0.0


# ------------------------------------------------------ variance policy
def test_variance_policy_decomposition():
    mu = np.array([1.0, 3.0, 2.0])
    var_f = np.array([0.1, 0.2, 0.3])
    w = np.full(3, 1 / 3)
    r = adj._mixture_moments(mu, var_f, 0.05, w, n_obs=3)
    spread = np.mean((mu - 2.0) ** 2)
    assert r["mean"] == pytest.approx(2.0)
    assert r["aleatoric"] == pytest.approx(0.05 + spread)
    assert r["epistemic"] == pytest.approx(0.2)
    assert r["sampling"] == pytest.approx(spread / 3)
    assert adj.VARIANCE_POLICY == "predictive"              # engine default
    assert r["v"] == pytest.approx(r["aleatoric"] + r["epistemic"])


def test_variance_policy_switch(monkeypatch):
    mu, var_f, w = np.array([1.0, 3.0]), np.array([0.1, 0.3]), np.full(2, 0.5)
    monkeypatch.setattr(adj, "VARIANCE_POLICY", "total")
    r = adj._mixture_moments(mu, var_f, 0.05, w, n_obs=2)
    assert r["v"] == pytest.approx(r["aleatoric"]) and r["v"] == pytest.approx(0.05 + 1.0)
    monkeypatch.setattr(adj, "VARIANCE_POLICY", "epistemic")
    r = adj._mixture_moments(mu, var_f, 0.05, w, n_obs=2)
    assert r["v"] == pytest.approx(0.2 + 1.0 / 2)
    monkeypatch.setattr(adj, "VARIANCE_POLICY", "bogus")
    with pytest.raises(ValueError):
        adj._mixture_moments(mu, var_f, 0.05, w, n_obs=2)


def test_backdoor_predictor_variance_grows_off_support(monkeypatch):
    """The predictive policy keeps the GP's epistemic term, so the prior
    variance grows where the outcome regression is unsupported; the "total"
    policy collapses there (documented failure mode: a ~0 variance
    adjustment zeroes the CausalRBF kernel and the arm can never learn)."""
    rng = np.random.default_rng(2)
    n = 80
    Z = rng.normal(size=n)
    S = 0.5 * Z + rng.normal(size=n)
    Y = np.sin(S) + Z + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"S": S, "Z": Z, "Y": Y})
    pred = adj._build_backdoor_predictor(["S"], ["Z"], "Y", df)
    m, far = pred.moments([0.3]), pred.moments([25.0])
    assert m["v"] == pytest.approx(m["lik_var"] + m["spread"] + m["epistemic"])
    assert far["epistemic"] > 10 * m["epistemic"]
    assert far["v"] > 10 * m["v"]
    monkeypatch.setattr(adj, "VARIANCE_POLICY", "total")
    far_total = pred.moments([25.0])
    assert far_total["v"] == pytest.approx(far_total["lik_var"] + far_total["spread"])
    assert far_total["v"] < far["v"] / 10


# ------------------------------------------------------ fitted path (slow)
@pytest.mark.slow
def test_fitted_frontdoor_is_no_worse_than_mean_substitution():
    """On the paper's FrontDoor pool (100 rows) the integrated estimator must
    not be worse than the legacy plug-in; regression error dominates at this
    sample size, so the test asserts ordering, not a tight absolute error."""
    from ccbo.minimal_suite import load_scm
    _, obs, _ = load_scm(mb.FD_NAME, 100)
    p20 = adj._build_frontdoor_predictor(["X1"], ["M"], "Y", obs, n_nodes=20)
    p1 = adj._build_frontdoor_predictor(["X1"], ["M"], "Y", obs, n_nodes=1)
    xs = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
    e20 = np.array([p20([x])[0] - mb.fd_do_x1(x) for x in xs])
    e1 = np.array([p1([x])[0] - mb.fd_do_x1(x) for x in xs])
    assert np.sqrt(np.mean(e20 ** 2)) <= np.sqrt(np.mean(e1 ** 2)) + 1e-9
    assert np.sqrt(np.mean(e20 ** 2)) < 0.25
