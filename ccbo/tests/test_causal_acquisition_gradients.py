"""Predictive, EI and EI/cost gradients of the causal surrogate vs central
differences (engine v3).  Also pins the corrected maximisation EI."""

import os

import numpy as np
import pytest

from ccbo.cbo.utils.BO_functions import update_BO_models, CausalGPyModelWrapper
from ccbo.cbo.utils.causal_acquisition_functions import CausalExpectedImprovement
from ccbo.cbo.utils.cost_functions import Cost


def _v(X):
    return (0.5 + 0.3 * np.sin(X[:, 0]) + 0.1 * X[:, 1] ** 2)[:, None]


def _m(X):
    return (0.2 * X[:, 0] ** 2 - X[:, 1])[:, None]


def _fdj(f, X, h=1e-5):
    """Row-wise Jacobian of f: (N,Q) -> (N,1) by central differences."""
    g = np.zeros_like(X)
    for i in range(X.shape[0]):
        for q in range(X.shape[1]):
            Xp = X.copy(); Xp[i, q] += h
            Xm = X.copy(); Xm[i, q] -= h
            g[i, q] = (np.ravel(f(Xp))[i] - np.ravel(f(Xm))[i]) / (2 * h)
    return g


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(-2, 2, size=(6, 2))
    ytr = (np.sin(Xtr[:, 0]) + 0.2 * Xtr[:, 1] ** 2)[:, None]
    m = update_BO_models(_m, _v, Xtr, ytr, True)
    m.optimize()
    return m, Xtr, ytr


@pytest.fixture(scope="module")
def Xq():
    return np.random.default_rng(1).uniform(-2, 2, size=(3, 2))


def test_wrapper_type_and_prediction_gradients(model, Xq):
    m, _, _ = model
    assert isinstance(m, CausalGPyModelWrapper)
    dmu, dvar = m.get_prediction_gradients(Xq)
    assert np.allclose(dmu, _fdj(lambda X: m.predict(X)[0], Xq), atol=1e-6)
    assert np.allclose(dvar, _fdj(lambda X: m.predict(X)[1], Xq), atol=1e-6)


@pytest.mark.parametrize("task", ["min", "max"])
def test_expected_improvement_gradients(model, Xq, task):
    m, _, ytr = model
    best = float(ytr.min()) if task == "min" else float(ytr.max())
    acq = CausalExpectedImprovement(best, task, m)
    ei, dei = acq.evaluate_with_gradients(Xq)
    assert np.all(ei >= -1e-12)                                    # EI is non-negative in both tasks
    assert np.allclose(dei, _fdj(acq.evaluate, Xq), atol=1e-6)


def test_max_ei_equals_min_ei_of_negated_objective():
    rng = np.random.default_rng(5)
    Xtr = rng.uniform(-2, 2, size=(6, 2)); ytr = (np.sin(Xtr[:, 0]))[:, None]
    m_pos = update_BO_models(None, None, Xtr, ytr, False)
    m_neg = update_BO_models(None, None, Xtr, -ytr, False)
    Xq = rng.uniform(-2, 2, size=(4, 2))
    ei_max = CausalExpectedImprovement(float(ytr.max()), "max", m_pos).evaluate(Xq)
    ei_min = CausalExpectedImprovement(float((-ytr).min()), "min", m_neg).evaluate(Xq)
    assert np.allclose(ei_max, ei_min, atol=1e-8)


def test_ei_over_cost_gradients_with_value_dependent_cost(model, Xq):
    m, _, ytr = model
    acq = CausalExpectedImprovement(float(ytr.max()), "max", m)   # max EI is O(1) here
    costs = {"a": lambda x, **k: 1.0 + np.abs(np.ravel(x)), "b": lambda x, **k: 2.0}
    cost = Cost(costs, ["a", "b"])
    f, g = cost.evaluate_with_gradients(Xq)
    assert np.allclose(g[:, 0], np.sign(Xq[:, 0]), atol=1e-6) and np.allclose(g[:, 1], 0.0)
    q = acq / cost
    val, grad = q.evaluate_with_gradients(Xq)
    assert np.allclose(grad, _fdj(q.evaluate, Xq), atol=1e-6)


def test_constant_cost_gradient_is_zero():
    cost = Cost({"a": lambda x, **k: 1.0}, ["a"])
    f, g = cost.evaluate_with_gradients(np.array([[0.3], [-1.0]]))
    assert np.allclose(g, 0.0) and f.shape == (2, 1) and np.allclose(f, 1.0)


def test_approx_grad_escape_hatch_agrees(monkeypatch, model):
    """Analytic-gradient L-BFGS and finite-difference L-BFGS reach the same
    acquisition value (within tolerance) on a 1-D arm."""
    from emukit.core import ParameterSpace, ContinuousParameter
    from ccbo.cbo.utils.causal_optimizer import CausalGradientAcquisitionOptimizer
    rng = np.random.default_rng(7)
    Xtr = rng.uniform(-2, 2, size=(5, 1)); ytr = np.sin(2 * Xtr)
    v1 = lambda X: (0.3 + 0.2 * np.cos(X[:, 0]))[:, None]
    m1 = lambda X: (0.1 * X[:, 0] ** 2)[:, None]
    mod = update_BO_models(m1, v1, Xtr, ytr, True)
    space = ParameterSpace([ContinuousParameter("x", -2.0, 2.0)])
    acq = CausalExpectedImprovement(float(ytr.min()), "min", mod)
    x_a, f_a = CausalGradientAcquisitionOptimizer(space).optimize(acq)
    monkeypatch.setenv("CCBO_ACQ_APPROX_GRAD", "1")
    x_n, f_n = CausalGradientAcquisitionOptimizer(space).optimize(acq)
    assert abs(float(np.ravel(f_a)[0]) - float(np.ravel(f_n)[0])) < 1e-3 * max(1.0, abs(float(np.ravel(f_a)[0])))
