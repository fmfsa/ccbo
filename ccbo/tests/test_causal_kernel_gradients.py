"""CausalRBF gradients vs central differences (engine v3).

The inherited ``Stationary`` methods ignore the rank-one term
``sqrt(v(x)) sqrt(v(x'))``; these tests pin the overrides.
"""

import numpy as np
import pytest

from ccbo.cbo.utils.causal_kernels import CausalRBF


def _v(X):
    return (0.5 + 0.3 * np.sin(X[:, 0]) + 0.1 * X[:, 1] ** 2)[:, None]


def _fd(fun, X, h=1e-6):
    g = np.zeros_like(X)
    for i in range(X.shape[0]):
        for q in range(X.shape[1]):
            Xp = X.copy(); Xp[i, q] += h
            Xm = X.copy(); Xm[i, q] -= h
            g[i, q] = (fun(Xp) - fun(Xm)) / (2 * h)
    return g


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    k = CausalRBF(2, variance_adjustment=_v, lengthscale=0.7, variance=1.3)
    X = rng.normal(size=(4, 2)); X2 = rng.normal(size=(6, 2))
    W = rng.normal(size=(4, 6)); Wsq = rng.normal(size=(4, 4))
    return k, X, X2, W, Wsq


def test_kdiag_matches_diagonal_of_K(setup):
    k, X, *_ = setup
    assert np.allclose(k.Kdiag(X), np.diag(k.K(X)))
    assert k.Kdiag(X).shape == (4,)
    one = X[:1]
    assert k.Kdiag(one).shape == (1,) and np.allclose(k.Kdiag(one), k.K(one)[0, 0])


def test_gradients_X_with_X2(setup):
    k, X, X2, W, _ = setup
    ana = k.gradients_X(W, X, X2)
    num = _fd(lambda Xa: np.sum(W * k.K(Xa, X2)), X)
    assert np.allclose(ana, num, rtol=1e-5, atol=1e-7)


def test_gradients_X_without_X2(setup):
    k, X, _, _, Wsq = setup
    ana = k.gradients_X(Wsq, X)
    num = _fd(lambda Xa: np.sum(Wsq * k.K(Xa)), X)
    assert np.allclose(ana, num, rtol=1e-5, atol=1e-7)


def test_gradients_X_diag_is_dv_dx(setup):
    k, X, *_ = setup
    w = np.array([0.3, -1.2, 2.0, 0.7])
    ana = k.gradients_X_diag(w, X)
    num = _fd(lambda Xa: np.sum(w * k.Kdiag(Xa)), X)
    assert np.allclose(ana, num, rtol=1e-5, atol=1e-7)
    assert np.any(np.abs(ana) > 1e-3)          # the inherited version returned zeros


def test_hyperparameter_gradients_exclude_rank_one_term(setup):
    k, X, _, _, Wsq = setup
    k.update_gradients_full(Wsq, X)

    def L(var, ls):
        kk = CausalRBF(2, variance_adjustment=_v, lengthscale=ls, variance=var)
        return np.sum(Wsq * kk.K(X))

    h = 1e-6
    assert float(k.variance.gradient) == pytest.approx((L(1.3 + h, 0.7) - L(1.3 - h, 0.7)) / (2 * h), rel=1e-5)
    assert float(k.lengthscale.gradient) == pytest.approx((L(1.3, 0.7 + h) - L(1.3, 0.7 - h)) / (2 * h), rel=1e-5)


def test_constant_adjustment_reduces_to_rbf_plus_constant():
    rng = np.random.default_rng(3)
    k = CausalRBF(1, variance_adjustment=lambda X: np.full((X.shape[0], 1), 0.25))
    X = rng.normal(size=(5, 1))
    assert np.allclose(k.gradients_X_diag(np.ones(5), X), 0.0)
    assert np.allclose(k.K(X) - k.K_of_r(k._scaled_dist(X, X)), 0.25)


def test_second_order_gradients_raise(setup):
    k, X, *_ = setup
    with pytest.raises(NotImplementedError):
        k.gradients_XX(np.eye(4), X)
    with pytest.raises(NotImplementedError):
        k.gradients_XX_diag(np.ones(4), X)
