"""Actual-backend gates for union-parent temporal models (no expensive fits)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ccbo.qdcbo.quotient_dbn import ensure_dcbo_on_path, build_quotient_dbn

try:
    ensure_dcbo_on_path()
except FileNotFoundError:
    pytest.skip('Pinned DCBO checkout required', allow_module_level=True)

from ccbo.qdcbo.coherent import (fit_conditional_mechanisms, integrate_predictive,
                                make_prior_functions, predictive_moments)

PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'validate_dynamic_sem.py'
spec = importlib.util.spec_from_file_location('dynamic_diagnostic_fixture', PATH)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def graph():
    import networkx as nx
    g = nx.MultiDiGraph()
    for t in range(2):
        g.add_nodes_from(f'{v}_{t}' for v in ('X', 'Z', 'Y'))
        g.add_edge(f'X_{t}', f'Y_{t}')
    g.add_edge('Y_0', 'Y_1')
    g.T = 2
    return g


@pytest.mark.parametrize('partition', [[['X'], ['Z']], [['X', 'Z']]])
@pytest.mark.parametrize('intercept,root_mean', [(3., 1.), (0., 0.)])
def test_union_parent_conditional_removes_double_counting(partition, intercept, root_mean):
    q = build_quotient_dbn(graph(), partition)
    fitted = fit_conditional_mechanisms(q, fixture.moment_data(
        fixture.analytic_case(intercept, root_mean)), fixture.LinearProjection)
    assert fitted[1]['Y']['inputs'] == ('Y_0',) + q.emit_input_nodes('Y', 1)
    for x in (-2., 0., 2.):
        blanket = {'X': [0., x], 'Z': [0., 0.], 'Y': [7., None]}
        mean, variance = integrate_predictive(q, fitted, 1, blanket, samples=16, seed=0)
        assert mean == pytest.approx(intercept + 2*x + 3.5, abs=1e-10)
        assert variance == pytest.approx(.25, abs=1e-10)


def test_predictive_variance_is_integrated_through_nonlinear_child():
    import networkx as nx
    g = nx.MultiDiGraph([('X_0', 'Y_0')]); g.T = 1
    q = build_quotient_dbn(g, [['X']])

    class Square:
        def predict(self, x):
            return x ** 2, np.zeros_like(x)

    fitted = {0: {'X': dict(inputs=(), members=('X',), empirical=np.array([[-1.], [1.]])),
                  'Y': dict(inputs=('X_0',), members=('Y',), model=Square())}}
    result = integrate_predictive(q, fitted, 0, {'X': [None], 'Y': [None]}, samples=32)
    assert result == pytest.approx((1., 0.))  # f(E[X]) would incorrectly give 0


def test_correlated_temporal_parents_require_joint_regression_not_intercept_repair():
    g = graph()
    g.add_edge('Y_0', 'X_1')
    data = fixture.moment_data(fixture.analytic_case())
    original_x1 = data['X'][:, 1].copy()
    original_y1 = data['Y'][:, 1].copy()
    data['X'][:, 1] += .7 * data['Y'][:, 0]
    data['Y'][:, 1] = original_y1 + 2 * (data['X'][:, 1] - original_x1)
    for partition in ([['X'], ['Z']], [['X', 'Z']]):
        q = build_quotient_dbn(g, partition)
        fitted = fit_conditional_mechanisms(q, data, fixture.LinearProjection)
        mean, variance = integrate_predictive(q, fitted, 1,
            {'X': [0., 0.], 'Z': [0., 0.], 'Y': [7., None]}, samples=16)
        assert mean == pytest.approx(6.5, abs=1e-10)
        assert variance == pytest.approx(.25, abs=1e-10)


def test_fixed_common_blanket_and_rng_are_preserved():
    q = build_quotient_dbn(graph(), [['X'], ['Z']])
    fitted = fit_conditional_mechanisms(q, fixture.moment_data(
        fixture.analytic_case()), fixture.LinearProjection)
    blanket = {'X': [0., None], 'Z': [0., 0.], 'Y': [7., None]}
    before = {k: list(v) for k, v in blanket.items()}
    mean, variance = make_prior_functions(q, fitted, 1, ('X',), blanket, samples=16, seed=0)
    np.random.seed(23); expected = np.random.random(5)
    np.random.seed(23)
    first = mean(np.array([[0.], [1.]]))
    assert np.array_equal(first, mean(np.array([[0.], [1.]])))
    assert variance(np.array([[0.]]))[0, 0] == pytest.approx(.25)
    assert np.array_equal(np.random.random(5), expected)
    assert blanket == before


def test_partial_cluster_clamp_is_not_misrepresented_as_joint_identification():
    q = build_quotient_dbn(graph(), [['X', 'Z']])
    fitted = fit_conditional_mechanisms(q, fixture.moment_data(
        fixture.analytic_case()), fixture.LinearProjection)
    blanket = {'X': [0., 0.], 'Z': [0., None], 'Y': [7., None]}
    with pytest.raises(ValueError, match='Partial clamp'):
        integrate_predictive(q, fitted, 1, blanket, samples=16)


def test_joint_covariance_is_used_in_multivariate_predictive_draws():
    from ccbo.qdcbo.coherent import _gaussian_draw
    mean = np.zeros((3, 2))
    cov = np.tile(np.array([[1., 1.], [1., 1.]]), (3, 1, 1))
    z = np.array([[.2, -.3], [.4, .8], [-1., .1]])
    draw = _gaussian_draw(mean, cov, z)
    assert np.allclose(draw[:, 0], draw[:, 1])


def test_noiseless_conditional_variance_and_zero_clamp():
    q = build_quotient_dbn(graph(), [['X'], ['Z']])
    fitted = fit_conditional_mechanisms(q, fixture.moment_data(
        fixture.analytic_case()), fixture.LinearProjection)
    fitted[1]['Y']['model'].variance[:] = 0.
    mean, variance = integrate_predictive(q, fitted, 1,
        {'X': [0., 0.], 'Z': [0., 0.], 'Y': [0., None]}, samples=16, seed=0)
    assert mean == pytest.approx(3.)
    assert variance == pytest.approx(0., abs=1e-20)


def test_residual_joint_law_orders_tasks_and_does_not_double_count_likelihood():
    from types import SimpleNamespace
    from ccbo.qdcbo.coherent import ResidualJointConditional
    from ccbo.qdcbo.qdcbo_base import JointClusterGP

    class FakeGP:
        X = np.array([[0., 0.], [1., 0.], [2., 0.],
                      [0., 1.], [1., 1.], [2., 1.]])
        posterior = SimpleNamespace(woodbury_vector=np.array([-1., 0., 1., -2., 0., 2.])[:, None],
                                    woodbury_inv=np.eye(6))

        def predict(self, x, full_cov, include_likelihood, Y_metadata):
            assert full_cov and not include_likelihood
            assert np.array_equal(Y_metadata['output_index'].ravel(), x[:, -1])
            return (x[:, :1] + 10*x[:, -1:]), .2*np.eye(len(x))

    base = object.__new__(JointClusterGP); base.model = FakeGP()
    fitted = ResidualJointConditional(base, 2)
    mean, cov = fitted.predict_joint(np.array([[3.], [4.]]))
    assert np.array_equal(mean, [[3., 13.], [4., 14.]])
    assert cov[0, 0, 1] == pytest.approx(1.8)
    assert cov[0, 0, 0] == pytest.approx(1.2, abs=1e-6)
    assert cov[0, 1, 1] == pytest.approx(4.2, abs=1e-6)


def test_residual_leaveout_excludes_sibling_tasks_from_held_out_row():
    from types import SimpleNamespace
    from ccbo.qdcbo.coherent import ResidualJointConditional
    from ccbo.qdcbo.qdcbo_base import JointClusterGP
    x = np.array([[0., 0.], [1., 0.], [2., 0.],
                  [0., 1.], [1., 1.], [2., 1.]])
    precision = np.zeros((6, 6))
    block = np.linalg.inv(np.array([[1., .9], [.9, 1.]]))
    for row in range(3):
        indices = [row, row+3]
        precision[np.ix_(indices, indices)] = block
    y = np.array([-1., 0., 1., -1., 0., 1.])
    fake = SimpleNamespace(X=x, posterior=SimpleNamespace(
        woodbury_inv=precision, woodbury_vector=(precision @ y)[:, None]))
    base = object.__new__(JointClusterGP); base.model = fake
    fitted = ResidualJointConditional(base, 2)
    assert fitted.residual_covariance[0, 0] == pytest.approx(1., abs=1e-6)
    assert fitted.residual_covariance[0, 1] == pytest.approx(.9, abs=1e-6)


def test_actual_regularized_setup_dispatch_builds_horizon_and_true_crosslags():
    from ccbo.qdcbo.runner import get_setup
    from ccbo.qdcbo.population import RegularizedNonStationarySEM
    sem, graph, arms, domains, target, change_points = get_setup('nonstat_regularized', 3)
    assert sem is RegularizedNonStationarySEM
    assert graph.T == 3
    assert graph.has_edge('X_0', 'Z_1') and graph.has_edge('Z_1', 'Y_2')
    assert change_points == [False, True, False]
