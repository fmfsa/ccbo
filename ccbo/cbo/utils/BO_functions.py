"""Arm-surrogate construction for the CBO loop (engine v3, 2026-09).

* ``update_BO_models`` builds one GPy regression model per arm: a plain RBF
  GP for ``Causal_prior=False`` and the causal GP (``CausalRBF`` kernel +
  do-calculus mean function) for ``Causal_prior=True``.
* The interventional likelihood variance is **fixed** at ``NOISE_VAR`` in
  both branches: interventional targets in the paper's suites are exact
  population expectations, so a free noise term would only absorb model
  misfit.  (Previously it was initialised at 1e-10 and then re-fitted.)
* ``CausalGPyModelWrapper`` completes the predictive gradients: GPy's
  ``predictive_gradients`` never differentiates the mean function, so the
  causal mean's input gradient is added by central differences.
"""

import numpy as np

import GPy
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

from .causal_kernels import CausalRBF

NOISE_VAR = 1e-10      # fixed interventional likelihood variance
FD_STEP = 1e-4         # relative central-difference step for black-box means


def define_initial_data_BO(interventional_data, num_interventions, intervention_sets, name_index, ):
    data_x = (interventional_data[0][len(intervention_sets)]).copy()
    data_y = (np.asarray(interventional_data[0][len(intervention_sets)+1])).copy()
    all_data = np.concatenate((data_x, data_y), axis =1)

    ## Need to reset the global seed
    state = np.random.get_state()

    np.random.seed(name_index)
    np.random.shuffle(all_data)

    np.random.set_state(state)

    data_x = all_data[:num_interventions, :len(intervention_sets)]
    data_y = all_data[:num_interventions, len(intervention_sets):]

    min_y = np.min(data_y)
    min_intervention_value = np.transpose(all_data[np.where(data_y == min_y)[0][0]][:len(intervention_sets)][:,np.newaxis])

    return data_x, data_y, min_intervention_value, min_y


def fd_gradient(f, X, step=FD_STEP):
    """Central-difference Jacobian of a scalar-output map ``f: (N,Q)->(N,1)``
    with respect to each input coordinate; returns ``(N, Q)``."""
    X = np.asarray(X, dtype=float)
    N, Q = X.shape
    grad = np.zeros((N, Q))
    for q in range(Q):
        h = step * np.maximum(1.0, np.abs(X[:, q]))
        Xp = X.copy(); Xp[:, q] += h
        Xm = X.copy(); Xm[:, q] -= h
        fp = np.ravel(np.asarray(f(Xp), dtype=float))
        fm = np.ravel(np.asarray(f(Xm), dtype=float))
        grad[:, q] = (fp - fm) / (2.0 * h)
    return grad


def fix_noise(gpy_model, value=NOISE_VAR):
    """Fix the Gaussian likelihood variance of a GPy model."""
    gpy_model.likelihood.variance.fix(value, warning=False)
    return gpy_model


class CausalGPyModelWrapper(GPyModelWrapper):
    """emukit wrapper whose predictive mean gradient includes the GPy mean
    function (GPy's ``predictive_gradients`` omits it)."""

    def __init__(self, gpy_model, n_restarts=1, fd_step=FD_STEP):
        super().__init__(gpy_model, n_restarts)
        self.fd_step = float(fd_step)

    def get_prediction_gradients(self, X):
        d_mean_dx, d_variance_dx = self.model.predictive_gradients(X)
        d_mean_dx = d_mean_dx[:, :, 0]
        mf = getattr(self.model, 'mean_function', None)
        if mf is not None:
            d_mean_dx = d_mean_dx + fd_gradient(mf.f, X, self.fd_step)
        return d_mean_dx, d_variance_dx


class PriorOnlyModel(IModel, IDifferentiable):
    """Arm surrogate before the arm's first measurement (no initial design).

    Predicts the GP prior the engine would use: the do-calculus mean m(x) and
    variance 1 + v(x) for a causal arm (``CausalRBF`` with unit signal
    variance), or mean 0 and variance 1 for a plain arm, plus the fixed
    likelihood variance. Gradients use central differences, as in
    ``CausalGPyModelWrapper``. On ``set_data`` it builds the regular surrogate
    with ``update_BO_models`` and delegates to it from then on.
    """

    def __init__(self, mean_function, var_function, input_dim, causal):
        self.mean_function, self.var_function = mean_function, var_function
        self.input_dim, self.causal = int(input_dim), bool(causal)
        self._inner = None

    @property
    def model(self):
        return None if self._inner is None else self._inner.model

    @property
    def X(self):
        return np.zeros((0, self.input_dim)) if self._inner is None else self._inner.X

    @property
    def Y(self):
        return np.zeros((0, 1)) if self._inner is None else self._inner.Y

    def _mean(self, X):
        if not self.causal:
            return np.zeros((X.shape[0], 1))
        return np.asarray(self.mean_function(X), dtype=float).reshape(X.shape[0], 1)

    def _var(self, X):
        if not self.causal:
            return np.ones((X.shape[0], 1)) + NOISE_VAR
        v = np.asarray(self.var_function(X), dtype=float)
        v = v.reshape(X.shape[0], 1) if v.size == X.shape[0] else np.full((X.shape[0], 1), float(np.ravel(v)[0]))
        return 1.0 + np.maximum(v, 0.0) + NOISE_VAR

    def predict(self, X):
        if self._inner is not None:
            return self._inner.predict(X)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._mean(X), self._var(X)

    def get_prediction_gradients(self, X):
        if self._inner is not None:
            return self._inner.get_prediction_gradients(X)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not self.causal:
            zero = np.zeros(X.shape)
            return zero, zero.copy()
        return fd_gradient(self._mean, X), fd_gradient(self._var, X)

    def set_data(self, X, Y):
        if self._inner is None:
            self._inner = update_BO_models(self.mean_function, self.var_function, X, Y, self.causal)
        else:
            self._inner.set_data(X, Y)

    def optimize(self, *args, **kwargs):
        if self._inner is not None:
            return self._inner.optimize(*args, **kwargs)


def update_BO_models(mean_function, var_function, data_x, data_y, Causal_prior):
    """Build the arm surrogate.  ``Causal_prior`` is a bool for *this arm*.

    An arm without data gets a :class:`PriorOnlyModel` (GPy cannot condition
    on an empty dataset)."""
    if np.shape(data_x)[0] == 0:
        if Causal_prior and (mean_function is None or var_function is None):
            raise ValueError("update_BO_models: Causal_prior=True requires the "
                             "arm's mean and variance functions")
        return PriorOnlyModel(mean_function, var_function, np.shape(data_x)[1], Causal_prior)
    if not Causal_prior:
        gpy_model = GPy.models.GPRegression(
            data_x, data_y,
            GPy.kern.RBF(data_x.shape[1], lengthscale=1., variance=1.),
            noise_var=NOISE_VAR)
        fix_noise(gpy_model)
        return GPyModelWrapper(gpy_model)

    if mean_function is None or var_function is None:
        raise ValueError("update_BO_models: Causal_prior=True requires the "
                         "arm's mean and variance functions")
    mf = GPy.core.Mapping(data_x.shape[1], 1)
    mf.f = lambda x: mean_function(x)
    mf.update_gradients = lambda a, b: None
    causal_kernel = CausalRBF(data_x.shape[1], variance_adjustment=var_function,
                              lengthscale=1., variance=1., ARD=False)
    gpy_model = GPy.models.GPRegression(data_x, data_y, causal_kernel,
                                        noise_var=NOISE_VAR, mean_function=mf)
    fix_noise(gpy_model)
    return CausalGPyModelWrapper(gpy_model)
