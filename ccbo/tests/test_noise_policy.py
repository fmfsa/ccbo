"""The arm surrogates' likelihood variance is fixed at 1e-10 (engine v3)."""

import numpy as np
import pytest

from ccbo.cbo.utils.BO_functions import update_BO_models, NOISE_VAR


@pytest.mark.parametrize("causal", [False, True])
def test_likelihood_variance_is_fixed(causal):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(5, 1)); y = np.sin(3 * X)
    m = update_BO_models((lambda Z: 0.0 * Z[:, :1]) if causal else None,
                         (lambda Z: 0.2 + 0.0 * Z[:, :1]) if causal else None,
                         X, y, causal)
    lik = m.model.likelihood.variance
    assert lik.is_fixed and float(lik) == NOISE_VAR == 1e-10
    m.optimize()
    assert float(m.model.likelihood.variance) == 1e-10


def test_causal_branch_requires_prior_functions():
    X = np.zeros((2, 1)); y = np.zeros((2, 1))
    with pytest.raises(ValueError):
        update_BO_models(None, None, X, y, True)
