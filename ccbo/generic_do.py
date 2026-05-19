"""
Generic Monte Carlo do-calculus for arbitrary intervention sets.

Instead of hand-coding do-calculus formulas for each graph/intervention
combination, this module uses the SEM directly:
1. Mutilate the SEM (replace structural equations of intervened variables)
2. Forward-sample through the mutilated model
3. Return mean and variance of the target variable Y
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from numpy.random import randn


def sample_from_model(model, epsilon=None):
    """Produce a single sample from a structural equation model."""
    if epsilon is None:
        epsilon = randn(len(model))
    sample = {}
    for variable, function in model.items():
        sample[variable] = function(epsilon, **sample)
    return sample


def intervene(interventions, model):
    """Mutilate a SEM by fixing intervention variables to constants."""
    new_model = model.copy()
    for variable, value in interventions.items():
        new_model[variable] = lambda epsilon, _v=value, **kwargs: _v
    return new_model


def compute_do_generic(sem_fn, intervention_vars, observational_samples,
                       functions, value, num_mc_samples=10000, seed=1):
    """
    Compute E[Y|do(intervention_vars = value)] via Monte Carlo through the SEM.

    Parameters
    ----------
    sem_fn : callable
        Function that returns the SEM as an OrderedDict (graph.define_SEM()).
    intervention_vars : list of str
        Variables being intervened on.
    observational_samples : pd.DataFrame
        Observational data (not used in MC, but kept for API compatibility).
    functions : dict
        Fitted GP models (not used in MC, but kept for API compatibility).
    value : array-like
        Intervention values. Scalar if 1 variable, array if multiple.
    num_mc_samples : int
        Number of Monte Carlo samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mean_do : float
        E[Y | do(intervention_vars = value)]
    var_do : float
        Var[Y | do(intervention_vars = value)]
    """
    model = sem_fn()

    # Build intervention dict
    intervention_dict = {}
    if np.isscalar(value) or (hasattr(value, 'shape') and value.shape == ()):
        assert len(intervention_vars) == 1
        intervention_dict[intervention_vars[0]] = float(value)
    else:
        value = np.atleast_1d(value)
        for i, var in enumerate(intervention_vars):
            intervention_dict[var] = float(value[i])

    # Mutilate the model
    mutilated = intervene(intervention_dict, model)

    # Monte Carlo sampling
    np.random.seed(seed)
    samples = [sample_from_model(mutilated) for _ in range(num_mc_samples)]
    y_values = np.array([s['Y'] for s in samples])

    mean_do = float(np.mean(y_values))
    var_do = float(np.var(y_values))

    return mean_do, var_do


def make_do_function(sem_fn, intervention_vars, num_mc_samples=10000):
    """
    Create a do-calculus function with the signature expected by the CBO codebase.

    The returned function has signature:
        compute_do(observational_samples, functions, value) -> (mean_do, var_do)

    This matches the interface in compute_update_do_functions.py.

    Parameters
    ----------
    sem_fn : callable
        Function returning the SEM OrderedDict.
    intervention_vars : list of str
        Variables being intervened on.
    num_mc_samples : int
        Number of MC samples for each evaluation.

    Returns
    -------
    callable
        Do-calculus function compatible with CBO.
    """
    def compute_do(observational_samples, functions, value):
        return compute_do_generic(
            sem_fn, intervention_vars,
            observational_samples, functions, value,
            num_mc_samples=num_mc_samples
        )

    return compute_do
