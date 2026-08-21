"""
Generate interventional data for arbitrary exploration sets.

The CBO codebase expects interventional data in a specific format
(as stored in interventional_data.npy). This module generates that
data for any exploration set by sampling from the true SEM.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

from .generic_do import sample_from_model, intervene


def generate_interventional_data(sem_fn, exploration_set, dict_ranges,
                                 num_points=20, num_mc_samples=100000, seed=0,
                                 target_evaluator=None):
    """
    Generate interventional data for each element of the exploration set.

    Parameters
    ----------
    sem_fn : callable
        Function returning the SEM OrderedDict.
    exploration_set : list of list of str
        Each element is a list of variable names to intervene on.
    dict_ranges : OrderedDict
        Interventional ranges for each variable.
    num_points : int
        Number of interventional data points per exploration set element.
    num_mc_samples : int
        Number of MC samples per evaluation.
    seed : int
        Random seed.
    target_evaluator : callable, optional
        Exact population evaluator with signature ``(arm, values) -> float``.
        When provided, no SEM sampling is used for target evaluations.

    Returns
    -------
    list
        Interventional data in CBO format. Each element corresponds to
        one exploration set entry: [num_vars, var1, ..., vark, data_x, data_y]
    """
    interventional_data = []

    for es_entry in exploration_set:
        num_vars = len(es_entry)

        # Sample intervention values uniformly from ranges
        rng = np.random.RandomState(seed)
        data_x = np.zeros((num_points, num_vars))
        for j, var in enumerate(es_entry):
            lo, hi = dict_ranges[var]
            data_x[:, j] = rng.uniform(lo, hi, num_points)

        # Evaluate the true target function at each point
        data_y = np.zeros((num_points, 1))
        model = None if target_evaluator is not None else sem_fn()
        for i in range(num_points):
            if target_evaluator is not None:
                data_y[i, 0] = target_evaluator(es_entry, data_x[i])
            else:
                intervention_dict = {var: data_x[i, j]
                                     for j, var in enumerate(es_entry)}
                mutilated = intervene(intervention_dict, model)
                np.random.seed(seed + i)
                samples = [sample_from_model(mutilated)
                           for _ in range(num_mc_samples)]
                data_y[i, 0] = np.mean([s['Y'] for s in samples])

        # Pack in the format expected by define_initial_data_CBO:
        # [num_vars, var_name_1, ..., var_name_k, data_x_array, data_y_array]
        entry = [num_vars] + list(es_entry) + [data_x, data_y]
        interventional_data.append(entry)

    return interventional_data
