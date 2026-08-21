
## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns



def update_hull(observational_samples, manipulative_variables):
    ## This function computes the coverage of the observations
    list_variables = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])

    stack_variables = np.transpose(np.vstack((list_variables)))
    try:
        coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume
    except Exception:
        coverage_obs = 0.0

    return coverage_obs


def observe(complete_dataset, start, batch_size, stop):
    """Reveal the next unseen batch of observational rows.

    Positional slicing (``.iloc``) so the result never depends on the
    DataFrame's index labels.  The batch is truncated at ``stop`` (the
    effective observational cap), so the final batch may be smaller than
    ``batch_size`` and an exhausted pool yields an empty frame.

    Parameters
    ----------
    complete_dataset : pandas.DataFrame
        The full observational pool.
    start : int
        Cursor: position of the first not-yet-revealed row.
    batch_size : int
        Requested number of rows.
    stop : int
        Effective cap ``min(max_N, len(complete_dataset))``.

    Returns
    -------
    (rows, new_cursor) : (pandas.DataFrame, int)
    """
    start = int(start)
    stop = int(stop)
    end = min(start + int(batch_size), stop)
    if end <= start:
        return complete_dataset.iloc[start:start].copy(), start
    return complete_dataset.iloc[start:end].copy(), end


def observation_probability(observational_samples, manipulative_variables,
                            coverage_total, obs_cap):
    """Probability of taking an *observe* action at the current trial.

    This is the schedule of the released CBO implementation
    (Aglietti et al. 2020),

        eps_t = [ Vol(C(D_t^O)) / Vol(D(X)) ] / [ N_t / N_cap ],

    i.e. the coverage ratio *divided* by the sample ratio.  Note this
    differs from the multiplication-based expression printed in the paper;
    we follow the reference code, and only clip the result to [0, 1] so it
    is a valid probability.  Clipping is behaviour-neutral: the caller
    draws u ~ U[0, 1), so any eps >= 1 selects observation either way.

    ``obs_cap`` is the *effective* cap ``min(max_N, len(pool))`` rather
    than the requested ``max_N``: the available pool may be shorter than
    the declared maximum.

    Returns
    -------
    (raw, clipped) : (float, float)
        The unclipped value (for logging) and the probability to use.
    """
    n_obs = len(observational_samples)
    if n_obs <= 0 or obs_cap <= 0 or coverage_total <= 0:
        return 0.0, 0.0

    coverage_obs = update_hull(observational_samples, manipulative_variables)
    coverage_ratio = coverage_obs / coverage_total
    rescale = n_obs / obs_cap

    raw = float(coverage_ratio / rescale)
    return raw, float(np.clip(raw, 0.0, 1.0))



def compute_coverage(observational_samples, manipulative_variables, dict_ranges):
    list_variables = []
    list_ranges = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])
      list_ranges.append(dict_ranges[manipulative_variables[i]])

    vertices = list(itertools.product(*[list_ranges[i] for i in range(len(manipulative_variables))]))
    try:
        coverage_total = scipy.spatial.ConvexHull(vertices).volume
    except Exception:
        # Fallback: box volume (product of range lengths)
        coverage_total = float(np.prod([abs(r[1] - r[0]) for r in list_ranges]))

    stack_variables = np.transpose(np.vstack((list_variables)))
    try:
        hull_obs = scipy.spatial.ConvexHull(stack_variables)
        coverage_obs = hull_obs.volume
    except Exception:
        # Degenerate point cloud (e.g. nearly collinear in high-dim space).
        # Return zero coverage and no hull; callers use coverage_total only.
        hull_obs = None
        coverage_obs = 0.0

    alpha_coverage = coverage_obs / coverage_total if coverage_total > 0 else 0.0
    return alpha_coverage, hull_obs, coverage_total


def define_initial_data_CBO(interventional_data, num_interventions, exploration_set, name_index, task):

    data_list = []
    data_x_list = []
    data_y_list = []
    opt_list = []


    for j in range(len(exploration_set)):
      data = interventional_data[j].copy()
      num_variables = data[0]
      if num_variables == 1:
        data_x = np.asarray(data[(num_variables+1)])
        data_y = np.asarray(data[-1])
      else:
        data_x = np.asarray(data[(num_variables+1):(num_variables*2)][0])
        data_y = np.asarray(data[-1])


      if len(data_y.shape) == 1:
          data_y = data_y[:,np.newaxis]

      if len(data_x.shape) == 1:
          data_x = data_x[:,np.newaxis]
      


      all_data = np.concatenate((data_x, data_y), axis =1)

      ## Need to reset the global seed 
      state = np.random.get_state()

      np.random.seed(name_index)
      np.random.shuffle(all_data)


      np.random.set_state(state)

      subset_all_data = all_data[:num_interventions]

      data_list.append(subset_all_data)
      data_x_list.append(data_list[j][:,:-1])
      data_y_list.append(data_list[j][:,-1][:,np.newaxis])


      if task == 'min':
        opt_list.append(np.min(subset_all_data[:,-1])) 
        var_min = exploration_set[np.where(opt_list == np.min(opt_list))[0][0]]
        opt_y = np.min(opt_list)
        opt_intervention_array = data_list[np.where(opt_list == np.min(opt_list))[0][0]]
      else:
        opt_list.append(np.max(subset_all_data[:,-1])) 
        var_min = exploration_set[np.where(opt_list == np.max(opt_list))[0][0]]
        opt_y = np.max(opt_list)
        opt_intervention_array = data_list[np.where(opt_list == np.max(opt_list))[0][0]]


    # Concatenate variable names regardless of how many (handles 1..N variables).
    best_variable = ''.join(var_min)


    shape_opt = opt_intervention_array.shape[1] - 1
    if task == 'min':
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.min(opt_intervention_array[:,shape_opt]), :shape_opt][0]
    else:
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.max(opt_intervention_array[:,shape_opt]), :shape_opt][0]
   

    return data_x_list, data_y_list, best_intervention_value, opt_y, best_variable

