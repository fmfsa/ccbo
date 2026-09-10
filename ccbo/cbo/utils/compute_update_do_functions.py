"""Per-arm causal-prior mean / variance functions with exact-key caching.

Engine v3 (2026-09):
* cache keys are the exact bytes of the query point (previously NumPy's
  ``str(x)``, which quantises coordinates to the print precision);
* one do-function evaluation fills *both* the mean and the variance cache
  (the kernel's finite-difference ``dv/dx`` queries ``v`` at ``x +/- h``,
  and the mean function is queried at the same points);
* ``update_all_do_functions`` accepts a per-arm ``prior_mask``; masked arms
  get ``(None, None)`` and the graph's do-functions are not even built when
  every arm is masked.
"""

import numpy as np


def get_do_function_name(intervention_variables):
    string = ''
    for i in range(len(intervention_variables)):
        string += str(intervention_variables[i])
    total_string = 'compute_do_' + string
    return total_string


def _key(xi):
    """Exact, collision-free cache key for one query point."""
    a = np.ascontiguousarray(np.asarray(xi, dtype=np.float64))
    return (a.shape, a.tobytes())


def _shared_lookup(do_fn, observational_samples, functions, cache_mean, cache_var):
    """Return ``xi -> (mean, var)`` memoised in the two per-arm caches."""
    def lookup(xi):
        k = _key(xi)
        if k in cache_mean and k in cache_var:
            return cache_mean[k], cache_var[k]
        m, v = do_fn(observational_samples, functions, xi)
        m = float(np.ravel(np.asarray(m, dtype=float))[0])
        v = float(np.ravel(np.asarray(v, dtype=float))[0])
        cache_mean[k] = m
        cache_var[k] = v
        return m, v
    return lookup


def _mean_var_functions(lookup):
    def mean_function_do(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.array([[lookup(x[i])[0]] for i in range(x.shape[0])], dtype=np.float64)

    def var_function_do(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.array([[lookup(x[i])[1]] for i in range(x.shape[0])], dtype=np.float64)

    return mean_function_do, var_function_do


## Given a do function, compute the mean and variance functions of the causal prior
def mean_var_do_functions(do_effects_function, observational_samples, functions):
    lookup = _shared_lookup(do_effects_function, observational_samples, functions, {}, {})
    return _mean_var_functions(lookup)


def update_all_do_functions(graph, exploration_set, functions, dict_interventions,
                            observational_samples, x_dict_mean, x_dict_var,
                            prior_mask=None):
    """Build the (mean, var) prior functions for every arm.

    ``x_dict_mean[label]`` / ``x_dict_var[label]`` are the per-arm caches
    created by ``initialise_dicts`` (cleared by the CBO loop on every
    observation).  Arms with ``prior_mask[j] is False`` receive ``(None,
    None)``; if every arm is masked, ``graph.get_all_do()`` is never called.
    """
    if prior_mask is None:
        prior_mask = [True] * len(exploration_set)
    assert len(prior_mask) == len(exploration_set)

    mean_functions_list = []
    var_functions_list = []
    do_functions = None
    for j in range(len(exploration_set)):
        if not prior_mask[j]:
            mean_functions_list.append(None)
            var_functions_list.append(None)
            continue
        if do_functions is None:
            do_functions = graph.get_all_do()
        label = dict_interventions[j]
        lookup = _shared_lookup(do_functions[get_do_function_name(label)],
                                observational_samples, functions,
                                x_dict_mean[label], x_dict_var[label])
        m, v = _mean_var_functions(lookup)
        mean_functions_list.append(m)
        var_functions_list.append(v)
    return mean_functions_list, var_functions_list


# Backwards-compatible single-arm builders (share one lookup per call).
def update_mean_fun(graph, functions, variables, observational_samples, xi_dict_mean):
    do_functions = graph.get_all_do()
    lookup = _shared_lookup(do_functions[get_do_function_name(variables)],
                            observational_samples, functions,
                            xi_dict_mean[variables], {})
    return _mean_var_functions(lookup)[0]


def update_var_fun(graph, functions, variables, observational_samples, xi_dict_var):
    do_functions = graph.get_all_do()
    lookup = _shared_lookup(do_functions[get_do_function_name(variables)],
                            observational_samples, functions,
                            {}, xi_dict_var[variables])
    return _mean_var_functions(lookup)[1]
