"""
LightTunnel do-functions.

The true DAG has every manipulable variable as a direct parent of Y and no
other observable parents, so adjusting for any intervention set
S ⊆ {R, G, B, P1, P2} reduces to: predict Y from the full-input GP with S
fixed at the intervention value and the rest marginalised over the
observational sample.

This file is used by the *original-graph* CBO path. The CoarsenedGraph
path overrides ``get_all_do`` with C-DAG-derived do-functions and does
not call these (see ``ccbo/coarsened_graph.py:get_all_do``).
"""

import numpy as np


def _predict(gp, X):
    mean, var = gp.predict(X)
    return float(np.mean(mean)), float(np.mean(var))


def _build_full_inputs(obs, intervened):
    """Build (n_obs, 5) array with R, G, B, P1, P2 columns in that order;
    intervened variables overwritten with their scalar values, rest taken
    from the observational sample."""
    n = len(obs)
    cols = []
    for var in ('R', 'G', 'B', 'P1', 'P2'):
        if var in intervened:
            cols.append(np.full((n, 1), float(intervened[var])))
        else:
            cols.append(np.asarray(obs[var])[:, np.newaxis])
    return np.hstack(cols)


def _do(observational_samples, functions, intervened):
    gp = functions['gp_Y_R_G_B_P1_P2']
    X = _build_full_inputs(observational_samples, intervened)
    return _predict(gp, X)


# 8 wrappers exactly matching the MIS in LightTunnel.get_sets().

def compute_do_R(obs, functions, value):
    return _do(obs, functions, {'R': value})

def compute_do_G(obs, functions, value):
    return _do(obs, functions, {'G': value})

def compute_do_B(obs, functions, value):
    return _do(obs, functions, {'B': value})

def compute_do_P1(obs, functions, value):
    return _do(obs, functions, {'P1': value})

def compute_do_P2(obs, functions, value):
    return _do(obs, functions, {'P2': value})

def compute_do_P1P2(obs, functions, value):
    return _do(obs, functions, {'P1': value[0], 'P2': value[1]})

def compute_do_RGB(obs, functions, value):
    return _do(obs, functions, {'R': value[0], 'G': value[1], 'B': value[2]})

def compute_do_RGBP1P2(obs, functions, value):
    return _do(obs, functions, {
        'R': value[0], 'G': value[1], 'B': value[2],
        'P1': value[3], 'P2': value[4],
    })
