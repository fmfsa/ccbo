"""Corrected numerical semantics for the DCBO authors' stack (engine v3).

Two defects of the pinned upstream (``third_party/DCBO`` @ 85a9bdf) were
mirrored by QDCBO so that the finest-partition identity held *against the
stock code*:

1. ``fit_arcs`` fits every transition mechanism on the wrong time slice:
   in its fork branch the estimand is read at the *parent's* slice, in its
   single-edge and many-to-one branches the regressors are read at the
   *child's* slice — while the fitted function is later evaluated on the
   parent values at ``t - 1``.  The corrected version regresses each child
   at its own slice on each parent at the parent's own slice.
2. ``sequential_sample_from_SEM_hat`` tests intervention values (and the
   seed) by truthiness, so ``do(X = 0.0)`` is silently dropped and
   ``seed=0`` is ignored.  The corrected version tests ``is not None``.

:func:`apply` installs the corrected functions at every binding the stock
package uses (``dcbo.bases.dcbo_base``, ``dcbo.methods.cbo``,
``dcbo.utils.gp_utils``, ``dcbo.utils.sequential_sampling``,
``dcbo.utils.sem_utils.sem_estimate``) or restores the stock originals
(``stock_quirks=True``, the historical-reproduction mode used by the
identity test and by ``--stock-quirks`` in the runner).  ``third_party``
is never edited.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Callable

import numpy as np

from ccbo.qdcbo.quotient_dbn import ensure_dcbo_on_path

ensure_dcbo_on_path()

import dcbo.bases.dcbo_base as _db                      # noqa: E402
import dcbo.methods.cbo as _cbo                          # noqa: E402
import dcbo.utils.gp_utils as _gu                        # noqa: E402
import dcbo.utils.sequential_sampling as _ss             # noqa: E402
import dcbo.utils.sem_utils.sem_estimate as _se          # noqa: E402
from dcbo.utils.dag_utils.adjacency_matrix_utils import (  # noqa: E402
    get_emit_and_trans_adjacency_mats)
from sklearn.neighbors import KernelDensity              # noqa: E402

_STOCK = {
    "fit_arcs": _se.fit_arcs,
    "sampler": _ss.sequential_sample_from_SEM_hat,
}
_MODE = {"stock_quirks": None}


def fixed_fit_arcs(G, data: dict, emissions: bool) -> dict:
    """Stock ``fit_arcs`` with every regressor read at the parent's slice
    and every estimand at the child's slice."""
    fit_gp = _gu.fit_gp
    if emissions:
        A, _ = get_emit_and_trans_adjacency_mats(G)
    else:
        _, A = get_emit_and_trans_adjacency_mats(G)
    edge_fit_track_mat = deepcopy(A)
    T = G.T
    nodes = np.array(G.nodes())
    fncs = {t: {} for t in range(T)}

    fork_idx = np.where(A.sum(axis=1) > 1)[0]
    fork_nodes = nodes[fork_idx]
    if any(fork_nodes):
        for i, v in zip(fork_idx, fork_nodes):
            coords = np.where(A[i, :] == 1)[0]
            ch = nodes[coords].tolist()
            var, t = v.split("_")
            t = int(t)
            xx = data[var][:, t].reshape(-1, 1)               # parent at ITS slice
            for j, y in enumerate(ch):
                var_y, t_y = y.split("_")
                t_y = int(t_y)
                yy = data[var_y][:, t_y].reshape(-1, 1)       # child at ITS slice
                if t_y == t:
                    fncs[t][(v, j, y)] = fit_gp(x=xx, y=yy)   # emission
                else:
                    fncs[t + 1][(v, j, y)] = fit_gp(x=xx, y=yy)  # transition
                edge_fit_track_mat[i, coords[j]] -= 1

    if emissions:
        for v in nodes[np.where(A.sum(axis=0) == 0)]:
            var, t = v.split("_")
            t = int(t)
            xx = data[var][:, t].reshape(-1, 1)
            fncs[t][(None, v)] = KernelDensity(kernel="gaussian").fit(xx)

    for i, j in zip(*np.where(edge_fit_track_mat == 1)):
        pa_y, t_pa = nodes[i].split("_")
        y, t_y = nodes[j].split("_")
        if emissions:
            assert t_pa == t_y, (i, j, nodes[i], nodes[j])
        else:
            assert t_pa != t_y, (i, j, nodes[i], nodes[j])
        t = int(t_y)
        xx = data[pa_y][:, int(t_pa)].reshape(-1, 1)          # parent at ITS slice
        yy = data[y][:, t].reshape(-1, 1)
        fncs[t][(nodes[i],)] = fit_gp(x=xx, y=yy)
        edge_fit_track_mat[i, j] -= 1

    assert edge_fit_track_mat.sum() == 0

    many_to_one = np.where(A.sum(axis=0) > 1)[0]
    if any(many_to_one):
        for i, v in zip(many_to_one, nodes[many_to_one]):
            y, y_t = v.split("_")
            t = int(y_t)
            pa_y = nodes[np.where(A[:, i] == 1)]
            assert len(pa_y) > 1, (pa_y, y, many_to_one)
            xx = np.hstack([data[vv.split("_")[0]][:, int(vv.split("_")[1])].reshape(-1, 1)
                            for vv in pa_y])                  # each parent at ITS slice
            yy = data[y][:, t].reshape(-1, 1)
            if y_t == pa_y[0].split("_")[1]:
                fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)
            else:
                assert t != 0, (t, pa_y, y)
                fncs[t][tuple(pa_y)] = fit_gp(x=xx, y=yy)
    return fncs


def fixed_sequential_sample_from_SEM_hat(static_sem: OrderedDict,
                                         dynamic_sem: OrderedDict,
                                         timesteps: int,
                                         node_parents: Callable,
                                         initial_values: dict = None,
                                         interventions: dict = None,
                                         seed: int = None) -> OrderedDict:
    """Stock ``sequential_sample_from_SEM_hat`` with ``is not None`` clamp
    and seed semantics (``do(X=0.0)`` is an intervention)."""
    if seed is not None:
        np.random.seed(seed)
    sample = OrderedDict([(k, np.zeros(timesteps)) for k in static_sem])
    if initial_values:
        assert sample.keys() == initial_values.keys()

    def _clamped(var, t):
        return interventions is not None and interventions[var][t] is not None

    for t in range(timesteps):
        if t == 0 or dynamic_sem is None:
            for var, function in static_sem.items():
                if interventions and initial_values:
                    if interventions[var][t] is not None and initial_values[var] is not None:
                        raise ValueError(
                            "You cannot provided an initial value and an intervention "
                            "for the same location(var,time) in the graph.")
                if _clamped(var, t):
                    sample[var][t] = interventions[var][t]
                elif initial_values:
                    sample[var][t] = initial_values[var]
                else:
                    V = var + "_" + str(t)
                    if node_parents(V, t):
                        sample[var][t] = function(t, None, node_parents(V, t), sample)
                    else:
                        sample[var][t] = function(t, (None, V))
        else:
            assert dynamic_sem is not None
            for var, function in dynamic_sem.items():
                V = var + "_" + str(t)
                if _clamped(var, t):
                    sample[var][t] = interventions[var][t]
                elif not node_parents(V, t - 1) and not node_parents(V, t):
                    sample[var][t] = function(t, (None, V))
                else:
                    sample[var][t] = function(t, node_parents(V, t - 1), node_parents(V, t), sample)
    return sample


def apply(stock_quirks: bool = False) -> None:
    """Install the corrected (``False``) or the stock (``True``) functions at
    every import site of the DCBO package."""
    fa = _STOCK["fit_arcs"] if stock_quirks else fixed_fit_arcs
    sm = _STOCK["sampler"] if stock_quirks else fixed_sequential_sample_from_SEM_hat
    _se.fit_arcs = fa
    _db.fit_arcs = fa
    _cbo.fit_arcs = fa
    _ss.sequential_sample_from_SEM_hat = sm
    _gu.sequential_sample_from_SEM_hat = sm
    _db.sequential_sample_from_SEM_hat = sm
    _MODE["stock_quirks"] = bool(stock_quirks)


def current_mode():
    """``None`` (never applied), ``True`` (stock) or ``False`` (fixed)."""
    return _MODE["stock_quirks"]
