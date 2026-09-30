"""Newly observed rows must actually reach the causal prior.

Before the fix an observe action was a no-op on the prior:
``CoarsenedGraph.refit_models`` only delegated to the fine graph, leaving
``_obs_samples`` and the memoized ``_do_cache`` untouched -- and the
do-estimators returned by ``make_cdag_do_function`` close over the frame they
were built with and ignore the ``obs`` argument they are handed. On top of
that the point-wise ``x_dict_mean`` / ``x_dict_var`` caches, keyed by
``str(x)``, survived for the whole run, so even a correctly rebuilt estimator
returned pre-observation values at any already-queried point.

The counting test also pins the flip side: an observation must cost exactly
*one* all-arm GP rebuild, not two (the observe branch rebuilds, and the
following intervene trial must not repeat it).
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from ccbo.tests.test_observation_policy import _StubGraph, _target, ES, MANIP, RANGES
from ccbo.cbo.cbo import CBO


# ---------------------------------------------------------------------------
# One observation costs exactly one all-arm rebuild (fast, stubbed)
# ---------------------------------------------------------------------------

def _count_rebuilds(trials, fixed_draw, monkeypatch):
    import ccbo.cbo.cbo as cbo_mod

    calls = []
    real = cbo_mod.update_BO_models

    def counting(mean_f, var_f, dx, dy, prior):
        calls.append(1)
        return real(mean_f, var_f, dx, dy, prior)

    monkeypatch.setattr(cbo_mod, "update_BO_models", counting)

    rng = np.random.RandomState(1)
    full = pd.DataFrame({"X": rng.randn(5000), "Z": rng.randn(5000),
                         "Y": rng.randn(5000)})
    obs = full.iloc[:100].copy()
    x = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    y = np.array([[0.0], [2.0], [2.0]])

    real_uniform = np.random.uniform

    def pinned(*a, **k):
        return fixed_draw if (a == (0., 1.) and not k) else real_uniform(*a, **k)

    np.random.seed(0)
    np.random.uniform = pinned
    try:
        _, state = CBO(
            trials, ES, MANIP, [x], [y], np.array([0.0, 0.0]), 0.0, "XZ",
            RANGES, {}, obs, 36.0, _StubGraph(), 20,
            {"X": lambda v: 1.0, "Z": lambda v: 1.0}, full, "min", 150, 100,
            3, Causal_prior=False, return_state=True,
            target_evaluator=_target)
    finally:
        np.random.uniform = real_uniform
    return len(calls), state["trial_log"]


def _expected_rebuilds(types):
    """One rebuild per observe trial; one per intervene trial EXCEPT when the
    previous trial was an observe, which already rebuilt every arm."""
    n = 0
    for k, t in enumerate(types):
        if t == "observe" or k == 0 or types[k - 1] != "observe":
            n += 1
    return n


@pytest.mark.parametrize("draw", [0.0, 1.0])
def test_an_observation_costs_exactly_one_all_arm_rebuild(draw, monkeypatch):
    """Single-arm exploration set, so one rebuild == one update_BO_models
    call. The observe branch rebuilds every arm; the intervene trial that
    follows must then do nothing, rather than repeating the full rebuild via
    the ``type_trial[-2] == 0`` branch."""
    n_calls, log = _count_rebuilds(6, draw, monkeypatch)
    types = [e["type"] for e in log]
    assert n_calls == _expected_rebuilds(types), (
        f"{n_calls} rebuilds for {types}")

    # Concretely: no intervene trial right after an observe rebuilds anything.
    doubled = sum(1 for k, t in enumerate(types)
                  if t == "intervene" and k and types[k - 1] == "observe")
    assert n_calls == len(types) - doubled


# ---------------------------------------------------------------------------
# The prior really moves (real CoarsenedGraph)
# ---------------------------------------------------------------------------

def _pp_graph(obs):
    from ccbo import minibench as mb
    from ccbo.scm_graphs import get_original_graph
    from ccbo.coarsened_graph import CoarsenedGraph
    mb.register_variants()
    graph = get_original_graph(mb.PP_NAME, obs)
    cg = CoarsenedGraph(graph, mb.coarse_partition(mb.PP_NAME), mb.PP_NAME,
                        obs, num_mc_samples=500)
    return graph, cg


def _pool():
    from ccbo import minibench as mb
    import os
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return pd.read_pickle(os.path.join(
        repo, "ccbo", "cbo", "data", mb.PP_NAME, "observations.pkl"))


@pytest.mark.slow
def test_refit_invalidates_the_do_cache_and_refreshes_the_frames():
    full = _pool()
    obs = full.iloc[:100].copy()
    graph, cg = _pp_graph(obs)

    cg.get_all_do()
    assert cg._do_cache is not None
    assert cg._identification_info

    grown = pd.concat([obs, full.iloc[100:150]], ignore_index=True)
    cg.refit_models(grown)

    assert cg._do_cache is None, "memoized do-estimators must not survive"
    assert cg._identification_info == {}
    assert len(cg._obs_samples) == 150
    # Graph-side column caches, on the quotient wrapper and on the fine graph
    # underneath it (the latter is what fit_all_models would otherwise reuse).
    assert cg.X1.shape[0] == 150
    assert graph.X1.shape[0] == 150


@pytest.mark.slow
def test_prior_changes_after_a_distribution_shifting_batch():
    """A deliberately shifted batch, so the assertion does not depend on the
    fine detail of a fully optimized GP fit."""
    full = _pool()
    obs = full.iloc[:100].copy()
    _, cg = _pp_graph(obs)

    probe = np.array([1.0, -1.0])
    before = cg.get_all_do()["compute_do_X1X2"](obs, {}, probe)

    shifted = full.iloc[100:150].copy()
    shifted["X1"] = shifted["X1"] + 4.0
    shifted["Y"] = shifted["Y"] + 4.0
    grown = pd.concat([obs, shifted], ignore_index=True)

    cg.refit_models(grown)
    after = cg.get_all_do()["compute_do_X1X2"](grown, {}, probe)

    assert not np.isclose(before[0], after[0], atol=1e-6), (
        f"prior mean must move after an observation: {before[0]} -> {after[0]}")


@pytest.mark.slow
def test_pointwise_caches_do_not_survive_an_observation():
    """Through the loop: the same probe queried before and after an observe
    trial must not come back from the str(x)-keyed cache."""
    from ccbo.cbo.utils import update_all_do_functions, initialise_dicts

    full = _pool()
    obs = full.iloc[:100].copy()
    _, cg = _pp_graph(obs)
    es = [["X1", "X2"]]
    _, _, x_dict_mean, x_dict_var, dict_int = initialise_dicts(es, "min")

    mean_f, _ = update_all_do_functions(cg, es, {}, dict_int, obs,
                                        x_dict_mean, x_dict_var)
    probe = np.array([[1.0, -1.0]])
    before = float(np.asarray(mean_f[0](probe)).ravel()[0])
    assert x_dict_mean[dict_int[0]], "probe should now be cached"

    shifted = full.iloc[100:150].copy()
    shifted["X1"] = shifted["X1"] + 4.0
    shifted["Y"] = shifted["Y"] + 4.0
    grown = pd.concat([obs, shifted], ignore_index=True)

    # Exactly what the observe branch does, in order.
    cg.refit_models(grown)
    for c in x_dict_mean.values():
        c.clear()
    for c in x_dict_var.values():
        c.clear()
    mean_f, _ = update_all_do_functions(cg, es, {}, dict_int, grown,
                                        x_dict_mean, x_dict_var)
    after = float(np.asarray(mean_f[0](probe)).ravel()[0])

    assert not np.isclose(before, after, atol=1e-6), (
        "a stale point-wise cache would return the pre-observation value")
