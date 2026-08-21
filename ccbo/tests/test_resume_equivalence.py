"""Splitting a run across two CBO calls must not change the run.

The observation cursor, the observational frame, the per-arm data and the
model list all travel in the state dict, and a resumed call must not re-apply
the fresh-run warm-up (forced observe at step 0, forced intervene at step 1).
Under a continuous RNG stream the split run and the uninterrupted run must
therefore produce the same decisions, the same batches and the same
trajectory.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from ccbo.cbo.cbo import CBO
from ccbo.tests.test_observation_policy import (_StubGraph, _target, ES,
                                                MANIP, RANGES)

COSTS = {"X": lambda v: 1.0, "Z": lambda v: 1.0}


def _pool(n=5000, seed=1):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"X": rng.randn(n), "Z": rng.randn(n),
                         "Y": rng.randn(n)})


def _call(trials, obs, full, x, y, state=None, cursor=None):
    return CBO(
        trials, ES, MANIP, x, y, np.array([0.0, 0.0]), 0.0, "XZ", RANGES, {},
        obs, 36.0, _StubGraph(), 20, COSTS, full, "min", 150, 100, 3,
        Causal_prior=False, return_state=True, target_evaluator=_target,
        state=state, observation_cursor=cursor)


def _fresh_inputs():
    full = _pool()
    obs = full.iloc[:100].copy()
    x = [np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])]
    y = [np.array([[0.0], [2.0], [2.0]])]
    return full, obs, x, y


def _summary(state):
    log = state["trial_log"]
    return {
        "types": [e["type"] for e in log],
        "spans": [(e["obs_start"], e["obs_end"]) for e in log
                  if e["type"] == "observe"],
        "cursor": state["obs_cursor"],
        "collected": state["num_observations_collected"],
        "global_opt": [float(v) for v in state["global_opt"]],
        "cost": [float(v) for v in state["current_cost"]],
    }


def test_split_run_matches_uninterrupted_run():
    np.random.seed(11)
    full, obs, x, y = _fresh_inputs()
    _, whole = _call(6, obs, full, x, y)

    np.random.seed(11)
    full, obs, x, y = _fresh_inputs()
    _, first = _call(3, obs, full, x, y)
    _, second = _call(3, first["observational_samples"], full,
                      first["data_x_list"], first["data_y_list"],
                      state=first)

    a, b = _summary(whole), _summary(second)
    assert a["types"] == b["types"]
    assert a["spans"] == b["spans"]
    assert a["cursor"] == b["cursor"]
    assert a["collected"] == b["collected"]
    assert np.allclose(a["global_opt"], b["global_opt"])
    assert np.allclose(a["cost"], b["cost"])


def test_resumed_call_does_not_replay_the_warm_up():
    """A resumed state must not force an observation at its local step 0 nor
    an intervention at its local step 1."""
    np.random.seed(11)
    full, obs, x, y = _fresh_inputs()
    _, first = _call(3, obs, full, x, y)
    _, second = _call(3, first["observational_samples"], full,
                      first["data_x_list"], first["data_y_list"],
                      state=first)
    resumed_log = second["trial_log"][len(first["trial_log"]):]
    assert all(e["forced"] is None for e in resumed_log), (
        f"resumed trials must be unforced: {resumed_log}")


def test_resumed_call_never_re_reveals_a_batch():
    np.random.seed(11)
    full, obs, x, y = _fresh_inputs()
    _, first = _call(3, obs, full, x, y)
    _, second = _call(6, first["observational_samples"], full,
                      first["data_x_list"], first["data_y_list"],
                      state=first)
    spans = [(e["obs_start"], e["obs_end"]) for e in second["trial_log"]
             if e["type"] == "observe"]
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:])), spans
    assert second["obs_cursor"] <= second["obs_cap"] == 150
    assert len(second["observational_samples"]) == second["obs_cursor"]
