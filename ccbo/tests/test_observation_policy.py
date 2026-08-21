"""Observe/intervene policy contract for the CBO loop.

Two separable concerns, tested separately:

* ``observation_probability`` -- the *schedule*. We follow the released CBO
  implementation, which divides the coverage ratio by the sample ratio; the
  expression printed in Aglietti et al. (2020) multiplies. Only the clip to
  [0, 1] is ours, and it is behaviour-neutral.
* the loop's decision rule -- the *budget guard* and the phase-entry forcing,
  which must sit outside the probabilistic branch: an exhausted pool always
  means intervene, whatever the draw or the forcing flag says.

Dataset-free: driven through a stub graph so a whole run costs milliseconds
and the observational budget is exactly controllable.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from ccbo.cbo.cbo import CBO
from ccbo.cbo.utils import observation_probability, update_hull

RANGES = OrderedDict(X=[-3.0, 3.0], Z=[-3.0, 3.0])
MANIP = ["X", "Z"]
ES = [["X", "Z"]]


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------

def _obs(n=100, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"X": rng.randn(n), "Z": rng.randn(n),
                         "Y": rng.randn(n)})


def test_epsilon_is_the_division_form_of_the_reference_implementation():
    obs = _obs()
    total = 36.0                       # the 6x6 box
    cap = 150
    raw, eps = observation_probability(obs, MANIP, total, cap)
    ratio = update_hull(obs, MANIP) / total
    assert raw == pytest.approx(ratio / (len(obs) / cap))
    # ... and emphatically NOT the multiplication form printed in the paper.
    assert raw != pytest.approx(ratio * (len(obs) / cap))


def test_epsilon_uses_the_effective_cap_not_the_requested_max_N():
    """A pool shorter than max_N must bind: obs_cap = min(max_N, |pool|)."""
    obs = _obs()
    total = 36.0
    raw_declared, _ = observation_probability(obs, MANIP, total, 150)
    raw_effective, _ = observation_probability(obs, MANIP, total, 120)
    assert raw_effective < raw_declared


def test_epsilon_is_always_a_valid_probability():
    obs = _obs()
    for cap in (1, 10, 100, 150, 10_000):
        raw, eps = observation_probability(obs, MANIP, 36.0, cap)
        assert 0.0 <= eps <= 1.0
        assert eps == pytest.approx(min(max(raw, 0.0), 1.0))


def test_epsilon_is_zero_on_degenerate_inputs():
    obs = _obs()
    assert observation_probability(obs, MANIP, 36.0, 0)[1] == 0.0
    assert observation_probability(obs, MANIP, 0.0, 150)[1] == 0.0
    assert observation_probability(obs.iloc[:0], MANIP, 36.0, 150)[1] == 0.0


# ---------------------------------------------------------------------------
# The decision rule, through a real (stubbed) run
# ---------------------------------------------------------------------------

class _StubGraph:
    """Minimal GraphStructure surface the CBO loop actually touches."""

    def __init__(self):
        self.refit_calls = []

    def define_SEM(self):
        return OrderedDict([
            ("X", lambda epsilon, **kw: epsilon[0]),
            ("Z", lambda epsilon, X, **kw: X + epsilon[1]),
            ("Y", lambda epsilon, X, Z, **kw: X + Z + epsilon[2]),
        ])

    def get_interventional_ranges(self):
        return RANGES

    def refit_models(self, observational_samples):
        self.refit_calls.append(len(observational_samples))
        return {}

    def get_all_do(self):
        return {"compute_do_XZ":
                lambda obs, functions, value: (float(obs["Y"].mean()), 1.0)}


def _target(arm, values):
    v = np.ravel(values)
    return float(v[0] ** 2 + v[1] ** 2)


def _run(trials, *, pool_n=5000, initial=100, max_N=150, batch=20,
         seed=0, force_observe_on_entry=None, causal_prior=False,
         coverage_total=36.0, fixed_draw=None):
    """One stubbed CBO run; returns (trial_log, state, graph).

    ``initial`` doubles as the entry cursor -- the frame is always the first
    ``initial`` rows of the pool, which is the contract the loop enforces.
    ``coverage_total`` is the denominator of the coverage ratio: making it
    huge drives epsilon to ~0, isolating the *forced* decisions from the
    probabilistic ones. ``fixed_draw`` pins the loop's own U[0,1) draw
    without disturbing the RNG that emukit's optimizer needs.
    """
    np.random.seed(seed)
    full = _obs(pool_n, seed=1)
    obs = full.iloc[:initial].copy()
    graph = _StubGraph()
    x = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    y = np.array([[0.0], [2.0], [2.0]])

    real_uniform = np.random.uniform

    def _pinned(*args, **kwargs):
        # The loop's decision draw is the only bare uniform(0., 1.) call;
        # everything else (emukit's sampling) keeps the real generator.
        if args == (0., 1.) and not kwargs:
            return fixed_draw
        return real_uniform(*args, **kwargs)

    if fixed_draw is not None:
        np.random.uniform = _pinned
    try:
        _, state = CBO(
            trials, ES, MANIP, [x.copy()], [y.copy()], np.array([0.0, 0.0]),
            0.0, "XZ", RANGES, {}, obs, coverage_total, graph, batch,
            {"X": lambda v: 1.0, "Z": lambda v: 1.0}, full, "min", max_N,
            initial, 3, Causal_prior=causal_prior, return_state=True,
            target_evaluator=_target,
            force_observe_on_entry=force_observe_on_entry)
    finally:
        np.random.uniform = real_uniform
    return state["trial_log"], state, graph


def _types(log):
    return [e["type"] for e in log]


def test_fresh_run_observes_first_then_intervenes():
    log, state, _ = _run(3)
    assert _types(log)[:2] == ["observe", "intervene"]
    assert log[0]["forced"] == "observe"
    assert log[1]["forced"] == "intervene"


def test_force_observe_on_entry_false_permits_an_intervention_at_step_zero():
    """The old code doctored `uniform = 0.` at i == 0, so the entry
    observation was unconditional whenever epsilon > 0 and could not be
    turned off. With epsilon pinned to ~0 the flag alone decides."""
    on, _, _ = _run(3, force_observe_on_entry=True, coverage_total=1e9)
    off, _, _ = _run(3, force_observe_on_entry=False, coverage_total=1e9)
    assert on[0]["type"] == "observe" and on[0]["forced"] == "observe"
    assert off[0]["type"] == "intervene" and off[0]["forced"] is None


def test_fresh_phase_observes_the_next_unseen_batch():
    """HQCBO phase 2: a fresh phase entered mid-pool still observes, and it
    consumes the NEXT unseen batch rather than re-revealing the first."""
    log, state, _ = _run(2, initial=120, force_observe_on_entry=True,
                         coverage_total=1e9)
    assert log[0]["type"] == "observe"
    assert (log[0]["obs_start"], log[0]["obs_end"]) == (120, 140)
    assert state["obs_cursor"] == 140


def test_fresh_phase_does_not_observe_when_the_budget_is_exhausted():
    log, state, _ = _run(2, initial=150, force_observe_on_entry=True)
    assert _types(log) == ["intervene", "intervene"]
    assert all(e["exhausted"] for e in log)
    assert state["num_observations_collected"] == 0


@pytest.mark.parametrize("draw", [0.0, 1.0])
def test_exhaustion_overrides_the_draw_however_it_falls(draw):
    """Neither an always-observe nor an always-intervene draw may push a run
    past the cap."""
    log, state, _ = _run(8, fixed_draw=draw)
    assert state["obs_cursor"] <= state["obs_cap"] == 150
    assert state["num_observations_collected"] <= 50
    spans = [(e["obs_start"], e["obs_end"]) for e in log
             if e["type"] == "observe"]
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_batches_are_disjoint_and_the_last_one_is_short():
    """u = 0 always observes, so the whole budget drains in order."""
    log, state, _ = _run(6, fixed_draw=0.0)
    spans = [(e["obs_start"], e["obs_end"]) for e in log
             if e["type"] == "observe"]
    assert spans == [(100, 120), (120, 140), (140, 150)]
    assert state["obs_cursor"] == 150


def test_run_is_deterministic_under_a_fixed_seed():
    a, _, _ = _run(6, seed=3)
    b, _, _ = _run(6, seed=3)
    assert _types(a) == _types(b)
    assert [e["u"] for e in a] == [e["u"] for e in b]


def test_causal_prior_path_also_runs():
    log, state, _ = _run(3, causal_prior=True)
    assert len(log) == 3
    assert state["num_observations_collected"] > 0


def test_observation_refreshes_the_graph_and_counts_actions():
    log, state, graph = _run(4)
    n_obs_actions = _types(log).count("observe")
    assert len(graph.refit_calls) == n_obs_actions
    assert state["num_observations_collected"] == sum(
        e["obs_end"] - e["obs_start"] for e in log if e["type"] == "observe")
    # The observational frame really grew.
    assert len(state["observational_samples"]) == state["obs_cursor"]


def test_exhausted_at_entry_still_builds_the_causal_prior():
    """Regression: the causal prior used to be built only by the first
    observe trial, so a phase entered with the budget already spent reached
    ``update_BO_models`` with ``mean_function=None`` and crashed. That path
    was unreachable while step 0 always observed; the budget guard makes it
    reachable, and HQCBO phase 2 hits it whenever phase 1 drained the pool."""
    log, state, _ = _run(3, initial=150, causal_prior=True,
                         force_observe_on_entry=True)
    assert _types(log) == ["intervene"] * 3
    assert log[0]["exhausted"] is True
    assert state["num_observations_collected"] == 0
