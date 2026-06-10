"""
Seed-handling contract for the BO loop's target function.

The seeding contract is:
  * Experiment entry points (run_experiment.py / run_experiments.py) seed
    the global RNG once per (seed, method) run.
  * Target evaluations (Intervention_function) use common random numbers
    via a LOCAL RandomState, so the evaluated objective is a deterministic
    surface — but they must NOT touch the global RNG state, otherwise every
    in-loop draw (epsilon-greedy, acquisition restarts) becomes identical
    across nominally different experiment seeds.

A previous version called np.random.seed(1) inside the target function and
silently destroyed seed diversity; these tests lock the fix in.
"""

from collections import OrderedDict

import numpy as np
import pytest

from ccbo.cbo.utils.graph_functions import Intervention_function


def _toy_sem():
    """X -> Y with additive noise; one intervention variable."""
    return OrderedDict([
        ('X', lambda epsilon, **kw: epsilon[0]),
        ('Y', lambda epsilon, X, **kw: 2.0 * X + 0.1 * epsilon[1]),
    ])


def _make_target():
    fn, _space = Intervention_function(
        {'X': 0.0}, model=_toy_sem(), target_variable='Y',
        min_intervention=[-1.0], max_intervention=[1.0])
    return fn


def test_target_function_does_not_touch_global_rng():
    fn = _make_target()
    np.random.seed(123)
    expected_stream = np.random.RandomState(123).uniform(size=4)

    draws = []
    for i in range(4):
        fn(np.array([[0.5]]))  # would reset the stream under the old bug
        draws.append(np.random.uniform())

    assert np.allclose(draws, expected_stream), (
        "Target evaluation perturbed the global RNG stream — in-loop "
        "randomness would be identical across experiment seeds.")


def test_target_function_is_deterministic():
    fn = _make_target()
    a = fn(np.array([[0.3]]))
    np.random.seed(999)  # global state must be irrelevant
    b = fn(np.array([[0.3]]))
    assert float(a[0, 0]) == float(b[0, 0])

    # And it actually responds to the intervention value.
    c = fn(np.array([[-0.3]]))
    assert float(a[0, 0]) != float(c[0, 0])


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
