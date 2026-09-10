"""Exact-key caching of the per-arm prior functions (engine v3)."""

import numpy as np

from ccbo.cbo.utils import compute_update_do_functions as cud


class _Graph:
    def __init__(self, fn):
        self.calls = 0
        self._fn = fn

    def get_all_do(self):
        self.calls += 1
        return {"compute_do_X": self._fn}


def test_close_points_get_distinct_entries():
    seen = []

    def do_fn(obs, functions, value):
        seen.append(float(np.ravel(value)[0]))
        return float(np.ravel(value)[0]), 1.0

    g = _Graph(do_fn)
    mean_l, var_l = cud.update_all_do_functions(
        g, [["X"]], None, ["X"], None, {"X": {}}, {"X": {}})
    mean_f, var_f = mean_l[0], var_l[0]
    a = np.array([[0.1234567891]]); b = np.array([[0.1234567892]])
    assert str(a[0]) == str(b[0])                    # NumPy would have collided
    assert mean_f(a)[0, 0] != mean_f(b)[0, 0]
    assert len(seen) == 2


def test_one_call_fills_both_caches():
    counter = {"n": 0}

    def do_fn(obs, functions, value):
        counter["n"] += 1
        return 2.0, 3.0

    g = _Graph(do_fn)
    cm, cv = {"X": {}}, {"X": {}}
    mean_l, var_l = cud.update_all_do_functions(g, [["X"]], None, ["X"], None, cm, cv)
    mean_f, var_f = mean_l[0], var_l[0]
    x = np.array([[0.4]])
    assert mean_f(x)[0, 0] == 2.0 and var_f(x)[0, 0] == 3.0
    assert counter["n"] == 1
    assert len(cm["X"]) == 1 and len(cv["X"]) == 1


def test_prior_mask_skips_get_all_do():
    def boom(obs, functions, value):
        raise AssertionError("do-function must not be evaluated")

    g = _Graph(boom)
    mean_f, var_f = cud.update_all_do_functions(
        g, [["X"], ["Z"]], None, ["X", "Z"], None,
        {"X": {}, "Z": {}}, {"X": {}, "Z": {}}, prior_mask=[False, False])
    assert mean_f == [None, None] and var_f == [None, None]
    assert g.calls == 0
