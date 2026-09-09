"""Graph-free refinement (HQCBOGF) vs graph-informed refinement (HQCBO),
engine v3.  MediatedChain, short budget, seeds 0-1."""

import numpy as np
import pytest

from ccbo import minibench as mb
from ccbo import minimal_suite as ms

TRIALS = 14
SEEDS = (0, 1)


def _run(mode, seed, coarse_traj=None):
    return ms.run_refine_unit(mb.MC_NAME, "C0", seed, TRIALS, num_interventions=3,
                              coarse_traj=coarse_traj, mode=mode)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_graph_free_exposes_all_sub_arms_with_plain_gps(seed):
    rows, info = _run("free", seed)
    assert info["mode"] == "free" and info["trigger_step"] is not None
    ref = info["decision_log"]["refine"]
    assert ref["refined_es"] == [["X1", "X2"], ["X1"], ["X2"]]
    assert ref["refined_prior_mask"] == [True, False, False]
    assert info["split_arms"] == ["X1", "X2"]
    assert info["refused"] is False


@pytest.mark.slow
def test_graph_free_never_touches_the_fine_graph_after_the_split(monkeypatch):
    """Any CoarsenedGraph built on a non-coarse partition after the split is a
    fine-graph access.  Graph-free must complete; graph-informed must trip."""
    import ccbo.coarsened_graph as cgm
    real_init = cgm.CoarsenedGraph.__init__
    coarse = mb.coarse_partition(mb.MC_NAME)

    def guarded(self, original_graph, partition, *a, **k):
        if list(partition) != list(coarse):
            raise AssertionError("fine graph accessed after the split")
        return real_init(self, original_graph, partition, *a, **k)

    monkeypatch.setattr(cgm.CoarsenedGraph, "__init__", guarded)
    rows, info = _run("free", 0)
    assert info["trigger_step"] is not None and len(rows) == TRIALS + 1
    with pytest.raises(AssertionError, match="fine graph accessed"):
        _run("graph", 0)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_phase1_identical_to_qcbo_and_split_is_charged(seed):
    q_rows, q_info = ms.run_plain_unit(mb.MC_NAME, "C0", "QCBO", seed, TRIALS,
                                       num_interventions=3)
    traj = [r[1] for r in q_rows]
    g_rows, g_info = _run("graph", seed, coarse_traj=traj)
    f_rows, f_info = _run("free", seed, coarse_traj=traj)
    t = g_info["trigger_step"]
    assert t == f_info["trigger_step"]
    # identical prefix up to (and excluding the charge on) the trigger row
    for k in range(t):
        assert g_rows[k] == q_rows[k] == f_rows[k]
    # row 0 = init cost of the single coarse arm (3 points x 2 variables)
    assert q_rows[0][2] == g_rows[0][2] == f_rows[0][2] == 6.0
    # split design charged at the trigger row: 2 new arms x 3 points x 1 var
    assert g_rows[t][2] - q_rows[t][2] == pytest.approx(6.0)
    assert f_rows[t][2] - q_rows[t][2] == pytest.approx(6.0)
    assert g_info["split_init_cost"] == f_info["split_init_cost"] == 6.0
    # the free variant's incumbent is monotone and finite
    ys = [r[1] for r in f_rows]
    assert all(ys[i] <= ys[i - 1] + 1e-12 for i in range(1, len(ys)))
