"""Engine v3 cost axis: the initial design is charged on row 0 and the
ablation arms behave as declared (observe policy, prior usage)."""

import numpy as np
import pytest

from ccbo import minibench as mb
from ccbo.minimal_suite import run_plain_unit, run_bo_unit, all_subsets


TRIALS = 4


def _obs_rows(rows):
    return sum(1 for r in rows[1:] if r[3] == "")


@pytest.mark.slow
@pytest.mark.parametrize("arm, expected_init", [
    ("CBO", 3 * (1 + 1 + 2)),     # fine MIS {X1},{X2},{X1,X2}: 3 pts x (1+1+2)
    ("QCBO", 3 * 2),              # single cluster arm {X1,X2}
    ("BOS", 3 * (1 + 1 + 2)),     # all nonempty subsets
    ("CBONP", 3 * (1 + 1 + 2)),
    ("QCBONP", 3 * 2),
])
def test_row0_is_initial_design_cost(arm, expected_init):
    rows, info = run_plain_unit(mb.PP_NAME, "A0", arm, 0, TRIALS, num_interventions=3)
    assert rows[0][2] == expected_init == info["init_cost"]
    assert rows[0][3] == "init"
    assert all(rows[i][2] >= rows[i - 1][2] for i in range(1, len(rows)))
    assert len(rows) == TRIALS + 1


@pytest.mark.slow
def test_bo_row0_and_no_observations():
    rows, info = run_bo_unit(mb.PP_NAME, "A0", 0, TRIALS, num_interventions=3)
    assert rows[0][2] == 3 * 2 == info["init_cost"]
    assert _obs_rows(rows) == 0 and info["observed"] == 0
    assert all(r[3] == "X1+X2" for r in rows[1:])


@pytest.mark.slow
def test_bos_never_observes_and_uses_all_subsets():
    rows, info = run_plain_unit(mb.PP_NAME, "A0", "BOS", 0, TRIALS, num_interventions=3)
    assert info["es"] == all_subsets(["X1", "X2"]) == [["X1"], ["X2"], ["X1", "X2"]]
    assert _obs_rows(rows) == 0 and info["observed"] == 0
    assert info["decision_log"]["init"]["prior_mask"] == [False, False, False]


@pytest.mark.slow
def test_cbonp_observes_but_never_builds_causal_priors(monkeypatch):
    import ccbo.coarsened_graph as cgm
    calls = {"n": 0}
    real = cgm.CoarsenedGraph.get_all_do

    def counting(self):
        calls["n"] += 1
        return real(self)

    monkeypatch.setattr(cgm.CoarsenedGraph, "get_all_do", counting)
    rows, info = run_plain_unit(mb.PP_NAME, "A0", "CBONP", 0, TRIALS, num_interventions=3)
    assert info["observed"] >= 1                       # forced first observation
    assert calls["n"] == 0
    assert info["decision_log"]["init"]["prior_mask"] == [False, False, False]


@pytest.mark.slow
def test_qcbo_reports_identified_gate_status():
    rows, info = run_plain_unit(mb.PP_NAME, "A0", "QCBO", 0, TRIALS, num_interventions=3)
    assert set(info["gate_status"].values()) <= {"identified", "not_identified"}
    assert info["decision_log"]["init"]["prior_mask"] == [True]
