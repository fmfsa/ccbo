"""Tri-state identification gate (engine v3).

``_ananke_id_check`` returns ``identified`` / ``not_identified`` / ``error``;
an ``error`` is fail-closed (constant fallback prior, recorded separately)
and must never be reported as identification.  The audit test asserts that
no arm set used by the paper's experiments ever hits the ``error`` state.
"""

import numpy as np
import pandas as pd
import pytest

from ccbo import adjustment as adj
from ccbo import minibench as mb
from ccbo.coarsened_graph import CoarsenedGraph
from ccbo.tests.test_minibench_structure import _build, _obs, _StubGraph, _RANGES


def _pp_admg():
    mb.register_variants()
    cg = _build(mb.PP_NAME, mb.fine_partition(mb.PP_NAME), "A0")
    return cg


def test_gate_reports_identified_on_parallel_parent():
    cg = _pp_admg()
    status, functional, reason = adj._ananke_id_check(
        cg._coarsened_admg, {frozenset({"X1"})}, frozenset({"Y"}))
    assert status == "identified" and reason is None
    assert all(v == "identified" for v in cg._arm_gate_status.values())
    assert all(cg._arm_identifiable.values())


def test_gate_error_is_fail_closed(monkeypatch):
    import ananke.identification as ai

    def boom(self, *a, **k):
        raise ValueError("forced")

    monkeypatch.setattr(ai.OneLineID, "id", boom)
    cg = _pp_admg()
    with pytest.warns(RuntimeWarning):
        status, functional, reason = adj._ananke_id_check(
            cg._coarsened_admg, {frozenset({"X1"})}, frozenset({"Y"}))
    assert status == "error" and functional is None and "forced" in reason
    with pytest.warns(RuntimeWarning):
        do_fn, ident = adj.make_cdag_do_function(
            cg._coarsened_admg, ["X1"], cg.partition, "Y", _obs(mb.PP_NAME))
    assert do_fn is None
    assert ident["gate_status"] == "error" and ident["identifiable"] is False
    # the graph records the error state, never a positive identification
    with pytest.warns(RuntimeWarning):
        cg_err = _build(mb.PP_NAME, mb.fine_partition(mb.PP_NAME), "A0")
    assert set(cg_err._arm_gate_status.values()) == {"error"}
    assert not any(cg_err._arm_identifiable.values())
    assert set(cg_err._gate_errors) == set(cg_err._arm_gate_status)


def test_ident_dicts_carry_gate_status_and_estimator_info():
    cg = _pp_admg()
    do_fn, ident = adj.make_cdag_do_function(
        cg._coarsened_admg, ["X1"], cg.partition, "Y", _obs(mb.PP_NAME)) \
        if False else (None, None)
    # use real data so the GP fits are well posed
    from ccbo.minimal_suite import load_scm
    _, obs, _ = load_scm(mb.PP_NAME, 60)
    do_fn, ident = adj.make_cdag_do_function(
        cg._coarsened_admg, ["X1"], cg.partition, "Y", obs)
    assert ident["gate_status"] == "identified"
    assert ident["estimator"]["variance_policy"] == "total"
    assert "gate_deferred" not in ident
    assert do_fn is not None and hasattr(do_fn, "moments")


@pytest.mark.parametrize("pert", [p["id"] for p in mb.PERTURBATIONS])
@pytest.mark.parametrize("which", ["fine", "coarse"])
def test_paper_arm_sets_never_hit_gate_error_minimal(pert, which):
    scm = next(p["scm"] for p in mb.PERTURBATIONS if p["id"] == pert)
    part = mb.fine_partition(scm) if which == "fine" else mb.coarse_partition(scm)
    cg = _build(scm, part, pert)
    assert cg._arm_gate_status, "no arms?"
    assert set(cg._arm_gate_status.values()) <= {"identified", "not_identified"}
    assert not cg._gate_errors


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["ToyGraph", "CompleteGraph", "SimplifiedCoralGraph"])
@pytest.mark.parametrize("arm", ["CBO", "QCBO"])
def test_paper_arm_sets_never_hit_gate_error_family(dataset, arm):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from run_cbo_family import PARTITIONS, _load_graph
    graph, obs, _ = _load_graph(dataset, 100)
    cg = CoarsenedGraph(graph, PARTITIONS[(dataset, arm)], dataset, obs, num_mc_samples=200)
    assert set(cg._arm_gate_status.values()) <= {"identified", "not_identified"}
    assert not cg._gate_errors
