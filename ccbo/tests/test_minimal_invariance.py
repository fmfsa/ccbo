"""End-to-end invariance regression for the MinimalBench suite (slow).

Full (small-budget) CBO runs through the real pipeline, asserting the
paper's Experiment A/B claims as regressions:

  * QCBO trajectories under the protected edits A1 (quotient-redundant
    deletion) and A2 (intra-cluster bow) equal the A0 baseline — arm/value
    sequences exactly, numeric trajectories to tolerance.
  * QCBO under the quotient-visible A3 diverges (negative control).
  * Fine CBO under A1 is confined to X2 arms and lands near the analytic
    gap 4.25*LAM, bounded away from y* = 0.

Comparison semantics: exact equality for arm choices and intervention
values (strings from the runner), np.allclose for trajectories.

Requires ccbo/cbo/data/ParallelParent/observations.pkl (checked in;
regenerate with python -m ccbo.cbo.data.ParallelParent.generate_observations).
"""

import numpy as np
import pytest

from ccbo import minibench as mb

SEEDS = (0, 1)
TRIALS = 5
ATOL = 1e-8


def _run(cond, arm, seed):
    from ccbo.minimal_suite import run_plain_unit
    rows, info = run_plain_unit(mb.PP_NAME, cond, arm, seed, TRIALS)
    traj = np.array([r[1] for r in rows])
    arms = [(r[3], r[4]) for r in rows]
    return traj, arms, info


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_qcbo_invariant_under_protected_edits(seed):
    traj0, arms0, _ = _run("A0", "QCBO", seed)
    for cond in ("A1", "A2"):
        traj, arms, _ = _run(cond, "QCBO", seed)
        assert arms == arms0, (
            f"QCBO arm/value sequence must be identical under {cond}")
        assert np.allclose(traj, traj0, atol=ATOL), (
            f"QCBO trajectory must be invariant under {cond}: "
            f"max|delta|={np.abs(traj - traj0).max():.3g}")


@pytest.mark.slow
def test_qcbo_diverges_under_quotient_visible_edit():
    seed = SEEDS[0]
    traj0, arms0, _ = _run("A0", "QCBO", seed)
    traj3, arms3, info3 = _run("A3", "QCBO", seed)
    assert ["X1", "X2"] in info3["es"], (
        "A3 must keep the cluster arm (no empty-ES degenerate)")
    assert "X1+X2" in info3["uninformative_arms"], (
        "A3 must drop the cluster arm to the uninformative tier")
    assert (arms3 != arms0) or not np.allclose(traj3, traj0, atol=ATOL), (
        "the quotient-visible A3 must change the QCBO trajectory")


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_qcbo_invariant_on_frontdoor(seed):
    """Experiment B (FrontDoor): the assumed intra-cluster X1<->M flips both
    fine arms' prior tiers, but the quotient drops it — QCBO is exactly
    invariant."""
    from ccbo.minimal_suite import run_plain_unit
    out = {}
    for cond in ("B0", "B1"):
        rows, _ = run_plain_unit(mb.FD_NAME, cond, "QCBO", seed, TRIALS)
        out[cond] = (np.array([r[1] for r in rows]),
                     [(r[3], r[4]) for r in rows])
    assert out["B0"][1] == out["B1"][1], (
        "QCBO arm/value sequence must be identical under B1")
    assert np.allclose(out["B0"][0], out["B1"][0], atol=ATOL)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_fine_cbo_pays_the_analytic_gap_under_a1(seed):
    traj, arms, info = _run("A1", "CBO", seed)
    played = {a for a, _ in arms} - {"", "init"}
    assert played <= {"X2"}, (
        f"A1 fine CBO must only ever play X2 arms: {played}")
    gap = mb.pp_gap()
    assert traj[-1] == pytest.approx(gap, abs=0.5), (
        f"A1 fine CBO should land near the analytic gap {gap}: {traj[-1]}")
    assert traj[-1] > gap / 2, (
        "A1 fine CBO must stay bounded away from y* = 0 (permanent damage)")


# ---------------------------------------------------------------------------
# Engine v3 arms
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_qcbonp_invariant_under_protected_edits(seed):
    """The no-prior quotient arm reads the graph only through the coarse
    arm set, so it must be exactly invariant under A1/A2 as well."""
    traj0, arms0, _ = _run("A0", "QCBONP", seed)
    for cond in ("A1", "A2"):
        traj, arms, _ = _run(cond, "QCBONP", seed)
        assert arms == arms0
        assert np.allclose(traj, traj0, atol=ATOL)


@pytest.mark.slow
def test_bo_and_bos_byte_identical_across_conditions():
    """BO and BOS read no graph: every condition of an SCM yields the same
    rows (the free harness check behind Table 1's BO column)."""
    from ccbo.minimal_suite import run_plain_unit, run_bo_unit
    seed = SEEDS[0]
    bos = {c: run_plain_unit(mb.PP_NAME, c, "BOS", seed, TRIALS)[0] for c in ("A0", "A1", "A3")}
    bo = {c: run_bo_unit(mb.PP_NAME, c, seed, TRIALS)[0] for c in ("A0", "A1", "A3")}
    assert bos["A0"] == bos["A1"] == bos["A3"]
    assert bo["A0"] == bo["A1"] == bo["A3"]
