"""Structural validation of the MinimalBench family (Experiments A-C).

Dataset-free (the pattern of test_clusterbench10_structure.py): the C-DAG,
the MIS exploration set, and the per-arm prior tiers are derived purely from
the registered assumed structure + partition, with a stub original graph
exposing only ``get_interventional_ranges()``.

Comparison semantics: quotient/MIS metadata is compared with EXACT equality
(C-DAG signatures, exploration-set membership, prior tiers); numerical
trajectory comparisons live in test_minimal_invariance.py with tolerances.

Claims checked (the experimental premise of the paper's Experiments A-C):
  A1  del X1->Y is quotient-redundant: coarse C-DAG identical, fine MIS
      collapses to {X2} only (every X1-containing arm gone).
  A2  add X1->X2 is an intra-cluster bow: coarse C-DAG identical, fine MIS
      unchanged, fine do(X1) drops to the uninformative tier.
  A3  assumed X1<->Y is quotient-VISIBLE: coarse C-DAG differs (bidirected
      C1<->Y), quotient MIS unchanged, cluster arm drops to the
      uninformative tier — QCBO runs but is not protected.
  C   MediatedChain: fine MIS = {X1},{X2} (joint NOT minimal), quotient
      MIS = the whole-cluster arm; the declared refine map splits C1
      validly.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from ccbo.coarsened_graph import CoarsenedGraph
from ccbo import minibench as mb
from ccbo.benchmark import _split_partition


class _StubGraph:
    """Minimal original-graph stand-in: only ranges are read at construction."""

    def __init__(self, ranges):
        self._ranges = ranges

    def get_interventional_ranges(self):
        return OrderedDict(self._ranges)


_RANGES = {
    mb.PP_NAME: [("X1", list(mb.DOMAIN)), ("X2", list(mb.DOMAIN))],
    mb.FD_NAME: [("X1", list(mb.DOMAIN)), ("M", list(mb.DOMAIN))],
    mb.MC_NAME: [("X1", list(mb.DOMAIN)), ("X2", list(mb.MC_X2_BOX))],
}


def _obs(scm):
    nodes = {mb.PP_NAME: mb.PP_NODES, mb.FD_NAME: mb.FD_NODES,
             mb.MC_NAME: mb.MC_NODES}[scm]
    return pd.DataFrame({n: np.zeros(4) for n in nodes})


def _admg_signature(admg):
    """Order-independent fingerprint of the CoarsenedGraph C-DAG."""
    di = frozenset(tuple(sorted(map(str, e))) for e in admg.get("di", set()))
    bi = frozenset(frozenset(map(str, e)) for e in admg.get("bi", set()))
    vs = frozenset(map(str, admg.get("vertices", set())))
    return (vs, di, bi)


def _build(scm, partition, perturbation_id):
    mb.register_variants()
    return CoarsenedGraph(
        _StubGraph(_RANGES[scm]), partition, scm, _obs(scm),
        num_mc_samples=50,
        assumed_graph_name=mb.variant_name(perturbation_id))


def _es_as_set(cg):
    return {tuple(sorted(arm)) for arm in cg._exploration_set}


# ---------------------------------------------------------------------------
# ParallelParent — fine (identity partition) exploration sets
# ---------------------------------------------------------------------------

def test_pp_fine_mis_correct_graph():
    cg = _build(mb.PP_NAME, mb.fine_partition(mb.PP_NAME), "A0")
    assert _es_as_set(cg) == {("X1",), ("X2",), ("X1", "X2")}, (
        f"A0 fine MIS wrong: {cg._exploration_set}")
    # Both singletons and the joint arm are identifiable under the true graph
    # (backdoor on the co-parent; joint = all parents of Y).
    assert all(cg._arm_identifiable[k] for k in cg._arm_identifiable), (
        f"A0 fine tiers wrong: {cg._arm_identifiable}")


def test_pp_fine_mis_collapses_under_a1():
    """Experiment A's premise: del X1->Y removes EVERY X1-containing arm from
    the fine MIS (X1 leaves An(Y) in every mutilation), leaving {X2} alone."""
    cg = _build(mb.PP_NAME, mb.fine_partition(mb.PP_NAME), "A1")
    assert _es_as_set(cg) == {("X2",)}, (
        f"A1 fine MIS must be exactly {{X2}}: {cg._exploration_set}")


def test_pp_fine_mis_unchanged_under_a2_and_a3():
    for pid in ("A2", "A3"):
        cg = _build(mb.PP_NAME, mb.fine_partition(mb.PP_NAME), pid)
        assert _es_as_set(cg) == {("X1",), ("X2",), ("X1", "X2")}, (
            f"{pid} must not change fine MIS membership: {cg._exploration_set}")


def test_pp_fine_a2_flips_do_x1_tier_only():
    """Experiment B's premise: the intra-cluster bow (add X1->X2 on the
    confounded pair) makes do(X1) non-identifiable — the arm stays, its
    prior drops to the uninformative tier. The joint arm keeps its prior."""
    cg = _build(mb.PP_NAME, mb.fine_partition(mb.PP_NAME), "A2")
    assert cg._arm_identifiable[("X1",)] is False, (
        "A2 must flip do(X1) to the uninformative tier")
    assert cg._arm_identifiable[("X1", "X2")], (
        "A2 must NOT touch the joint arm's tier (all parents of Y intervened)")


# ---------------------------------------------------------------------------
# ParallelParent — coarse C-DAG invariance (the contract) + negative control
# ---------------------------------------------------------------------------

def test_pp_coarse_invariance_matches_predicted_protection():
    coarse = mb.coarse_partition(mb.PP_NAME)
    sig0 = _admg_signature(_build(mb.PP_NAME, coarse, "A0")._coarsened_admg)
    for pert in mb.PERTURBATIONS:
        if pert["scm"] != mb.PP_NAME or pert["id"] == "A0":
            continue
        sig = _admg_signature(
            _build(mb.PP_NAME, coarse, pert["id"])._coarsened_admg)
        invariant = (sig == sig0)
        assert invariant == pert["protected"], (
            f"{pert['id']}: predicted protected={pert['protected']} but "
            f"coarse C-DAG invariant={invariant}")
        assert invariant == (not pert["quotient_changed"]), (
            f"{pert['id']}: taxonomy quotient_changed flag inconsistent")


def test_pp_coarse_a3_keeps_cluster_arm_but_drops_tier():
    """The negative control must NOT be an empty-ES degenerate: under A3 the
    quotient MIS still holds the cluster arm (bidirected edges never gate
    membership) but the cluster bow C1->Y + C1<->Y drops it to the
    uninformative tier — QCBO runs, with different priors."""
    coarse = mb.coarse_partition(mb.PP_NAME)
    cg0 = _build(mb.PP_NAME, coarse, "A0")
    cg3 = _build(mb.PP_NAME, coarse, "A3")
    key = ("X1", "X2")
    assert _es_as_set(cg0) == {key} and _es_as_set(cg3) == {key}, (
        f"quotient MIS must be the cluster arm under both: "
        f"{cg0._exploration_set} vs {cg3._exploration_set}")
    assert cg0._arm_identifiable[key], (
        "A0 cluster arm should carry the do-calculus prior")
    assert not cg3._arm_identifiable[key], (
        "A3 cluster arm must drop to the uninformative tier (cluster bow)")
    # And the assumed C-DAG indeed gained the bidirected cluster edge.
    sig0 = _admg_signature(cg0._coarsened_admg)
    sig3 = _admg_signature(cg3._coarsened_admg)
    assert sig0[0] == sig3[0] and sig0[1] == sig3[1] and sig0[2] != sig3[2], (
        "A3 must differ from A0 exactly in the bidirected edge set")


# ---------------------------------------------------------------------------
# FrontDoor — Experiment B: prior corruption of the OPTIMAL arm
# ---------------------------------------------------------------------------

def test_fd_fine_b0_both_arms_identifiable():
    """Correct graph: do(M) via backdoor through X1, do(X1) via front-door
    through M; the joint is not minimal on a chain."""
    cg = _build(mb.FD_NAME, mb.fine_partition(mb.FD_NAME), "B0")
    assert _es_as_set(cg) == {("X1",), ("M",)}, (
        f"FrontDoor fine MIS wrong: {cg._exploration_set}")
    assert all(cg._arm_identifiable[k] for k in cg._arm_identifiable), (
        f"B0 fine tiers wrong: {cg._arm_identifiable}")


def test_fd_fine_b1_flips_both_tiers_membership_intact():
    """Experiment B's premise: the assumed X1<->M costs BOTH fine arms —
    including the optimal do(M) — their identification; membership is
    untouched."""
    cg = _build(mb.FD_NAME, mb.fine_partition(mb.FD_NAME), "B1")
    assert _es_as_set(cg) == {("X1",), ("M",)}, (
        f"B1 must not change fine MIS membership: {cg._exploration_set}")
    assert cg._arm_identifiable[("M",)] is False, (
        "B1 must flip the optimal do(M) to the uninformative tier")
    assert cg._arm_identifiable[("X1",)] is False, (
        "B1 must flip do(X1) to the uninformative tier")


def test_fd_coarse_invariant_and_bow():
    """The quotient drops the intra-cluster X1<->M: the coarse C-DAG is
    identical under B0 and B1 — and it is the ToyGraph bow (C1->Y, C1<->Y),
    so the cluster arm runs on the uninformative tier in BOTH conditions
    (equal tiers, hence invariance)."""
    coarse = mb.coarse_partition(mb.FD_NAME)
    cg0 = _build(mb.FD_NAME, coarse, "B0")
    cg1 = _build(mb.FD_NAME, coarse, "B1")
    assert _admg_signature(cg0._coarsened_admg) == \
        _admg_signature(cg1._coarsened_admg), (
        "B1 must be invisible in the quotient")
    key = ("M", "X1")
    assert _es_as_set(cg0) == {key} and _es_as_set(cg1) == {key}
    assert cg0._arm_identifiable[key] is False, (
        "the cluster arm is a bow with Y: uninformative tier (ToyGraph "
        "pattern) — in both conditions")
    assert cg1._arm_identifiable[key] is False


# ---------------------------------------------------------------------------
# MediatedChain — exploration sets and refine map
# ---------------------------------------------------------------------------

def test_mc_fine_mis_excludes_joint_arm():
    """On the chain X1->X2->Y the joint {X1,X2} is NOT minimal (mutilating X2
    severs X1 from An(Y)) — the sub-cluster optimum is reachable only through
    the singleton do(X1)."""
    cg = _build(mb.MC_NAME, mb.fine_partition(mb.MC_NAME), "C0")
    assert _es_as_set(cg) == {("X1",), ("X2",)}, (
        f"MediatedChain fine MIS wrong: {cg._exploration_set}")
    assert all(cg._arm_identifiable[k] for k in cg._arm_identifiable), (
        "no confounders: every fine arm is identifiable")


def test_mc_quotient_mis_is_the_cluster_arm():
    cg = _build(mb.MC_NAME, mb.coarse_partition(mb.MC_NAME), "C0")
    assert _es_as_set(cg) == {("X1", "X2")}, (
        f"MediatedChain quotient MIS wrong: {cg._exploration_set}")
    assert cg._arm_identifiable[("X1", "X2")], (
        "the whole-cluster arm is identifiable (no confounders)")


def test_mc_refine_map_splits_validly():
    coarse = mb.coarse_partition(mb.MC_NAME)
    new_partition, split = _split_partition(
        coarse, mb.MC_REFINE_MAP, ["X1", "X2"])
    assert split == [frozenset({"X1", "X2"})]
    assert sorted(new_partition, key=lambda s: sorted(s)) == sorted(
        mb.fine_partition(mb.MC_NAME), key=lambda s: sorted(s)), (
        f"one split must reach the identity partition: {new_partition}")


# ---------------------------------------------------------------------------
# Oracles (numeric identities of the closed forms; cheap, no MC)
# ---------------------------------------------------------------------------

def test_oracle_identities():
    o = mb.ORACLE[mb.PP_NAME]
    assert o["y_star"] == 0.0
    assert o["best_do_x1"] == pytest.approx(4.25)
    assert o["best_do_x2"] == pytest.approx(4.25 * mb.PP_LAM)
    assert o["gap"] == pytest.approx(4.25 * mb.PP_LAM)
    assert mb.pp_do_joint(*o["x_star"]) == pytest.approx(0.0)
    assert mb.pp_do_x1(mb.PP_A) == pytest.approx(o["best_do_x1"])
    assert mb.pp_do_x2(mb.PP_B) == pytest.approx(o["best_do_x2"])
    assert mb.PP_CORR == pytest.approx(0.64)

    o = mb.ORACLE[mb.FD_NAME]
    assert o["y_star"] == 0.0
    assert o["best_do_x1"] == pytest.approx(
        mb.FD_AMP * (1 - mb.FD_W / np.hypot(mb.FD_W, mb.FD_SIGMA_M)))
    assert o["v_pi"] == 0.0
    assert 0.0 < o["obs_mean_y"] < mb.FD_AMP
    assert mb.fd_do_m(mb.FD_C) == pytest.approx(0.0)
    assert mb.fd_do_x1(mb.FD_C / mb.FD_B) == pytest.approx(o["best_do_x1"])
    # The well is narrow relative to the domain (the global-search premise).
    assert (mb.DOMAIN[1] - mb.DOMAIN[0]) / mb.FD_W >= 10

    o = mb.ORACLE[mb.MC_NAME]
    assert o["y_star"] == pytest.approx(0.04)
    assert o["v_pi"] == pytest.approx(4.0)
    assert o["price"] == pytest.approx(3.96)
    assert mb.mc_do_x1(o["x1_star"]) == pytest.approx(o["y_star"])
    assert min(mb.mc_do_x2(x) for x in np.linspace(*mb.MC_X2_BOX, 401)) == \
        pytest.approx(o["v_pi"])
    # The mechanism: the fine optimum drives X2's natural response outside
    # the policy box.
    assert mb.MC_BETA * o["x1_star"] > mb.MC_X2_BOX[1]


@pytest.mark.slow
def test_mc_oracle_agreement():
    """Monte-Carlo arm values through the *implemented* SEMs match the
    closed forms in minibench (guards against SEM/oracle drift without
    regenerating observations.pkl; the generators run the strict gate)."""
    from ccbo.generic_do import compute_do_generic
    from ccbo.cbo.graphs.ParallelParent import ParallelParent
    from ccbo.cbo.graphs.MediatedChain import MediatedChain

    boot_pp = pd.DataFrame({c: [] for c in mb.PP_NODES})
    sem_pp = ParallelParent(boot_pp).define_SEM
    for arm, value, analytic in [
            (['X1', 'X2'], np.array([mb.PP_A, mb.PP_B]), 0.0),
            (['X1'], mb.PP_A, mb.ORACLE[mb.PP_NAME]['best_do_x1']),
            (['X2'], mb.PP_B, mb.ORACLE[mb.PP_NAME]['best_do_x2'])]:
        mean, _ = compute_do_generic(sem_pp, arm, boot_pp, {}, value,
                                     num_mc_samples=50000, seed=1)
        assert mean == pytest.approx(analytic, abs=0.1), (arm, mean, analytic)

    from ccbo.cbo.graphs.FrontDoor import FrontDoor
    boot_fd = pd.DataFrame({c: [] for c in mb.FD_NODES})
    sem_fd = FrontDoor(boot_fd).define_SEM
    for arm, value, analytic in [
            (['M'], mb.FD_C, 0.0),
            (['X1'], mb.FD_C / mb.FD_B, mb.ORACLE[mb.FD_NAME]['best_do_x1']),
            (['X1', 'M'], np.array([0.0, mb.FD_C]), 0.0)]:
        mean, _ = compute_do_generic(sem_fd, arm, boot_fd, {}, value,
                                     num_mc_samples=50000, seed=1)
        assert mean == pytest.approx(analytic, abs=0.05), (arm, mean, analytic)

    boot_mc = pd.DataFrame({c: [] for c in mb.MC_NODES})
    sem_mc = MediatedChain(boot_mc).define_SEM
    x1_star = mb.ORACLE[mb.MC_NAME]['x1_star']
    fine, _ = compute_do_generic(sem_mc, ['X1'], boot_mc, {}, x1_star,
                                 num_mc_samples=50000, seed=1)
    coarse, _ = compute_do_generic(sem_mc, ['X2'], boot_mc, {},
                                   mb.MC_X2_BOX[1], num_mc_samples=50000,
                                   seed=1)
    assert fine == pytest.approx(mb.ORACLE[mb.MC_NAME]['y_star'], abs=0.05)
    assert coarse == pytest.approx(mb.ORACLE[mb.MC_NAME]['v_pi'], abs=0.05)
    assert coarse - fine == pytest.approx(mb.mc_price(), abs=0.1)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"{len(fns)} structural checks passed", file=sys.stderr)
