"""MinimalBench: the minimal SCM family behind the mechanism-isolation suite.

This module is the single source of truth for the *structure* of the two
deliberately minimal SCMs used in the paper's controlled experiments
(Experiments A-C), their coarse partitions, the perturbation taxonomy
(paper Table 1 is emitted from ``PERTURBATIONS``), and the closed-form
oracles. Numeric SEM implementations live in the paired GraphStructure
classes (``ccbo/cbo/graphs/ParallelParent.py``, ``MediatedChain.py``);
both read their coefficients from THIS module so nothing can drift.

The two SCMs share one parameterization: quadratic bowls in the target,
Gaussian roots, manipulable domains [-3, 3] (except the MediatedChain
policy box), unit costs, no clipping — every arm value is closed-form.

ParallelParent (Experiments A + B)
----------------------------------
    U  = 0.4 eps0                  (latent; projected X1 <-> X2)
    X1 = U + 0.3 eps1
    X2 = U + 0.3 eps2
    Y  = LAM (X1 - A)^2 + (X2 - B)^2 + 0.1 epsY        A=2, B=-2, LAM=1

Coarse partition C1 = {X1, X2} | {Y}. Closed forms (Var Xi = 0.25):
    y* = 0 at do(X1=A, X2=B)                (the joint arm, uniquely optimal)
    best do(X1) only = B^2 + Var(X2)              = 4.25
    best do(X2) only = LAM (A^2 + Var(X1))        = 4.25 LAM

Conditions:
    A0  correct graph.
    A1  del X1->Y : quotient-REDUNDANT (the cluster edge C1->Y survives via
        X2->Y) so the C-DAG is unchanged, but at the fine level X1 leaves
        An(Y) and *every X1-containing arm drops out of the MIS* — CBO
        permanently loses the joint arm and pays the analytic gap
        G(LAM) = 4.25 LAM. Exploration-set corruption (Experiment A).
    A2  add X1->X2 : intra-cluster bow with the latent X1<->X2 — fine MIS
        unchanged, but do(X1) flips to the uninformative prior tier and
        do(X2)'s adjustment formula changes. Prior corruption without arm
        loss (Experiment B).
    A3  add confounder X1<->Y : quotient-VISIBLE negative control — the
        bidirected C1<->Y appears in the C-DAG (a cluster-level bow with
        C1->Y), so QCBO still runs (quotient MIS is unchanged; bidirected
        edges never gate membership) but its cluster arm drops to the
        uninformative tier: invariance is correctly NOT expected.

FrontDoor (Experiment B)
------------------------
    U  = 0.4 eps0                  (latent; projected X1 <-> Y)
    X1 = U + 0.3 eps1
    M  = B X1 + 0.2 eps2                               B = 2
    Y  = AMP (1 - exp(-(M-C)^2 / 2 W^2)) + G U + 0.1 epsY
                                     AMP = 2, C = 1, W = 0.3, G = 1

The target is a NARROW WELL (width W over the 6-wide domain), not a bowl:
prior corruption is a global-search phenomenon — on a smooth quadratic any
BO self-corrects within a couple of trials (measured sup|Delta| ~ 0), so
the experiment isolates the regime where the do-calculus prior genuinely
matters: finding the well vs searching for it.

Coarse partition C1 = {X1, M} | {Y}; the quotient is the bow C1 -> Y,
C1 <-> Y (the ToyGraph pattern: the cluster arm runs on the uninformative
tier in every condition). Fine MIS = {X1}, {M} (chain: the joint is not
minimal). Closed forms (Gaussian smoothing of the well):
    y* = 0 at do(M = C)                     (do(M): backdoor through X1)
    best do(X1) = AMP (1 - W/sqrt(W^2+SIGMA_M^2)) ~ 0.336  (front-door)
    V(Pi) = 0                               (the cluster arm sets M: lossless)

Condition B1 assumes a spurious intra-cluster confounder X1 <-> M: fine
MIS unchanged, but the assumed fine graph loses identification for BOTH
arms — including the optimal do(M) — so fine CBO forfeits its do-calculus
head start (transient damage, recovered through interventional data),
while the quotient drops intra-cluster bidirected edges and QCBO is
exactly invariant. (Prior corruption needs the corrupted prior to sit on
an arm that matters: on ParallelParent the optimal joint arm's prior is
uncorruptible, which is why Experiment B lives here.)

MediatedChain (Experiment C)
----------------------------
    X1 = 0.5 eps1
    X2 = BETA X1 + 0.2 eps2                            BETA=2
    Y  = (X2 - C)^2 + 0.1 epsY                         C=4

Ranges: X1 in [-3, 3]; X2 in [-2, 2] (the POLICY BOX: setting X1 = C/BETA
drives X2's natural response to C = 4, outside the clampable box).
Fine MIS = {X1}, {X2} (the joint is not minimal: mutilating X2 severs X1
from An(Y)); quotient MIS = {C1} whose whole-cluster arm clamps X2 into
the box. Closed forms:
    y*     = SIGMA_2^2 = 0.04      at do(X1 = C/BETA = 2)
    V(Pi)  = (box_hi - C)^2 = 4    best whole-cluster (X1 irrelevant once
                                    X2 is clamped)
    price  = V(Pi) - y* = 3.96
HQCBO plateaus at V(Pi), splits C1 (REFINE_MAP), and descends toward y*.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ccbo import coarsening
from ccbo.clusterbench10 import apply_ops, is_acyclic

# ---------------------------------------------------------------------------
# Shared parameterization
# ---------------------------------------------------------------------------

SIGMA_Y = 0.1          # target noise (both SCMs)
DOMAIN = (-3.0, 3.0)   # default manipulable range

# --- ParallelParent -------------------------------------------------------
PP_NAME = "ParallelParent"
PP_MANIPULATIVE: List[str] = ["X1", "X2"]
PP_TARGET = "Y"
PP_NODES: List[str] = PP_MANIPULATIVE + [PP_TARGET]
PP_TRUE_EDGES: List[Tuple[str, str]] = [("X1", "Y"), ("X2", "Y")]
PP_CONFOUNDERS: List[Tuple[str, List[str]]] = [("U", ["X1", "X2"])]
PP_COARSE_CLUSTERS: List[List[str]] = [["X1", "X2"]]

PP_A = 2.0             # X1 bowl centre
PP_B = -2.0            # X2 bowl centre
PP_LAM = 1.0           # X1 bowl curvature (the severity dial)
PP_SIGMA_U = 0.4       # latent confounder scale
PP_SIGMA_IN = 0.3      # root idiosyncratic scale

PP_VAR_X = PP_SIGMA_U ** 2 + PP_SIGMA_IN ** 2          # Var(X1) = Var(X2) = 0.25
PP_CORR = PP_SIGMA_U ** 2 / PP_VAR_X                   # corr(X1, X2) = 0.64

# --- FrontDoor ------------------------------------------------------------
FD_NAME = "FrontDoor"
FD_MANIPULATIVE: List[str] = ["X1", "M"]
FD_TARGET = "Y"
FD_NODES: List[str] = FD_MANIPULATIVE + [FD_TARGET]
FD_TRUE_EDGES: List[Tuple[str, str]] = [("X1", "M"), ("M", "Y")]
FD_CONFOUNDERS: List[Tuple[str, List[str]]] = [("U", ["X1", "Y"])]
FD_COARSE_CLUSTERS: List[List[str]] = [["X1", "M"]]

FD_B = 2.0             # X1 -> M slope
FD_C = 1.0             # centre of the Y well in M (interior, off-zero)
FD_G = 1.0             # confounder loading on Y
FD_SIGMA_U = 0.4       # latent confounder scale
FD_SIGMA_1 = 0.3       # X1 idiosyncratic scale
FD_SIGMA_M = 0.2       # M mechanism noise
# Narrow-well target: Y = AMP (1 - exp(-(M-C)^2 / 2 W^2)) + G U + noise.
# Prior corruption is a *global-search* phenomenon: on a smooth 1-D
# quadratic any BO self-corrects within a couple of trials (we measured
# sup|Delta| ~ 0 for the quadratic variant), so Experiment B's target is a
# narrow well (width W over a 6-wide domain) where the do-calculus prior's
# head start is the difference between finding the well immediately and
# searching for it.
FD_AMP = 2.0           # well amplitude (baseline plateau of Y)
FD_W = 0.3             # well width

# --- MediatedChain --------------------------------------------------------
MC_NAME = "MediatedChain"
MC_MANIPULATIVE: List[str] = ["X1", "X2"]
MC_TARGET = "Y"
MC_NODES: List[str] = MC_MANIPULATIVE + [MC_TARGET]
MC_TRUE_EDGES: List[Tuple[str, str]] = [("X1", "X2"), ("X2", "Y")]
MC_CONFOUNDERS: List[Tuple[str, List[str]]] = []
MC_COARSE_CLUSTERS: List[List[str]] = [["X1", "X2"]]

MC_BETA = 2.0          # X1 -> X2 slope
MC_C = 4.0             # Y bowl centre (outside the X2 policy box)
MC_SIGMA_1 = 0.5       # X1 scale
MC_SIGMA_2 = 0.2       # X2 mechanism noise
MC_X2_BOX = (-2.0, 2.0)   # the restricted intervention (policy) box on X2

# Refinement hierarchy for HQCBO on MediatedChain: one split, C1 -> {X1}|{X2}.
MC_REFINE_MAP: Dict[frozenset, List[frozenset]] = {
    frozenset({"X1", "X2"}): [frozenset({"X1"}), frozenset({"X2"})],
}


def coarse_partition(scm: str) -> List[frozenset]:
    """Coarse partition (manipulable clusters + {Y}) for ``scm``."""
    clusters = {PP_NAME: PP_COARSE_CLUSTERS, FD_NAME: FD_COARSE_CLUSTERS,
                MC_NAME: MC_COARSE_CLUSTERS}[scm]
    return [frozenset(c) for c in clusters] + [frozenset({"Y"})]


def fine_partition(scm: str) -> List[frozenset]:
    """Identity (all-singleton) partition for ``scm`` — full-DAG CBO."""
    manip = {PP_NAME: PP_MANIPULATIVE, FD_NAME: FD_MANIPULATIVE,
             MC_NAME: MC_MANIPULATIVE}[scm]
    return [frozenset({v}) for v in manip] + [frozenset({"Y"})]


# ---------------------------------------------------------------------------
# Closed-form oracles
# ---------------------------------------------------------------------------

def pp_do_joint(x1: float, x2: float, lam: float = PP_LAM) -> float:
    """E[Y | do(X1=x1, X2=x2)] for ParallelParent."""
    return lam * (x1 - PP_A) ** 2 + (x2 - PP_B) ** 2


def pp_do_x1(x1: float, lam: float = PP_LAM) -> float:
    """E[Y | do(X1=x1)]: X2 free, E X2 = 0, Var X2 = PP_VAR_X."""
    return lam * (x1 - PP_A) ** 2 + (PP_B ** 2 + PP_VAR_X)


def pp_do_x2(x2: float, lam: float = PP_LAM) -> float:
    """E[Y | do(X2=x2)]: X1 free, E X1 = 0, Var X1 = PP_VAR_X."""
    return lam * (PP_A ** 2 + PP_VAR_X) + (x2 - PP_B) ** 2


def pp_gap(lam: float = PP_LAM) -> float:
    """Experiment A analytic headline gap: wrong-fine CBO's best reachable
    value (only do(X2) arms survive the A1 deletion) minus y* = 0."""
    return lam * (PP_A ** 2 + PP_VAR_X)          # = 4.25 lam


def _gauss_smooth(mean: float, var: float) -> float:
    """E[exp(-(Z - C)^2 / 2 W^2)] for Z ~ N(mean, var) (closed form)."""
    import math
    s2 = FD_W ** 2 + var
    return FD_W / math.sqrt(s2) * math.exp(-(mean - FD_C) ** 2 / (2 * s2))


def fd_do_x1(x1: float) -> float:
    """E[Y | do(X1=x1)] for FrontDoor: M ~ N(B x1, SIGMA_M^2) smooths the
    well (the front-door functional; the U path contributes E U = 0)."""
    return FD_AMP * (1.0 - _gauss_smooth(FD_B * x1, FD_SIGMA_M ** 2))


def fd_do_m(m: float) -> float:
    """E[Y | do(M=m)] = AMP (1 - exp(-(m-C)^2/2W^2)) (backdoor through X1;
    E U = 0)."""
    import math
    return FD_AMP * (1.0 - math.exp(-(m - FD_C) ** 2 / (2 * FD_W ** 2)))


def mc_do_x1(x1: float) -> float:
    """E[Y | do(X1=x1)] = (BETA x1 - C)^2 + SIGMA_2^2 for MediatedChain."""
    return (MC_BETA * x1 - MC_C) ** 2 + MC_SIGMA_2 ** 2


def mc_do_x2(x2: float) -> float:
    """E[Y | do(X2=x2)] = (x2 - C)^2 (X1 becomes irrelevant)."""
    return (x2 - MC_C) ** 2


def mc_price() -> float:
    """Price of the fixed coarse partition: V(Pi) - y*.

    The whole-cluster arm clamps X2 into the policy box, so
    V(Pi) = min over the box of (x2 - C)^2 = (box_hi - C)^2, while the fine
    optimum do(X1 = C/BETA) reaches y* = SIGMA_2^2.
    """
    return (MC_X2_BOX[1] - MC_C) ** 2 - MC_SIGMA_2 ** 2   # = 4 - 0.04 = 3.96


ORACLE: Dict[str, Dict[str, float]] = {
    PP_NAME: {
        "y_star": 0.0,
        "x_star": (PP_A, PP_B),
        "best_do_x1": PP_B ** 2 + PP_VAR_X,                # 4.25
        "best_do_x2": PP_LAM * (PP_A ** 2 + PP_VAR_X),     # 4.25 LAM
        "gap": pp_gap(),                                   # 4.25 LAM
    },
    FD_NAME: {
        "y_star": 0.0,                                     # at do(M = C)
        "m_star": FD_C,                                    # 1.0
        # best do(X1): the well smoothed by the M-mechanism noise, ~0.336
        "best_do_x1": FD_AMP * (1.0 - FD_W / (FD_W ** 2 + FD_SIGMA_M ** 2)
                                ** 0.5),
        "v_pi": 0.0,                                       # lossless partition
        # E[Y]: the well smoothed by M's marginal N(0, B^2 VarX1 + SIGMA_M^2)
        "obs_mean_y": FD_AMP * (1.0 - _gauss_smooth(
            0.0, FD_B ** 2 * (FD_SIGMA_U ** 2 + FD_SIGMA_1 ** 2)
            + FD_SIGMA_M ** 2)),                           # ~1.637
    },
    MC_NAME: {
        "y_star": MC_SIGMA_2 ** 2,                         # 0.04
        "x1_star": MC_C / MC_BETA,                         # 2.0
        "v_pi": (MC_X2_BOX[1] - MC_C) ** 2,                # 4.0
        "price": mc_price(),                               # 3.96
    },
}


# ---------------------------------------------------------------------------
# Perturbation taxonomy (paper Table 1 is emitted from this list)
# ---------------------------------------------------------------------------
# Fields:
#   id / scm / label / ops              structural edit on the ASSUMED graph
#   add_confounders / drop_confounders  latent edits on the ASSUMED graph
#   quotient_changed    does the coarse C-DAG change?
#   fine_mis_changed    does the fine (identity-partition) MIS change?
#   fine_prior_changed  does any fine arm's prior tier / estimand change?
#                       (None = not applicable: the arm itself is gone)
#   protected           predicted QCBO-coarse invariance (True iff the edit
#                       is invisible in the quotient)
Perturbation = Dict[str, object]

PERTURBATIONS: List[Perturbation] = [
    {"id": "A0", "scm": PP_NAME, "label": "correct", "ops": [],
     "quotient_changed": False, "fine_mis_changed": False,
     "fine_prior_changed": False, "protected": True},
    # Experiment A: exploration-set corruption. Quotient-redundant deletion —
    # C1->Y survives through X2->Y, but X1 leaves An(Y) at the fine level so
    # every X1-containing arm (including the uniquely optimal joint arm)
    # drops out of the fine MIS. Permanent analytic gap pp_gap().
    {"id": "A1", "scm": PP_NAME, "label": "del X1->Y (quotient-redundant)",
     "ops": [("del", "X1", "Y")],
     "quotient_changed": False, "fine_mis_changed": True,
     "fine_prior_changed": None, "protected": True},
    # Experiment B: prior corruption without arm loss. The spurious X1->X2
    # plus the (real) latent X1<->X2 forms an intra-cluster bow: do(X1) flips
    # to the uninformative tier, do(X2)'s adjustment formula changes, no arm
    # is added or removed. Recoverable through interventional data.
    {"id": "A2", "scm": PP_NAME, "label": "add X1->X2 (intra-cluster bow)",
     "ops": [("add", "X1", "X2")],
     "quotient_changed": False, "fine_mis_changed": False,
     "fine_prior_changed": True, "protected": True},
    # Negative control: a quotient-VISIBLE latent. The assumed X1<->Y crosses
    # the C1 | {Y} boundary, so the C-DAG gains C1<->Y (a cluster bow with
    # C1->Y) and the cluster arm drops to the uninformative tier. QCBO still
    # runs — membership is never gated — but invariance is NOT expected.
    {"id": "A3", "scm": PP_NAME, "label": "assume X1<->Y (quotient-visible)",
     "ops": [], "add_confounders": [("UXY", ["X1", "Y"])],
     "quotient_changed": True, "fine_mis_changed": False,
     "fine_prior_changed": True, "protected": False},
    # Experiment B (FrontDoor): prior corruption of the OPTIMAL arm without
    # arm loss. On ParallelParent the optimal (joint) arm's prior is
    # uncorruptible — every intra edit leaves the fine trajectory literally
    # unchanged (the A2 pilot showed sup|Delta| = 0) — so the visible prior
    # channel needs an SCM whose optimal arm is a singleton with a
    # non-trivial estimand. Here do(M) (backdoor through X1) is optimal and
    # do(X1) is front-door; B1 assumes a spurious intra-cluster confounder
    # X1<->M, which flips both fine arms to the uninformative tier while the
    # quotient — which drops intra-cluster bidirected edges — is unchanged.
    {"id": "B0", "scm": FD_NAME, "label": "correct", "ops": [],
     "quotient_changed": False, "fine_mis_changed": False,
     "fine_prior_changed": False, "protected": True},
    {"id": "B1", "scm": FD_NAME,
     "label": "assume X1<->M (intra-cluster confounder)",
     "ops": [], "add_confounders": [("UXM", ["X1", "M"])],
     "quotient_changed": False, "fine_mis_changed": False,
     "fine_prior_changed": True, "protected": True},
    # Experiment C: no misspecification — the price of coarsening and its
    # recovery by refinement are properties of the (correct) partition.
    {"id": "C0", "scm": MC_NAME, "label": "correct", "ops": [],
     "quotient_changed": False, "fine_mis_changed": False,
     "fine_prior_changed": False, "protected": True},
]

VARIANT_SUFFIX: Dict[str, str] = {
    "A0": "", "A1": "_NoX1Y", "A2": "_WrongX1X2", "A3": "_ConfXY",
    "B0": "", "B1": "_ConfX1M", "C0": "",
}

_SCM_SPEC = {
    PP_NAME: (PP_NODES, PP_TRUE_EDGES, PP_CONFOUNDERS, PP_MANIPULATIVE),
    FD_NAME: (FD_NODES, FD_TRUE_EDGES, FD_CONFOUNDERS, FD_MANIPULATIVE),
    MC_NAME: (MC_NODES, MC_TRUE_EDGES, MC_CONFOUNDERS, MC_MANIPULATIVE),
}


def _pert(perturbation_id: str) -> Perturbation:
    return next(p for p in PERTURBATIONS if p["id"] == perturbation_id)


def variant_name(perturbation_id: str) -> str:
    pert = _pert(perturbation_id)
    return str(pert["scm"]) + VARIANT_SUFFIX[perturbation_id]


def variant_edges(perturbation_id: str) -> List[Tuple[str, str]]:
    """Observable edge list of the ASSUMED graph for a condition."""
    pert = _pert(perturbation_id)
    _, true_edges, _, _ = _SCM_SPEC[str(pert["scm"])]
    return apply_ops(true_edges, pert["ops"])  # type: ignore[arg-type]


def variant_confounders(perturbation_id: str) -> List[Tuple[str, List[str]]]:
    """Latent-confounder list of the ASSUMED graph for a condition.

    Conditions may ADD confounders (A3 assumes a spurious X1<->Y) or drop
    them; the true SCM — and therefore the objective and the observational
    sampler — always keeps the true ``*_CONFOUNDERS`` list, so these edits
    only change the method's reasoning (the fixed-objective seam).
    """
    pert = _pert(perturbation_id)
    _, _, confounders, _ = _SCM_SPEC[str(pert["scm"])]
    dropped = set(pert.get("drop_confounders", []))  # type: ignore[arg-type]
    out = [(lat, list(vs)) for lat, vs in confounders if lat not in dropped]
    out += [(lat, list(vs))
            for lat, vs in pert.get("add_confounders", [])]  # type: ignore[union-attr]
    return out


# ---------------------------------------------------------------------------
# Registration (so QCBO can use any variant via assumed_graph_name)
# ---------------------------------------------------------------------------

def register_variants() -> List[str]:
    """Register both true graphs + every perturbation variant with
    ccbo.coarsening. Idempotent; call before constructing any CoarsenedGraph
    over a MinimalBench SCM (including inside runner worker processes)."""
    registered = []
    for pert in PERTURBATIONS:
        name = variant_name(str(pert["id"]))
        nodes, _, _, manip = _SCM_SPEC[str(pert["scm"])]
        edges = variant_edges(str(pert["id"]))
        if not is_acyclic(edges, nodes):
            raise ValueError(f"variant {name} is cyclic: {edges}")
        coarsening.register_graph(
            name, dag_edges=edges, nodes=list(nodes), hidden_nodes=[],
            manipulative_variables=list(manip),
            confounders=variant_confounders(str(pert["id"])))
        registered.append(name)
    return registered
