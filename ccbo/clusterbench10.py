"""Canonical *structural* spec for the ClusterBench10 misspecification benchmark.

This module is the single source of truth for ClusterBench10's DAG, its coarse
partition, and the family of misspecifications used in the head-to-head
misspecification stress test (see plan
``do-a-full-overview-ticklish-stonebraker.md``). It contains **structure only**
(edges / partition / perturbations); the numeric SEM coefficients live in
``scripts/generate_clusterbench10.py`` because they require Monte-Carlo tuning.

Design goals (validated by ``ccbo/tests`` and the generator's oracle pre-checks):

* ~10 observables (incl. ``Y``) + 1 latent ``U12 -> {X1, X2}`` (bidirected X1<->X2).
* Coarse partition ``C1 = {X1,X2,X3}``, ``C2 = {X4,X5}``; everything else singleton.
* The bow misspecification ``P1`` (add ``X1->X2``) sits *inside* ``C1`` -> by the
  intra-cluster invariance property the coarse C-DAG is unchanged, while at the
  finest partition the bow ``X1->X2`` + latent ``X1<->X2`` makes ``do(X1)``
  non-identifiable, deleting the best singleton arm (Tier-2, unrecoverable).
* The global optimum is ``do(C1)=do(X1,X2,X3)`` (a *union of clusters*), so the
  coarse partition is lossless (no price-of-coarsening), isolating robustness.

Perturbations are classified ``intra`` / ``inter`` w.r.t. the coarse partition:
QCBO-coarse is provably invariant to ``intra`` perturbations and *not* to
``inter`` ones (the inter-cluster control). The taxonomy is exposed as data so
the driver and tests can iterate it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ccbo import coarsening

# ---------------------------------------------------------------------------
# Ground-truth structure
# ---------------------------------------------------------------------------

NAME = "ClusterBench10"

MANIPULATIVE: List[str] = ["X1", "X2", "X3", "X4", "X5"]
NON_MANIPULATIVE: List[str] = ["M1", "M2", "M3", "M4"]
TARGET = "Y"

NODES: List[str] = MANIPULATIVE + NON_MANIPULATIVE + [TARGET]

# Latent confounders (bidirected edges in the projected ADMG):
#   U12 -> {X1, X2}  (intra-C1): makes the spurious X1->X2 edge a *bow*, so do(X1)
#                    is non-identifiable at the finest partition (intra control P1).
#   U15 -> {X1, X5}  (X1 in C1, X5 in C2): with NO directed X1->X5 in truth this is
#                    mere confounding (do(C1) stays identifiable). The inter-cluster
#                    control `Pic` ADDS the directed X1->X5, forming a *cluster-level*
#                    bow C1->C2 + C1<->C2 that deletes do(C1) at the COARSE level --
#                    the same op as P1 (add a directed edge on a confounded pair) but
#                    inter-cluster, so QCBO-coarse is NOT protected (Prop. 2 scope).
# (X5 is a root with no other C1->C2 directed edge, so truth has no cluster bow.)
CONFOUNDERS: List[Tuple[str, List[str]]] = [
    ("U12", ["X1", "X2"]),
    ("U15", ["X1", "X5"]),
]

# Observable directed edges of the true DAG.
# X1 is intentionally a root (only the latent U12 feeds it) so that do(X1) is a
# clean, strong singleton; the intra-C1 real edge is X3->X2 (the Tier-1 target),
# which does NOT flow into X1, keeping the Tier-2 gap (best surviving singleton
# vs the deleted do(X1)) large under the bow.
TRUE_EDGES: List[Tuple[str, str]] = [
    # intra-C1 real edge (the Tier-1 / P2 target)
    ("X3", "X2"),
    # C1 -> non-manipulable mediators (inter-cluster wrt the coarse partition)
    ("X1", "M2"), ("X2", "M2"), ("X3", "M1"),
    # C2 -> non-manipulable mediators. NB there is intentionally NO directed
    # C1->C2 edge in truth: the only C1-C2 link is the bidirected X1<->X5 (U15),
    # so do(C1) is identifiable at the coarse level until `Pic` adds X1->X5.
    ("X4", "M3"), ("X5", "M3"), ("X5", "M4"),
    # parents of Y
    ("X1", "Y"), ("X2", "Y"), ("X3", "Y"),
    ("M1", "Y"), ("M2", "Y"), ("M3", "Y"), ("M4", "Y"),
]

# Coarse partition (manipulable-only clusters; non-manip + Y are singletons).
COARSE_CLUSTERS: List[List[str]] = [["X1", "X2", "X3"], ["X4", "X5"]]


def _cluster_of(var: str) -> str:
    """Return a stable cluster id for ``var`` under the coarse partition."""
    for cluster in COARSE_CLUSTERS:
        if var in cluster:
            return "+".join(cluster)
    return var  # singleton (non-manipulable or target)


def edge_locus(u: str, v: str) -> str:
    """Classify an edge as ``'intra'`` or ``'inter'`` w.r.t. the coarse partition."""
    return "intra" if _cluster_of(u) == _cluster_of(v) else "inter"


# ---------------------------------------------------------------------------
# Perturbation taxonomy
# ---------------------------------------------------------------------------
# Each op is one of:
#   ("add", u, v)     add directed edge u->v        (must be absent)
#   ("del", u, v)     delete directed edge u->v     (must be present)
#   ("rev", u, v)     reverse u->v into v->u         (u->v must be present)

Perturbation = Dict[str, object]

# ``protected`` is the *predicted* invariance of QCBO-coarse, i.e. whether the
# perturbation is invisible in the quotient C-DAG. The precise condition is NOT
# simply "intra-cluster": an inter-cluster edge that is *redundant* at the
# quotient level (a parallel edge from another cluster member keeps the cluster
# edge alive) is also invisible. The three buckets:
#   * intra-cluster            -> protected (edge lives inside a cluster)
#   * inter, non-redundant     -> NOT protected (sole edge between clusters)
#   * inter, redundant         -> protected (parallel cluster edge survives)
PERTURBATIONS: List[Perturbation] = [
    {"id": "P0", "label": "correct", "ops": [], "tier": "baseline",
     "locus": "n/a", "protected": True},
    {"id": "P1", "label": "add X1->X2 (bow)", "ops": [("add", "X1", "X2")],
     "tier": "Tier-2 arm deletion", "locus": "intra", "protected": True},
    {"id": "P2", "label": "del X3->X2", "ops": [("del", "X3", "X2")],
     "tier": "Tier-1 prior bias", "locus": "intra", "protected": True},
    {"id": "P3", "label": "rev X3->X2", "ops": [("rev", "X3", "X2")],
     "tier": "Tier-1 prior bias", "locus": "intra", "protected": True},
    # Inter-cluster BOW (the headline scope control): add the directed X1->X5 on
    # the confounded pair X1<->X5 (U15). At the coarse level this is C1->C2 + C1<->C2
    # = a cluster bow -> do(C1) (the best arm) is non-identifiable -> QCBO-coarse is
    # NOT protected and loses its arm. Same op as P1 (add a directed edge on a
    # confounded pair) but inter-cluster -- the exact scope of Prop. 2.
    {"id": "Pic", "label": "add X1->X5 (inter-cluster bow)",
     "ops": [("add", "X1", "X5")], "tier": "Tier-2 cluster-arm deletion",
     "locus": "inter", "protected": False},
    # Inter-cluster, NON-redundant edge deletion: X3->M1 is the sole C1->M1 edge,
    # so deleting it changes the quotient C-DAG. Whether it shifts the trajectory
    # depends on whether it enters a used arm's identification (the nuance).
    {"id": "P5", "label": "del X3->M1 (inter, sole C1->M1)",
     "ops": [("del", "X3", "M1")], "tier": "structural",
     "locus": "inter", "protected": False},
    # Inter-cluster but REDUNDANT: X2->M2 keeps the {X1,X2,X3}->M2 cluster edge
    # alive, so the quotient is unchanged -> still protected (the nuance that
    # the protection condition is "invisible in the quotient", not "intra").
    {"id": "P6", "label": "del X1->M2 (inter, redundant)",
     "ops": [("del", "X1", "M2")], "tier": "structural",
     "locus": "inter-redundant", "protected": True},
    # The FORGOTTEN CONFOUNDER: the assumed graph omits the latent U12, i.e. the
    # bidirected X1<->X2 -- arguably the most realistic misspecification (nobody
    # adds bows; everybody misses confounders). Directed edges are untouched, so
    # no identifiability verdict flips: this is a pure PRIOR-BIAS perturbation.
    # At the finest partition do(X1)/do(X2) priors are computed as if
    # unconfounded (naive regressions on corr(X1,X2)=0.7 data -> biased); the
    # quotient drops intra-cluster bidirected edges anyway, so the coarse C-DAG
    # is unchanged -> QCBO-coarse is protected (Prop. 2, bidirected intra edit).
    {"id": "P7", "label": "omit X1<->X2 confounder",
     "ops": [], "drop_confounders": ["U12"], "tier": "prior bias (omission)",
     "locus": "intra", "protected": True},
    # Dial sweep: 1->2->3 simultaneous intra-C1 edits (QCBO-coarse Delta==0 at every severity).
    # S1 == P1 (identical ops); the drivers score it from P1's runs rather than
    # re-running the same variant (see run_misspec_fullfield.aggregate).
    {"id": "S1", "label": "sweep sev=1", "ops": [("add", "X1", "X2")],
     "tier": "sweep", "locus": "intra", "protected": True},
    {"id": "S2", "label": "sweep sev=2",
     "ops": [("add", "X1", "X2"), ("del", "X3", "X2")], "tier": "sweep",
     "locus": "intra", "protected": True},
    {"id": "S3", "label": "sweep sev=3",
     "ops": [("add", "X1", "X2"), ("del", "X3", "X2"), ("add", "X1", "X3")],
     "tier": "sweep", "locus": "intra", "protected": True},
]

# Stable suffix used for the registered graph name of each perturbation.
VARIANT_SUFFIX: Dict[str, str] = {
    "P0": "", "P1": "_WrongX1X2", "P2": "_NoX3X2", "P3": "_RevX3X2",
    "Pic": "_InterBow", "P5": "_NoX3M1", "P6": "_NoX1M2", "P7": "_NoU12",
    "S1": "_S1", "S2": "_S2", "S3": "_S3",
}


def apply_ops(edges: List[Tuple[str, str]],
              ops: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
    """Apply a sequence of structural ops to ``edges`` (returns a new list)."""
    out = list(edges)
    for op in ops:
        kind, u, v = op
        if kind == "add":
            if (u, v) in out:
                raise ValueError(f"add: edge {u}->{v} already present")
            out.append((u, v))
        elif kind == "del":
            if (u, v) not in out:
                raise ValueError(f"del: edge {u}->{v} not present")
            out.remove((u, v))
        elif kind == "rev":
            if (u, v) not in out:
                raise ValueError(f"rev: edge {u}->{v} not present")
            out.remove((u, v))
            if (v, u) in out:
                raise ValueError(f"rev: reversed edge {v}->{u} already present")
            out.append((v, u))
        else:
            raise ValueError(f"unknown op kind: {kind!r}")
    return out


def is_acyclic(edges: List[Tuple[str, str]], nodes: List[str]) -> bool:
    """Kahn-style acyclicity check (no external deps)."""
    succ = defaultdict(list)
    indeg = {n: 0 for n in nodes}
    for u, v in edges:
        succ[u].append(v)
        indeg[v] = indeg.get(v, 0) + 1
        indeg.setdefault(u, indeg.get(u, 0))
    queue = [n for n, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return seen == len(indeg)


def variant_edges(perturbation_id: str) -> List[Tuple[str, str]]:
    """Return the observable edge list for a given perturbation id."""
    pert = next(p for p in PERTURBATIONS if p["id"] == perturbation_id)
    return apply_ops(TRUE_EDGES, pert["ops"])  # type: ignore[arg-type]


def variant_confounders(perturbation_id: str) -> List[Tuple[str, List[str]]]:
    """Return the latent-confounder list for a given perturbation id.

    Perturbations may omit confounders from the *assumed* structure via a
    ``drop_confounders`` field (e.g. P7 forgets ``U12``); the true SCM --- and
    therefore the objective and the observational sampler --- always keeps the
    full ``CONFOUNDERS`` list, so this only starves the method's reasoning.
    """
    pert = next(p for p in PERTURBATIONS if p["id"] == perturbation_id)
    dropped = set(pert.get("drop_confounders", []))  # type: ignore[arg-type]
    return [(lat, list(vs)) for lat, vs in CONFOUNDERS if lat not in dropped]


def variant_name(perturbation_id: str, prefix: str = NAME) -> str:
    return prefix + VARIANT_SUFFIX[perturbation_id]


# ---------------------------------------------------------------------------
# Registration (so QCBO can use any variant via assumed_graph_name)
# ---------------------------------------------------------------------------

def register_variants(prefix: str = NAME) -> List[str]:
    """Register the true graph + every perturbation variant with ccbo.coarsening.

    Returns the list of registered names. Call this before constructing a
    CoarsenedGraph whose ``assumed_graph_name`` is one of the variants. The true
    dataset is also (re)registered from its SEM JSON by the benchmark adapter;
    we register it here too so the structural tests can run without the dataset
    files present.
    """
    registered = []
    for pert in PERTURBATIONS:
        name = variant_name(pert["id"], prefix)  # type: ignore[arg-type]
        edges = variant_edges(pert["id"])  # type: ignore[arg-type]
        if not is_acyclic(edges, NODES):
            raise ValueError(f"variant {name} is cyclic: {edges}")
        coarsening.register_graph(
            name, dag_edges=edges, nodes=list(NODES), hidden_nodes=[],
            manipulative_variables=list(MANIPULATIVE),
            confounders=variant_confounders(pert["id"]))  # type: ignore[arg-type]
        registered.append(name)
    return registered
