"""Quotient transformation of an MCBO env_profile (the heart of QMCBO v1).

MCBO's ``GaussianProcessNetwork`` fits one GP per node with features =
(actions on the node) + (values of the node's *parents*), and its posterior
samplers walk the DAG topologically. Nothing in that machinery requires a
node's parents to be its fine-DAG parents — the graph enters only through
``dag.get_parent_nodes(k)`` / ``get_root_nodes()``.

QMCBO v1 therefore needs **no model subclassing**: it is exactly the stock
MCBO run on a *quotient-transformed* env_profile in which

    parents_Q(i) = union of the members of the parent clusters of cluster(i),

where the cluster-level parent relation is the quotient of the fine DAG
(C' -> C iff some fine edge u -> v has u in C', v in C, C' != C). Each node's
mechanism GP then conditions on the *cluster-level* inputs of its cluster —
per-coordinate cluster mechanisms, composed along the C-DAG. Consequences:

* intra-cluster edges are dropped (invisible: the invariance property);
* inter-cluster edges are coarsened to whole-cluster dependence (so
  quotient-redundant edges are invisible too, mirroring Prop. 2's scope);
* ``valid_targets`` are lifted to whole-cluster do() patterns (the QCBO-coarse
  arm space); the ground-truth env and its input layout are untouched.

Within-cluster residual correlation (induced by intra-cluster edges) is NOT
modeled — coordinates are sampled independently given cluster inputs. This is
the documented v1 approximation; a joint multi-output cluster GP is the
upgrade path.

Causal sufficiency is assumed throughout (MCBO's own assumption): no
bidirected edges / hidden confounders.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_MCBO_ROOT = os.path.join(_REPO_ROOT, 'third_party', 'mcbo')


def ensure_mcbo_on_path():
    """Make ``import mcbo`` and ``import functions`` (the env module) work.

    The MCBO authors' stack (github.com/ssethz/mcbo) is fetched by
    ``scripts/fetch_mcbo.sh`` into ``third_party/mcbo``.
    """
    if not os.path.isdir(_MCBO_ROOT):
        raise FileNotFoundError(
            f"MCBO not found at {_MCBO_ROOT}; run scripts/fetch_mcbo.sh")
    for p in (_MCBO_ROOT, os.path.join(_MCBO_ROOT, 'scripts')):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Quotient construction
# ---------------------------------------------------------------------------

def _validate_partition(partition: Sequence[Sequence[int]], n_nodes: int):
    seen: Dict[int, int] = {}
    for ci, cluster in enumerate(partition):
        if not cluster:
            raise ValueError("empty cluster in partition")
        for v in cluster:
            if v in seen:
                raise ValueError(f"node {v} appears in two clusters")
            if not (0 <= v < n_nodes):
                raise ValueError(f"node {v} out of range")
            seen[v] = ci
    if len(seen) != n_nodes:
        missing = sorted(set(range(n_nodes)) - set(seen))
        raise ValueError(f"partition misses nodes {missing}")
    return seen  # node -> cluster index


def _cluster_parents(parent_nodes: Sequence[Sequence[int]],
                     node_to_cluster: Dict[int, int],
                     n_clusters: int) -> List[set]:
    """Quotient adjacency: cluster-level parent sets from fine parents."""
    cpar: List[set] = [set() for _ in range(n_clusters)]
    for v, pars in enumerate(parent_nodes):
        cv = node_to_cluster[v]
        for u in pars:
            cu = node_to_cluster[u]
            if cu != cv:
                cpar[cv].add(cu)
    return cpar


def _assert_acyclic(cpar: List[set]):
    """Kahn's algorithm on the cluster graph; raise if the quotient is cyclic
    (the Prop.-3 validity re-check, model-based edition)."""
    n = len(cpar)
    indeg = [len(p) for p in cpar]
    children: List[set] = [set() for _ in range(n)]
    for c, pars in enumerate(cpar):
        for p in pars:
            children[p].add(c)
    queue = [c for c in range(n) if indeg[c] == 0]
    seen = 0
    while queue:
        c = queue.pop()
        seen += 1
        for ch in children[c]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                queue.append(ch)
    if seen != n:
        raise ValueError("partition induces a cyclic quotient (invalid C-DAG)")


def quotient_parent_nodes(parent_nodes: Sequence[Sequence[int]],
                          partition: Sequence[Sequence[int]]) -> List[List[int]]:
    """The quotient view of the fine ``parent_nodes`` adjacency.

    parents_Q(i) = sorted union of members of cluster(i)'s parent clusters.
    Raises ValueError on an invalid (cyclic-quotient) partition.
    """
    n_nodes = len(parent_nodes)
    node_to_cluster = _validate_partition(partition, n_nodes)
    cpar = _cluster_parents(parent_nodes, node_to_cluster, len(partition))
    _assert_acyclic(cpar)
    out: List[List[int]] = []
    for v in range(n_nodes):
        cv = node_to_cluster[v]
        members: List[int] = []
        for cu in cpar[cv]:
            members.extend(partition[cu])
        out.append(sorted(members))
    return out


def lift_targets(valid_targets, partition: Sequence[Sequence[int]]):
    """Lift the env's do() patterns to whole-cluster patterns.

    Each fine target becomes the union of the clusters it touches; duplicates
    are removed (order-preserving). The all-zeros target survives unchanged.
    Mirrors QCBO-coarse's arm space: do() acts on full clusters only.
    """
    import torch
    n_nodes = int(valid_targets[0].shape[0])
    node_to_cluster = _validate_partition(partition, n_nodes)
    lifted, seen = [], set()
    for t in valid_targets:
        q = torch.zeros_like(t)
        for v in range(n_nodes):
            if int(t[v]) == 1:
                for m in partition[node_to_cluster[v]]:
                    q[m] = 1
        key = tuple(int(x) for x in q)
        if key not in seen:
            seen.add(key)
            lifted.append(q)
    return lifted


def quotient_env_profile(env_profile: dict,
                         partition: Sequence[Sequence[int]]) -> dict:
    """The QMCBO model view: same env, quotient graph + lifted targets.

    Everything else (input layout, do_map, noise dists, active_input_indices)
    is untouched, so the stock ``mcbo_trial`` runs unchanged on the result.
    """
    ensure_mcbo_on_path()
    from mcbo.utils.dag import DAG

    fine_parents = [list(env_profile["dag"].get_parent_nodes(k))
                    for k in range(env_profile["dag"].get_n_nodes())]
    qparents = quotient_parent_nodes(fine_parents, partition)
    out = dict(env_profile)
    out["dag"] = DAG(qparents)
    out["valid_targets"] = lift_targets(env_profile["valid_targets"], partition)
    return out


# ---------------------------------------------------------------------------
# Misspecification injection (fixed-objective seam)
# ---------------------------------------------------------------------------

def perturb_parent_nodes(parent_nodes: Sequence[Sequence[int]],
                         ops: Sequence[tuple]) -> List[List[int]]:
    """Apply ('add'|'del'|'rev', u, v) edge ops to a parent-list adjacency.

    This perturbs only the *model's view* of the graph; the environment's
    ``evaluate`` (ground truth) is never touched — the same fixed-objective
    seam as scripts/misspec_inject.py.
    """
    pars = [sorted(p) for p in parent_nodes]
    for kind, u, v in ops:
        if kind == "add":
            if u in pars[v]:
                raise ValueError(f"add: edge {u}->{v} already present")
            pars[v] = sorted(pars[v] + [u])
        elif kind == "del":
            if u not in pars[v]:
                raise ValueError(f"del: edge {u}->{v} not present")
            pars[v] = [p for p in pars[v] if p != u]
        elif kind == "rev":
            if u not in pars[v]:
                raise ValueError(f"rev: edge {u}->{v} not present")
            if v in pars[u]:
                raise ValueError(f"rev: edge {v}->{u} already present")
            pars[v] = [p for p in pars[v] if p != u]
            pars[u] = sorted(pars[u] + [v])
        else:
            raise ValueError(f"unknown op {kind!r}")
    return pars
