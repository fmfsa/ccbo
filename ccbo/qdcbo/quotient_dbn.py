"""Quotient transformation of a DCBO setup (QDCBO's structural core).

DCBO (Aglietti et al. 2021) works on a temporal MultiDiGraph with nodes
``<var>_<t>`` and splits its edges into within-slice *emission* arcs and
cross-slice *transition* arcs.  Every model component (arc fitting, SEM-hat
dispatch, parent resolution) reads that graph through ``<var>_<t>`` string
parsing.  This module builds the *quotient* view of such a setup under a
partition of the manipulable base variables into clusters:

* quotient temporal graph: one node ``<C>_<t>`` per cluster per slice, where
  multi-member clusters get fresh names (``C1``, ``C2``, ...) and singleton
  clusters (including the non-manipulable / target variables, which stay
  atomic) KEEP their base-variable name — this is what makes the finest
  partition (all singletons) produce a graph identical to the fine one;
* within-slice quotient (emission) edges: ``C_t -> D_t`` iff some fine edge
  ``u_t -> v_t`` has u in C, v in D, C != D — intra-cluster edges are dropped
  (the invariance property);
* cross-slice quotient (transition) edges: ``C_t -> D_{t+1}`` analogously.
  Temporal identity edges per cluster (``C_t -> C_{t+1}``) arise as the
  quotient of the members' own self-transitions, which every DCBO example
  setup has;
* exploration sets are lifted to whole-cluster patterns *in member
  coordinates* (a fine ES becomes the ordered tuple of all members of every
  cluster it touches, duplicates removed), so DCBO's blanket / target-function
  / parameter-space machinery — which assumes scalar base variables — runs
  unchanged; the intervention domain stays keyed by the member coordinates;
* observational samples stay per-member ``{base var: (n, T)}``; the spec
  exposes the cluster membership metadata the joint mechanism fits consume.

Misspecification seam: :func:`perturb_temporal_edges` applies within-slice
edge edits to the ASSUMED graph only (the SEM / objective are never touched),
mirroring ``ccbo.qmcbo.quotient.perturb_parent_nodes``.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_DCBO_ROOT = os.path.join(_REPO_ROOT, 'third_party', 'DCBO')


def ensure_dcbo_on_path():
    """Make ``import dcbo`` resolve to the authors' stack
    (github.com/neildhir/DCBO, fetched by ``scripts/fetch_dcbo.sh`` into
    ``third_party/DCBO``). Same bootstrap pattern as
    scripts/ceo_adapter.ensure_ceo.

    No monkeypatches are needed: unlike CEO, DCBO's ``fit_gp`` uses a plain
    RBF kernel with no ``InverseGamma.from_EV`` prior (checked against
    dcbo/utils/gp_utils.py), and its interventional-GP prior is a
    ``priors.Gamma(a, b)`` which is implemented in GPy 1.13.
    """
    if not os.path.isdir(_DCBO_ROOT):
        raise FileNotFoundError(
            f"DCBO not found at {_DCBO_ROOT}; run scripts/fetch_dcbo.sh")
    if _DCBO_ROOT not in sys.path:
        sys.path.insert(0, _DCBO_ROOT)


def make_temporal_graph(start_time, stop_time, topology, nodes,
                        target_node=None):
    """Authors' ``make_graphical_model`` + their DOT→networkx conversion.

    Upstream ``make_graphical_model`` returns a DOT string (or a graphviz
    ``Source``), while every DCBO consumer (``Root`` asserts it) needs a
    ``MultiDiGraph``.  The authors convert in their examples via
    ``nx_agraph.from_agraph(pygraphviz.AGraph(dot.source))``
    (dcbo/examples/example_setups.py); we reproduce exactly that call so
    the graph object matches theirs node-order for node-order.
    """
    ensure_dcbo_on_path()
    import pygraphviz
    from networkx.drawing import nx_agraph
    from dcbo.utils.dag_utils.graph_functions import make_graphical_model

    dot = make_graphical_model(start_time, stop_time, topology=topology,
                               nodes=list(nodes), target_node=target_node)
    source = dot if isinstance(dot, str) else dot.source
    return nx_agraph.from_agraph(pygraphviz.AGraph(source))


def _split_node(node: str) -> Tuple[str, int]:
    var, t = node.rsplit("_", 1)
    return var, int(t)


class QuotientDBNSpec:
    """Structural quotient of a DCBO temporal graph under a partition.

    Every graph-dependent question QDCBO asks (parents, dispatch, member
    metadata) is answered from this object, which was built from the quotient
    graph alone — intra-cluster fine structure is unrecoverable from it.

    Attributes
    ----------
    T : int
        Number of time slices.
    base_order : list[str]
        Fine base variables in slice topological (insertion) order.
    clusters : OrderedDict[str, tuple[str, ...]]
        Cluster name -> member base variables, in quotient topological order.
    var_to_cluster : dict[str, str]
        Base variable -> cluster name (singletons map to themselves).
    G_quotient : networkx.MultiDiGraph
        The quotient temporal graph (nodes ``<C>_<t>`` inserted slice-major
        in cluster topological order — DCBO's SEM-hat machinery assumes
        causally ordered node insertion).
    G_member_view : networkx.MultiDiGraph
        The quotient graph re-expressed in member coordinates (every
        inter-cluster quotient edge expanded all-to-all over members; no
        intra-cluster edges).  Used only for DCBO's environment-side blanket
        bookkeeping (``assign_blanket``'s successor fill-in); at the finest
        partition it equals the fine graph.
    """

    def __init__(self, G_fine, partition: Sequence[Sequence[str]],
                 base_target_variable: str = "Y"):
        import networkx as nx

        nodes = list(G_fine.nodes())
        slice_ids = sorted({_split_node(n)[1] for n in nodes})
        assert slice_ids == list(range(len(slice_ids))), slice_ids
        self.T = len(slice_ids)

        # Base variables in first-appearance (slice-topological) order.
        base_order: List[str] = []
        for n in nodes:
            v, _ = _split_node(n)
            if v not in base_order:
                base_order.append(v)
        self.base_order = base_order
        self.base_target_variable = base_target_variable

        # ---- validate the partition, auto-complete singletons -------------
        var_to_cluster: Dict[str, str] = {}
        multi: List[Tuple[str, Tuple[str, ...]]] = []
        n_multi = 0
        for cluster in partition:
            members = tuple(cluster)
            if not members:
                raise ValueError("empty cluster in partition")
            for v in members:
                if v not in base_order:
                    raise ValueError(f"unknown variable {v!r} in partition")
                if v in var_to_cluster:
                    raise ValueError(f"variable {v!r} in two clusters")
                if v == base_target_variable:
                    raise ValueError("target variable cannot be clustered")
            if len(members) == 1:
                name = members[0]          # singleton keeps its own name
            else:
                n_multi += 1
                name = f"C{n_multi}"
                if name in base_order:
                    raise ValueError(f"cluster name {name} collides with a "
                                     "base variable; rename your variables")
            for v in members:
                var_to_cluster[v] = name
            multi.append((name, members))
        # Uncovered variables (incl. target / non-manipulables) stay atomic.
        cluster_members: "OrderedDict[str, Tuple[str, ...]]" = OrderedDict()
        for name, members in multi:
            cluster_members[name] = members
        for v in base_order:
            if v not in var_to_cluster:
                var_to_cluster[v] = v
                cluster_members[v] = (v,)
        self.var_to_cluster = var_to_cluster

        # ---- quotient edge sets, per slice ---------------------------------
        # emit_edges[t] : set of (C, D) within slice t
        # trans_edges[t]: set of (C, D) from slice t-1 to slice t (t >= 1)
        emit_edges = [set() for _ in range(self.T)]
        trans_edges = [set() for _ in range(self.T)]
        for u, v in G_fine.edges():
            uv, ut = _split_node(u)
            vv, vt = _split_node(v)
            cu, cv = var_to_cluster[uv], var_to_cluster[vv]
            if ut == vt:
                if cu != cv:               # intra-cluster edges vanish here
                    emit_edges[ut].add((cu, cv))
            else:
                assert vt == ut + 1, (u, v)
                trans_edges[vt].add((cu, cv))

        # ---- cluster topological order (within-slice DAG, union over t) ----
        names = list(cluster_members)      # first-appearance order
        pos = {c: i for i, c in enumerate(names)}
        within = set()
        for t in range(self.T):
            within |= emit_edges[t]
        indeg = {c: 0 for c in names}
        children: Dict[str, List[str]] = {c: [] for c in names}
        for a, b in within:
            indeg[b] += 1
            children[a].append(b)
        ready = sorted([c for c in names if indeg[c] == 0], key=pos.get)
        order: List[str] = []
        while ready:
            c = ready.pop(0)
            order.append(c)
            for ch in sorted(set(children[c]), key=pos.get):
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    ready.append(ch)
            ready.sort(key=pos.get)
        if len(order) != len(names):
            raise ValueError("partition induces a cyclic within-slice "
                             "quotient (invalid C-DAG)")
        self.clusters: "OrderedDict[str, Tuple[str, ...]]" = OrderedDict(
            (c, cluster_members[c]) for c in order)
        self.cluster_order = order

        # ---- parent maps ----------------------------------------------------
        # emit_cluster_parents[t][C]  : tuple of clusters with C'_t -> C_t
        # trans_cluster_parents[t][C] : tuple of clusters with C'_{t-1} -> C_t
        self.emit_cluster_parents = [
            {C: tuple(P for P in order if (P, C) in emit_edges[t])
             for C in order} for t in range(self.T)]
        self.trans_cluster_parents = [
            {C: tuple(P for P in order if (P, C) in trans_edges[t])
             for C in order} for t in range(self.T)]

        # ---- quotient graph (slice-major, causally ordered insertion) ------
        Gq = nx.MultiDiGraph()
        for t in range(self.T):
            for C in order:
                Gq.add_node(f"{C}_{t}")
        for t in range(self.T):
            for (a, b) in sorted(emit_edges[t], key=lambda e: (pos[e[0]],
                                                               pos[e[1]])):
                Gq.add_edge(f"{a}_{t}", f"{b}_{t}")
        for t in range(1, self.T):
            for (a, b) in sorted(trans_edges[t], key=lambda e: (pos[e[0]],
                                                                pos[e[1]])):
                Gq.add_edge(f"{a}_{t-1}", f"{b}_{t}")
        Gq.T = self.T
        self.G_quotient = Gq

        # ---- member-coordinate view (environment-side bookkeeping only) ----
        Gm = nx.MultiDiGraph()
        for t in range(self.T):
            for v in base_order:
                Gm.add_node(f"{v}_{t}")
        for t in range(self.T):
            for (a, b) in sorted(emit_edges[t], key=lambda e: (pos[e[0]],
                                                               pos[e[1]])):
                for mu in self.clusters[a]:
                    for mv in self.clusters[b]:
                        Gm.add_edge(f"{mu}_{t}", f"{mv}_{t}")
        for t in range(1, self.T):
            for (a, b) in sorted(trans_edges[t], key=lambda e: (pos[e[0]],
                                                                pos[e[1]])):
                for mu in self.clusters[a]:
                    for mv in self.clusters[b]:
                        Gm.add_edge(f"{mu}_{t-1}", f"{mv}_{t}")
        Gm.T = self.T
        self.G_member_view = Gm

    # ------------------------------------------------------------------
    # Membership / parent resolution (the ONLY structural oracle QDCBO has)
    # ------------------------------------------------------------------
    def cluster_of(self, name: str) -> str:
        """Cluster of a base variable; cluster names pass through."""
        if name in self.clusters:
            return name
        return self.var_to_cluster[name]

    def members(self, C: str) -> Tuple[str, ...]:
        return self.clusters[self.cluster_of(C)]

    def member_nodes(self, cluster_seq: Sequence[str], t: int) -> tuple:
        """Canonical member-node tuple ``(<m>_<t>, ...)`` for a sequence of
        clusters — the key/column order every mechanism fit and prediction
        uses (clusters in quotient topological order, members in declaration
        order)."""
        return tuple(f"{m}_{t}" for C in cluster_seq
                     for m in self.clusters[C])

    def emit_input_nodes(self, C: str, t: int) -> tuple:
        """Within-slice inputs of cluster C at slice t, as member nodes."""
        return self.member_nodes(
            self.emit_cluster_parents[t][self.cluster_of(C)], t)

    def trans_input_nodes(self, C: str, t: int) -> tuple:
        """Cross-slice inputs of cluster C at slice t, as member nodes at
        slice t-1 (empty for t == 0)."""
        if t == 0:
            return ()
        return self.member_nodes(
            self.trans_cluster_parents[t][self.cluster_of(C)], t - 1)

    # ------------------------------------------------------------------
    # Lifting (arms and domains)
    # ------------------------------------------------------------------
    def lift_exploration_sets(self, exploration_sets: Sequence[tuple]) -> list:
        """Fine ES -> whole-cluster patterns in member coordinates.

        Each ES becomes the ordered tuple of all members of every cluster it
        touches (clusters in quotient topological order); duplicates are
        removed order-preservingly.  At the finest partition this is the
        identity map."""
        lifted, seen = [], set()
        for es in exploration_sets:
            touched = []
            for v in es:
                c = self.cluster_of(v)
                if c not in touched:
                    touched.append(c)
            touched.sort(key=self.cluster_order.index)
            q = tuple(m for C in touched for m in self.clusters[C])
            if q not in seen:
                seen.add(q)
                lifted.append(q)
        return lifted

    def lift_intervention_domain(self, intervention_domain: dict) -> dict:
        """The domain stays in member coordinates; this just checks coverage
        for every member of every lifted cluster arm."""
        for C, members in self.clusters.items():
            if len(members) > 1:
                missing = [m for m in members if m not in intervention_domain]
                if missing:
                    raise ValueError(
                        f"cluster {C} members {missing} lack domain entries")
        return dict(intervention_domain)


def build_quotient_dbn(G_fine, partition: Sequence[Sequence[str]],
                       base_target_variable: str = "Y") -> QuotientDBNSpec:
    """Convenience constructor (see :class:`QuotientDBNSpec`)."""
    return QuotientDBNSpec(G_fine, partition, base_target_variable)


# ---------------------------------------------------------------------------
# Misspecification injection (fixed-objective seam)
# ---------------------------------------------------------------------------

def perturb_temporal_edges(G_fine, ops: Sequence[tuple]):
    """Apply ``('add'|'del'|'rev', u, v)`` base-variable edge ops within every
    time slice of a copy of the ASSUMED temporal graph.

    Only the model's view of the DAG changes — the SEM / objective side never
    reads the perturbed graph.  Node order (which DCBO's SEM-hat machinery
    treats as causal order) is preserved by ``MultiDiGraph.copy()``.
    """
    H = G_fine.copy()
    if hasattr(G_fine, "T"):
        H.T = G_fine.T
    T = len({_split_node(n)[1] for n in H.nodes()})
    for kind, u, v in ops:
        for t in range(T):
            a, b = f"{u}_{t}", f"{v}_{t}"
            if kind == "add":
                if H.has_edge(a, b):
                    raise ValueError(f"add: edge {a}->{b} already present")
                H.add_edge(a, b)
            elif kind == "del":
                if not H.has_edge(a, b):
                    raise ValueError(f"del: edge {a}->{b} not present")
                H.remove_edge(a, b)
            elif kind == "rev":
                if not H.has_edge(a, b):
                    raise ValueError(f"rev: edge {a}->{b} not present")
                if H.has_edge(b, a):
                    raise ValueError(f"rev: edge {b}->{a} already present")
                H.remove_edge(a, b)
                H.add_edge(b, a)
            else:
                raise ValueError(f"unknown op {kind!r}")
    return H
