"""
Enumerate all valid coarsenings of a causal DAG.

Two vintages of API coexist during the Lee-2019 refactor:

* **Legacy (deprecated).** ``build_coarsened_dag`` / ``enumerate_valid_coarsenings``
  partition the full observable vertex set, freely mixing manipulable and
  non-manipulable variables. Cluster interventions then silently drop non-
  manipulable members. Kept so existing callers keep working while we migrate.

* **Lee-2019 (current).** ``latent_project`` projects non-manipulable +
  hidden nodes out of the DAG, yielding an ADMG over ``M ∪ {Y}``.
  ``enumerate_valid_coarsenings_manip`` then partitions ``M`` only, and each
  partition induces a coarsened ADMG whose bidirected edges absorb the
  projected-out confounding structure.  ``compute_POMIS`` (Lee-Bareinboim
  2018 Thm 6) runs on that ADMG and replaces structural MIS as the default
  exploration set.
"""

import itertools
from collections import OrderedDict

import networkx as nx


def _generate_partitions(elements):
    """Generate all set partitions of elements (Bell partitions)."""
    elements = list(elements)
    if len(elements) == 0:
        yield []
        return
    if len(elements) == 1:
        yield [frozenset(elements)]
        return

    first = elements[0]
    rest = elements[1:]

    for partition in _generate_partitions(rest):
        # Option 1: first element as a new singleton part
        yield [frozenset([first])] + partition
        # Option 2: first element added to each existing part
        for i, part in enumerate(partition):
            new_partition = list(partition)
            new_partition[i] = part | frozenset([first])
            yield new_partition


def build_coarsened_dag(partition, dag, hidden_confounders=None):
    """
    Build the coarsened DAG (C-DAG) from a partition and original DAG.

    When ``hidden_confounders`` is provided, latent confounder nodes are
    added for every confounder whose affected variables span more than one
    cluster.  These appear as frozenset nodes whose sole element starts
    with ``'__'`` (e.g. ``frozenset(['__U1__'])``).  They have only
    outgoing edges, so they never introduce cycles.

    Parameters
    ----------
    partition : list of frozenset
        Partition of the node set.
    dag : nx.DiGraph
        Original DAG (observable edges only).
    hidden_confounders : list of (str, list[str]), optional
        Each entry is ``(latent_name, [affected_var, ...])``.

    Returns
    -------
    nx.DiGraph
        Quotient DAG where each observed node is a frozenset (partition
        element) and each latent confounder is a frozenset like
        ``frozenset(['__U1__'])``.
    """
    node_to_part = {}
    for part in partition:
        for node in part:
            node_to_part[node] = part

    coarsened = nx.DiGraph()
    for part in partition:
        coarsened.add_node(part)

    for u, v in dag.edges:
        if u in node_to_part and v in node_to_part:
            part_u = node_to_part[u]
            part_v = node_to_part[v]
            if part_u != part_v:
                coarsened.add_edge(part_u, part_v)

    # Add latent confounders that span different clusters
    if hidden_confounders:
        for latent_name, affected_vars in hidden_confounders:
            affected_clusters = set()
            for v in affected_vars:
                if v in node_to_part:
                    affected_clusters.add(node_to_part[v])
            # Only add if the confounder spans ≥2 distinct clusters
            if len(affected_clusters) > 1:
                latent_node = frozenset([f'__{latent_name}__'])
                coarsened.add_node(latent_node)
                for c in affected_clusters:
                    coarsened.add_edge(latent_node, c)

    return coarsened


def is_latent_cdag_node(node):
    """Check if a C-DAG node is a latent confounder (vs. an observed cluster)."""
    return isinstance(node, frozenset) and any(v.startswith('__') for v in node)


def get_observed_cdag_nodes(cdag):
    """Return the set of observed (non-latent) nodes in a C-DAG."""
    return {n for n in cdag.nodes if not is_latent_cdag_node(n)}


def enumerate_valid_coarsenings(dag_edges, nodes, target='Y', hidden_nodes=None,
                                hidden_confounders=None):
    """
    Enumerate all valid coarsenings of a DAG.

    A coarsening is valid if and only if the induced coarsened graph is a DAG
    (Definition 2.1, Madaleno et al. 2026).

    Parameters
    ----------
    dag_edges : list of (str, str)
        Directed edges among observable nodes.
    nodes : list of str
        All observable node names.
    target : str
        Target variable (always kept as a singleton partition element).
    hidden_nodes : list of str, optional
        Hidden/latent nodes to exclude from the partition.
    hidden_confounders : list of (str, list[str]), optional
        Latent confounders for bidirected edges in the C-DAG.

    Returns
    -------
    list of dict
        Each dict has:
        - 'partition': list of frozenset (the partition of non-target nodes + {target})
        - 'coarsened_dag': nx.DiGraph (the coarsened DAG, with latent nodes if applicable)
        - 'num_parts': int (number of partition elements, excluding latent nodes)
    """
    if hidden_nodes is None:
        hidden_nodes = set()
    else:
        hidden_nodes = set(hidden_nodes)

    # Build DAG from edges (observable nodes only)
    dag = nx.DiGraph()
    observable = set(nodes) - hidden_nodes
    dag.add_nodes_from(observable)
    for u, v in dag_edges:
        if u in observable and v in observable:
            dag.add_edge(u, v)

    # Nodes to partition (exclude target)
    nodes_to_partition = sorted(observable - {target})

    # Generate and filter partitions
    valid_coarsenings = []
    for partition in _generate_partitions(nodes_to_partition):
        # Add target as singleton
        full_partition = partition + [frozenset([target])]

        # Build coarsened DAG with latent confounders and verify acyclicity.
        # (Latent nodes only have outgoing edges, so they never cause cycles,
        # but we include them so the stored C-DAG is complete.)
        coarsened = build_coarsened_dag(full_partition, dag, hidden_confounders)
        if not nx.is_directed_acyclic_graph(coarsened):
            continue

        valid_coarsenings.append({
            'partition': full_partition,
            'coarsened_dag': coarsened,
            'num_parts': len(full_partition),
        })

    # Sort by number of parts (finest first)
    valid_coarsenings.sort(key=lambda x: -x['num_parts'])

    return valid_coarsenings


# ---------------------------------------------------------------------------
# MIS / POMIS on the coarsened DAG
# ---------------------------------------------------------------------------

def _is_identifiable_from_cdag(coarsened_dag, intervention_nodes, target_node):
    """
    Check if do(intervention_nodes) is identifiable from the C-DAG.

    Uses backdoor / frontdoor / g-computation identification on the string
    representation of the C-DAG.  Only observed cluster nodes (not latent
    confounder nodes) are passed as ``observed`` to the identification
    routine, so latent confounders correctly induce unblockable backdoor
    paths.

    Returns True iff at least one identification strategy succeeds.
    """
    from .adjustment import identify_adjustment, cdag_to_string_dag

    string_dag, node_map = cdag_to_string_dag(coarsened_dag)
    S_str = [node_map[n] for n in intervention_nodes if n in node_map]
    Y_str = node_map.get(target_node, '')
    if not S_str or not Y_str:
        return False

    # Only observed (non-latent) cluster nodes
    observed_str = {node_map[n] for n in coarsened_dag.nodes
                    if not is_latent_cdag_node(n)}
    try:
        ident = identify_adjustment(string_dag, S_str, Y_str, observed_str)
    except (ValueError, KeyError, AssertionError, nx.NetworkXError) as e:
        import warnings
        warnings.warn(
            f"C-DAG identification crashed for S={S_str}, Y={Y_str} "
            f"({type(e).__name__}: {e}); treating as non-identifiable.",
            RuntimeWarning, stacklevel=2)
        return False
    return ident['method'] != 'none'


def compute_MIS(coarsened_dag, manipulative_nodes, target_node):
    """
    Compute Minimal Intervention Sets (MIS) on a coarsened DAG.

    A set S of manipulative nodes is in MIS if every node in S has a
    directed path to the target that is not intercepted by other nodes in S.
    This is a purely structural criterion — it does NOT check identifiability.

    Identifiability is handled separately: ``get_all_do()`` in
    ``CoarsenedGraph`` uses C-DAG identification where possible and falls
    back to an uninformative GP prior for non-identifiable sets. This way
    the BO loop can still explore non-identifiable interventions and learn
    their effects from interventional data.

    Parameters
    ----------
    coarsened_dag : nx.DiGraph
    manipulative_nodes : list of frozenset   (nodes of the coarsened DAG)
    target_node : frozenset                   ({Y})

    Returns
    -------
    list of list of frozenset
        Each element is a structurally valid intervention set.
    """
    def has_unblocked_path(node, S_others, dag):
        """Check if `node` has a directed path to target not passing through S_others."""
        visited = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current == target_node:
                return True
            if current in visited:
                continue
            visited.add(current)
            for succ in dag.successors(current):
                if succ not in S_others or succ == target_node:
                    stack.append(succ)
        return False

    # Check which nodes can reach target at all
    reachable = set()
    for node in manipulative_nodes:
        if nx.has_path(coarsened_dag, node, target_node):
            reachable.add(node)

    manip_list = [n for n in manipulative_nodes if n in reachable]

    # Enumerate all non-empty subsets; keep structurally minimal sets.
    valid_sets = []
    for size in range(1, len(manip_list) + 1):
        for combo in itertools.combinations(manip_list, size):
            combo_set = set(combo)
            all_unblocked = True
            for node in combo:
                others = combo_set - {node}
                if not has_unblocked_path(node, others, coarsened_dag):
                    all_unblocked = False
                    break
            if all_unblocked:
                valid_sets.append(list(combo))

    return valid_sets



# ---------------------------------------------------------------------------
# Graph metadata
# ---------------------------------------------------------------------------

def get_hidden_confounders(graph_name):
    """
    Return hidden confounder structure for a named graph.

    Each entry is ``(latent_name, [affected_observable_var, ...])``.
    Used by ``build_coarsened_dag`` to add bidirected edges (latent common
    causes) to the C-DAG when a confounder's affected variables span
    multiple clusters.

    The confounder information is the same for all structural variants of
    a graph (e.g. ``CompleteGraph``, ``CompleteGraph_NoBC``).
    """
    if graph_name in ('CompleteGraph', 'CompleteGraph_NoBC', 'CompleteGraph_NoCD'):
        return [
            ('U1', ['A', 'Y']),
            ('U2', ['B', 'Y']),
        ]
    elif graph_name == 'ToyGraph':
        return [
            ('U', ['X', 'Y']),
        ]
    elif graph_name in ('LightTunnel', 'LightTunnel_WrongRG'):
        # U_color is the latent confounder shared by R and G in the SEM;
        # it produces observational corr(R, G) > 0 which is what makes
        # the WrongRG misspec actually affect CBO's adjustment.
        return [
            ('U_color', ['R', 'G']),
        ]
    else:
        return []


def get_dag_edges_from_sem(graph_name):
    """
    Return the observable DAG edges and nodes for each known graph.

    Parameters
    ----------
    graph_name : str
        'ToyGraph' or 'CompleteGraph'

    Returns
    -------
    dag_edges : list of (str, str)
    nodes : list of str
    hidden_nodes : list of str
    manipulative_variables : list of str
    """
    if graph_name == 'ToyGraph':
        dag_edges = [('X', 'Z'), ('Z', 'Y')]
        nodes = ['X', 'Z', 'Y']
        hidden_nodes = []
        manipulative_variables = ['X', 'Z']

    elif graph_name == 'CompleteGraph':
        # From the SEM: B->C, C->D, A->E, C->E, D->Y, E->Y
        # U1->A, U1->Y, U2->B, U2->Y are through hidden confounders
        # Observable edges only (matches Aglietti et al. 2020, Figure 2):
        dag_edges = [
            ('B', 'C'), ('C', 'D'),
            ('A', 'E'), ('C', 'E'), ('D', 'Y'), ('E', 'Y'),
        ]
        nodes = ['A', 'B', 'C', 'D', 'E', 'Y']
        hidden_nodes = []  # U1, U2 are not in the observable node set
        manipulative_variables = ['B', 'D', 'E']

    elif graph_name == 'CompleteGraph_NoCD':
        # Misspecified CompleteGraph: C->D edge removed
        dag_edges = [
            ('B', 'C'),
            ('A', 'E'), ('C', 'E'), ('D', 'Y'), ('E', 'Y'),
        ]
        nodes = ['A', 'B', 'C', 'D', 'E', 'Y']
        hidden_nodes = []
        manipulative_variables = ['B', 'D', 'E']

    elif graph_name == 'CompleteGraph_NoBC':
        # Misspecified CompleteGraph: B->C edge removed
        dag_edges = [
            ('C', 'D'),
            ('A', 'E'), ('C', 'E'), ('D', 'Y'), ('E', 'Y'),
        ]
        nodes = ['A', 'B', 'C', 'D', 'E', 'Y']
        hidden_nodes = []
        manipulative_variables = ['B', 'D', 'E']

    elif graph_name == 'LightTunnel':
        # Five direct parents of Y; no manipulable-to-manipulable edges.
        # U_color is hidden (declared in get_hidden_confounders) and yields a
        # bidirected R<->G edge in the projected ADMG.
        dag_edges = [
            ('R', 'Y'), ('G', 'Y'), ('B', 'Y'),
            ('P1', 'Y'), ('P2', 'Y'),
        ]
        nodes = ['R', 'G', 'B', 'P1', 'P2', 'Y']
        hidden_nodes = []
        manipulative_variables = ['R', 'G', 'B', 'P1', 'P2']

    elif graph_name == 'LightTunnel_WrongRG':
        # Misspecification: a spurious intra-cluster edge R -> G inside the
        # color cluster {R, G, B}. CBO derived from this DAG uses the
        # observational fit G | R when adjusting for do(R), which (thanks
        # to U_color) has a nontrivial slope -> biased adjustment.
        # Under the coarsening {R,G,B}|{P1,P2}|{Y}, both endpoints of R->G
        # sit in cluster {R,G,B}; Lee-2019 latent projection drops this
        # edge, so the C-DAG is identical to LightTunnel's.
        dag_edges = [
            ('R', 'G'),
            ('R', 'Y'), ('G', 'Y'), ('B', 'Y'),
            ('P1', 'Y'), ('P2', 'Y'),
        ]
        nodes = ['R', 'G', 'B', 'P1', 'P2', 'Y']
        hidden_nodes = []
        manipulative_variables = ['R', 'G', 'B', 'P1', 'P2']

    elif graph_name in ('SimplifiedCoralGraph', 'SimplifiedCoralGraph_NoST'):
        # SEM from SimplifiedCoralGraph.py:
        #   N, L exogenous; TE<-L; C<-{N,L,TE}; S<-TE;
        #   T<-S; D<-S; P<-{S,T,D,TE}; O<-{S,T,D,TE};
        #   CO<-{S,T,D,TE}; Y<-{L,N,P,O,C,CO,TE}
        # Manipulative: N, O, C, T, D  (no hidden confounders)
        dag_edges = [
            ('L', 'TE'), ('L', 'C'), ('L', 'Y'),
            ('N', 'C'), ('N', 'Y'),
            ('TE', 'C'), ('TE', 'S'), ('TE', 'P'), ('TE', 'O'), ('TE', 'CO'), ('TE', 'Y'),
            ('S', 'T'), ('S', 'D'), ('S', 'P'), ('S', 'O'), ('S', 'CO'),
            ('T', 'P'), ('T', 'O'), ('T', 'CO'),
            ('D', 'P'), ('D', 'O'), ('D', 'CO'),
            ('P', 'Y'), ('O', 'Y'), ('C', 'Y'), ('CO', 'Y'),
        ]
        if graph_name == 'SimplifiedCoralGraph_NoST':
            # Misspecification: remove S->T (intra-cluster edge inside {S,T,D})
            dag_edges = [e for e in dag_edges if e != ('S', 'T')]
        nodes = ['N', 'L', 'TE', 'C', 'S', 'T', 'D', 'P', 'O', 'CO', 'Y']
        hidden_nodes = []
        manipulative_variables = ['N', 'O', 'C', 'T', 'D']

    else:
        raise ValueError(f"Unknown graph: {graph_name}")

    return dag_edges, nodes, hidden_nodes, manipulative_variables


# ---------------------------------------------------------------------------
# Printing / diagnostic
# ---------------------------------------------------------------------------

def print_coarsenings(graph_name):
    """Print all valid coarsenings for a given graph."""
    dag_edges, nodes, hidden_nodes, manip_vars = get_dag_edges_from_sem(graph_name)
    hidden_confounders = get_hidden_confounders(graph_name)
    coarsenings = enumerate_valid_coarsenings(dag_edges, nodes, target='Y',
                                              hidden_nodes=hidden_nodes,
                                              hidden_confounders=hidden_confounders)

    print(f"\n{'='*60}")
    print(f"Valid coarsenings for {graph_name}")
    print(f"Observable nodes: {nodes}")
    print(f"Manipulative variables: {manip_vars}")
    print(f"DAG edges: {dag_edges}")
    print(f"Number of valid coarsenings: {len(coarsenings)}")
    print(f"{'='*60}")

    for i, c in enumerate(coarsenings):
        parts_str = ' | '.join(
            '{' + ','.join(sorted(p)) + '}' for p in c['partition']
        )
        q_edges = list(c['coarsened_dag'].edges)
        q_edges_str = ', '.join(
            '{' + ','.join(sorted(u)) + '}→{' + ','.join(sorted(v)) + '}'
            for u, v in q_edges
        )
        print(f"\n  Coarsening {i}: [{parts_str}]")
        print(f"    Parts: {c['num_parts']}, Quotient edges: {q_edges_str}")

        # Show which manipulative variables are in each coarsened node
        for part in c['partition']:
            manip_in_part = sorted(set(manip_vars) & part)
            if manip_in_part:
                print(f"    Manipulative node {{{','.join(sorted(part))}}}: "
                      f"intervene on {manip_in_part}")


# ---------------------------------------------------------------------------
# Lee-2019 latent projection + manipulable-only coarsening + POMIS
# ---------------------------------------------------------------------------
#
# ADMG representation used below:
#   admg = {
#       'vertices': frozenset of node names (or frozensets, for C-DAGs),
#       'di':       set of (u, v)   directed edges u -> v,
#       'bi':       set of frozenset({u, v})   bidirected edges u <-> v,
#   }
# Bidirected edges are stored as frozensets so {u,v} == {v,u}.


def _admg(vertices, di=(), bi=()):
    return {
        'vertices': frozenset(vertices),
        'di': set(tuple(e) for e in di),
        'bi': set(frozenset(e) for e in bi),
    }


def latent_project(di_edges, bi_edges, vertices, keep):
    """
    Richardson-Spirtes / Verma-Pearl latent projection of an ADMG.

    Given an ADMG ``G = (V, di, bi)`` and a subset ``K ⊆ V``, return the
    ADMG ``G[K]`` on ``K`` such that every m-connecting path in ``G``
    between two vertices in ``K`` has a corresponding edge in ``G[K]``.

    For a DAG input (``bi_edges = ()``) the projection specialises to:
      * u → v in G[K]  iff  ∃ directed path u → n₁ → ... → nₖ → v in G
          with u, v ∈ K and every intermediate nᵢ ∈ V\K;
      * u ↔ v in G[K]  iff  ∃ z ∈ V\K with directed paths z → ... → u
          and z → ... → v whose intermediates all lie in V\K (divergent
          structure through projected-out ancestors).

    Parameters
    ----------
    di_edges : iterable of (str, str)
    bi_edges : iterable of {str, str} (frozensets or pairs)
    vertices : iterable of str
    keep     : iterable of str

    Returns
    -------
    dict (ADMG) over ``keep`` with keys ``vertices``, ``di``, ``bi``.
    """
    V = set(vertices)
    K = set(keep) & V
    drop = V - K

    # Build adjacency for traversal. We need directed successors in G.
    succ = {v: set() for v in V}
    pred = {v: set() for v in V}
    for u, v in di_edges:
        succ.setdefault(u, set()).add(v)
        pred.setdefault(v, set()).add(u)
    # Bidirected neighbors (undirected)
    bi_nbr = {v: set() for v in V}
    for e in bi_edges:
        a, b = tuple(e)
        bi_nbr.setdefault(a, set()).add(b)
        bi_nbr.setdefault(b, set()).add(a)

    # ----- projected directed edges -----
    # For each u in K, BFS through drop to find v in K reachable via
    # u → drop* → v (first edge outgoing from u).
    new_di = set()
    for u in K:
        visited = set()
        stack = [u]
        first = True
        while stack:
            cur = stack.pop()
            for nxt in succ.get(cur, ()):
                if nxt in K and cur == u:
                    new_di.add((u, nxt))  # direct edge
                elif nxt in K:
                    new_di.add((u, nxt))  # reached via drop path
                elif nxt in drop and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
            first = False

    # ----- projected bidirected edges -----
    # For each z in drop, find all k in K reachable from z via directed
    # paths whose strict intermediates lie entirely in drop.  Any two such
    # k, k' become k ↔ k'.  Also absorb bidirected edges whose endpoints
    # both fall in K directly.
    new_bi = set()
    for z in drop:
        reached = set()
        stack = list(succ.get(z, ()))
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in K:
                reached.add(cur)
                continue  # path must end at first K vertex hit
            # cur is in drop: keep walking downstream
            for nxt in succ.get(cur, ()):
                stack.append(nxt)
        # Pair everything reached as bidirected
        reached = list(reached)
        for i in range(len(reached)):
            for j in range(i + 1, len(reached)):
                new_bi.add(frozenset({reached[i], reached[j]}))

    # Bidirected edges present in input with both endpoints in K
    for e in bi_edges:
        a, b = tuple(e)
        if a in K and b in K:
            new_bi.add(frozenset({a, b}))

    # Bidirected edges through a drop-only chain that ends at two K vertices
    # via bidirected-then-directed paths — rare in CBO setups (no input bi),
    # skipped here for clarity. Extend if future benchmarks introduce them.

    return _admg(K, di=new_di, bi=new_bi)


def _dag_from_sem_with_latents(dag_edges, nodes, hidden_nodes, hidden_confounders):
    """
    Build the *full* DAG including observable nodes, latent confounder
    nodes (from ``hidden_confounders``), and directed edges from latents
    to their affected observables. Returns (V, di_edges).
    """
    V = set(nodes)
    di = set(tuple(e) for e in dag_edges)
    hidden_nodes = set(hidden_nodes or [])
    V -= hidden_nodes  # hidden_nodes passed here are *not* in the observable DAG
    if hidden_confounders:
        for name, affected in hidden_confounders:
            latent = f'__{name}__'
            V.add(latent)
            for v in affected:
                if v in V:
                    di.add((latent, v))
    return V, di


def project_graph_to_M_and_Y(graph_name):
    """
    **Strict** projection: project out *all* non-manipulable vertices
    (both hidden confounders and non-manipulable observables).

    Kept for the Prop 4 ablation — demonstrating that when ``N`` is absorbed
    into bidirected edges, POMIS may collapse (e.g. on CompleteGraph to
    ``{∅}``).  Not used in the main pipeline.  For the main pipeline use
    :func:`project_out_hidden`.
    """
    dag_edges, nodes, hidden_nodes, manip_vars = get_dag_edges_from_sem(graph_name)
    hidden_conf = get_hidden_confounders(graph_name)
    V, di = _dag_from_sem_with_latents(dag_edges, nodes, hidden_nodes, hidden_conf)
    keep = set(manip_vars) | {'Y'}
    return latent_project(di, (), V, keep)


def project_out_hidden(graph_name):
    """
    Lee-2019 standard projection: project out only the *hidden*
    confounders (latent U nodes), keeping every observable vertex —
    including non-manipulable observables — as a vertex of the resulting
    ADMG.

    This is the projection used throughout the main CCBO pipeline: the
    action space is later restricted to subsets of ``M``, but the full
    observable graph remains available for identification.  POMIS
    computed on this ADMG and restricted to manipulable arms matches
    POMIS on the original DAG with latent confounders (Lee-2019 Thm 4).

    Returns
    -------
    dict
        ADMG over all observable vertices with bidirected edges encoding
        hidden-confounder-induced common causes.
    """
    dag_edges, nodes, hidden_nodes, manip_vars = get_dag_edges_from_sem(graph_name)
    hidden_conf = get_hidden_confounders(graph_name)
    V, di = _dag_from_sem_with_latents(dag_edges, nodes, hidden_nodes, hidden_conf)
    observables = set(nodes) - set(hidden_nodes or [])
    return latent_project(di, (), V, observables)


def build_coarsened_admg(partition, projected_admg, atomic_vertices=None):
    """
    Quotient a projected ADMG under a partition of a subset of its vertices.

    Non-partition vertices (``atomic_vertices``) are lifted to singleton
    parts automatically, preserving them as atomic C-DAG nodes.  This is
    how non-manipulable observables participate in the coarsened graph
    under standard Lee-2019: they remain as atomic vertices alongside
    manipulable clusters.

    Parameters
    ----------
    partition : list of frozenset
        Partition of a subset of ``projected_admg['vertices']`` (typically
        ``M ∪ {Y}`` with ``{Y}`` as a singleton).
    projected_admg : dict
        ADMG returned by :func:`project_out_hidden` (or the stricter
        :func:`project_graph_to_M_and_Y`).
    atomic_vertices : iterable, optional
        Extra observable vertices to include as singleton parts (typically
        the non-manipulable observables).  Defaults to every vertex of
        ``projected_admg`` not already covered by ``partition``.

    Returns
    -------
    dict
        Coarsened ADMG. Self-loops (intra-cluster directed edges) and
        intra-cluster bidirected edges are dropped.
    """
    covered = set()
    for part in partition:
        covered |= set(part)

    if atomic_vertices is None:
        atomic_vertices = projected_admg['vertices'] - covered
    else:
        atomic_vertices = set(atomic_vertices) - covered

    full_partition = list(partition) + [frozenset({v}) for v in atomic_vertices]

    node_to_part = {}
    for part in full_partition:
        for v in part:
            node_to_part[v] = part

    c_vertices = frozenset(full_partition)
    c_di = set()
    for u, v in projected_admg['di']:
        pu, pv = node_to_part.get(u), node_to_part.get(v)
        if pu is None or pv is None or pu == pv:
            continue
        c_di.add((pu, pv))
    c_bi = set()
    for e in projected_admg['bi']:
        a, b = tuple(e)
        pa, pb = node_to_part.get(a), node_to_part.get(b)
        if pa is None or pb is None or pa == pb:
            continue
        c_bi.add(frozenset({pa, pb}))
    return _admg(c_vertices, di=c_di, bi=c_bi)


def _admg_is_acyclic(admg):
    g = nx.DiGraph()
    g.add_nodes_from(admg['vertices'])
    g.add_edges_from(admg['di'])
    return nx.is_directed_acyclic_graph(g)


def enumerate_valid_coarsenings_manip(graph_name, max_parts=None):
    """
    Enumerate all valid coarsenings of the manipulable set.

    Hidden confounders are projected out (becoming bidirected edges in the
    projected ADMG); non-manipulable observables remain as atomic vertices
    of the coarsened ADMG (Lee-2019 standard reading).  A coarsening is
    valid iff the coarsened ADMG's directed subgraph is acyclic.

    Parameters
    ----------
    graph_name : str
        A graph recognised by ``get_dag_edges_from_sem``.
    max_parts : int, optional
        Skip partitions with more than ``max_parts`` parts (counting the
        ``{Y}`` singleton *and* the non-manipulable atomic vertices).

    Returns
    -------
    list of dict
        Each dict has:
        - 'partition' : list of frozenset
              The partition of ``M`` plus ``{Y}`` as a singleton.  Does
              *not* include non-manipulable atomic vertices explicitly;
              those are present as singleton vertices of ``coarsened_admg``.
        - 'coarsened_admg' : dict (quotient ADMG including atomic
              non-manipulable vertices)
        - 'manip_clusters' : list of frozenset
              The cluster vertices of ``coarsened_admg`` that are
              manipulable (i.e. subsets of ``M``).
        - 'num_parts' : int (number of M-clusters + 1 for Y)
    """
    projected = project_out_hidden(graph_name)
    _, all_nodes, hidden_nodes, manip_vars = get_dag_edges_from_sem(graph_name)
    hidden_nodes = set(hidden_nodes or [])
    manip_sorted = sorted(manip_vars)
    nonmanip_observables = (set(all_nodes) - hidden_nodes
                             - set(manip_vars) - {'Y'})
    coarsenings = []
    for partition in _generate_partitions(manip_sorted):
        full = partition + [frozenset(['Y'])]
        if max_parts is not None and len(full) > max_parts:
            continue
        admg = build_coarsened_admg(full, projected,
                                    atomic_vertices=nonmanip_observables)
        if not _admg_is_acyclic(admg):
            continue
        coarsenings.append({
            'partition': full,
            'coarsened_admg': admg,
            'manip_clusters': list(partition),
            'num_parts': len(full),
        })
    coarsenings.sort(key=lambda c: -c['num_parts'])
    return coarsenings


# ---------------------------------------------------------------------------
# POMIS (Lee & Bareinboim 2018, Thm 6) on a coarsened ADMG
# ---------------------------------------------------------------------------

def _ancestors_di(admg, targets):
    """Ancestors of ``targets`` w.r.t. directed edges only (inclusive)."""
    anc = set(targets)
    changed = True
    preds = {v: set() for v in admg['vertices']}
    for u, v in admg['di']:
        preds.setdefault(v, set()).add(u)
    while changed:
        changed = False
        for v in list(anc):
            for p in preds.get(v, ()):
                if p not in anc:
                    anc.add(p)
                    changed = True
    return anc


def _c_component(admg, vertex):
    """C-component (district) of ``vertex``: connected component in bi-graph."""
    g = nx.Graph()
    g.add_nodes_from(admg['vertices'])
    for e in admg['bi']:
        a, b = tuple(e)
        g.add_edge(a, b)
    return set(nx.node_connected_component(g, vertex)) if vertex in g.nodes else {vertex}


def _subgraph(admg, vertices):
    V = set(vertices)
    di = {(u, v) for u, v in admg['di'] if u in V and v in V}
    bi = {e for e in admg['bi'] if set(tuple(e)) <= V}
    return _admg(V, di=di, bi=bi)


def _muct_ib(admg, target):
    """
    Compute (MUCT, IB) on ``admg`` for the given ``target`` per
    Lee & Bareinboim 2018.

    MUCT = minimal unobserved-confounder territory reachable from ``target``
    through the c-component closure + ancestors.
    IB   = interventional border: parents of MUCT (in the directed graph)
           that are not themselves in MUCT.
    """
    # Work on the ancestral subgraph of target — vertices not ancestors of
    # target are irrelevant.
    anc = _ancestors_di(admg, {target})
    sub = _subgraph(admg, anc)

    # Iteratively grow MUCT: start with {target}; close under c-component
    # within the ancestral subgraph; then add directed ancestors that share
    # a c-component edge with anything in MUCT; repeat to fixpoint.
    muct = {target}
    while True:
        old = set(muct)
        # Close under c-component in sub
        closed = set()
        for v in muct:
            closed |= _c_component(sub, v)
        muct |= closed
        # Add any vertex in sub that is an ancestor of a muct vertex AND
        # shares a bidirected edge with any muct vertex.  Equivalently:
        # take c-component neighbours of muct inside sub.
        if muct == old:
            break

    # IB = parents of MUCT in sub that are not in MUCT
    ib = set()
    for u, v in sub['di']:
        if v in muct and u not in muct:
            ib.add(u)
    return muct, ib, sub


def compute_POMIS(admg, manipulable_vertices, target):
    """
    Enumerate Possibly-Optimal Minimal Intervention Sets (Lee-Bareinboim 2018),
    restricted to the manipulable set M.

    A subset X ⊆ M is an M-POMIS for ``target`` iff, on the mutilated ADMG
    G_X̄ (all incoming edges to X removed), we have

        IB(G_X̄, target) ∩ M = X.

    The restriction to M is necessary whenever the graph contains
    non-manipulable variables with direct paths to ``target`` (e.g.
    SimplifiedCoralGraph: L→Y, TE→Y).  Those nodes are always in IB but
    cannot be controlled, so the strict Lee-Bareinboim condition
    IB = X would yield an empty POMIS.  Restricting to M recovers the
    correct semantics: X covers all *controllable* causal paths.

    On graphs where IB(G, target) ⊆ M (e.g. CompleteGraph), M-POMIS is
    identical to the original POMIS.

    Parameters
    ----------
    admg : dict
        ADMG (typically the coarsened one).
    manipulable_vertices : iterable
        Vertices available for intervention (for C-DAG: manipulable cluster
        nodes, i.e. every cluster node except the target singleton and any
        pure-latent node).
    target : hashable
        Target vertex (for C-DAG: frozenset({'Y'})).

    Returns
    -------
    list of frozenset
        Each element is a POMIS.  Always contains the empty set of
        interventions as a "baseline" only if IB(G, target) ∩ M == ∅;
        otherwise the empty set is not a POMIS.
    """
    M_set = frozenset(v for v in manipulable_vertices if v != target)
    M = list(M_set)
    pomis = []
    for r in range(0, len(M) + 1):
        for combo in itertools.combinations(M, r):
            X = set(combo)
            # Mutilate: drop all incoming directed edges to X
            di_mut = {(u, v) for u, v in admg['di'] if v not in X}
            bi_mut = set(admg['bi'])  # bi edges incident to X survive
            g_mut = _admg(admg['vertices'], di=di_mut, bi=bi_mut)
            _, ib, _ = _muct_ib(g_mut, target)
            # Restrict IB to the manipulable set before comparison
            if ib & M_set == X:
                pomis.append(frozenset(X))
    return pomis


def admg_to_ananke(admg, name_map=None):
    """
    Convert a coarsened ADMG (frozenset-typed vertices) to an ananke ADMG
    with string vertex names.

    Returns (ananke.graphs.ADMG, dict mapping original vertex -> string).
    """
    from ananke.graphs import ADMG
    if name_map is None:
        name_map = {}
        for i, v in enumerate(sorted(admg['vertices'], key=lambda x: sorted(x))):
            # Build a readable name like "B_D" for {B, D}, "Y" for {Y}
            name_map[v] = '_'.join(sorted(v))
    vs = [name_map[v] for v in admg['vertices']]
    di = [(name_map[u], name_map[v]) for u, v in admg['di']]
    bi = [tuple(sorted(name_map[x] for x in tuple(e))) for e in admg['bi']]
    return ADMG(vs, di, bi), name_map


if __name__ == '__main__':
    print_coarsenings('CompleteGraph')
    print()
    print('=' * 60)
    print('Lee-2019 projection of CompleteGraph onto M ∪ {Y}')
    print('=' * 60)
    projected = project_graph_to_M_and_Y('CompleteGraph')
    print('vertices:', sorted(projected['vertices']))
    print('directed:', sorted(projected['di']))
    print('bidirected:', sorted((sorted(tuple(e)) for e in projected['bi'])))
