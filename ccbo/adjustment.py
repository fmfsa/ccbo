"""
Causal effect identification for coarsened causal Bayesian optimisation.

This module exposes two identification pipelines:

(A) **C-DAG GID-PO gate** — :func:`make_cdag_do_function` (primary CCBO path).
    Given a coarsened ADMG (the quotient of the Lee-2019-projected graph
    under a manipulable-only partition), we:
      1. Convert the ADMG to :mod:`ananke.graphs.ADMG`.
      2. Run :class:`ananke.identification.OneLineID` on the *cluster-level*
         causal query ``P(Y | do(X))`` where ``X`` is the set of cluster
         vertices containing the fine-grained intervention variables.  This
         is a **complete** identification oracle (Shpitser & Pearl 2006 /
         Lee 2020), so the "is the effect identifiable from the C-DAG?"
         question is settled by one call.
      3. If identifiable, construct a GP-based estimator by expanding each
         bidirected edge ``u ↔ v`` in the ADMG into an explicit latent
         common-cause node, then running the classic backdoor → frontdoor →
         g-computation cascade on the resulting DAG.  This cascade is
         sound on the expanded DAG; the ananke gate ensures soundness is
         sufficient (whatever is identifiable on the ADMG has a working
         formula on the expanded DAG because the expansion preserves all
         m-separation statements).
      4. If not identifiable, return ``None`` so the caller uses an
         uninformative GP prior — a **principled** fallback, not a mask for
         a completeness gap.

(B) **Fine-grained DAG pipeline** — :func:`make_adjustment_do_function`.
    Legacy path kept for plain CBO benchmarks that operate directly on the
    fine-grained DAG.  Uses the same BD → FD → g-computation cascade on
    ``build_full_dag(graph_name)``.  No C-DAG / ananke involvement.

GPs are fitted ONCE when the do-function is created, not at each evaluation.
"""

import warnings

import numpy as np
import networkx as nx
from itertools import combinations
import GPy


# ---------------------------------------------------------------------------
# Helper: fit a GP
# ---------------------------------------------------------------------------

def _fit_gp(X, Y):
    """Fit a GPy regression model with RBF-ARD kernel."""
    kernel = GPy.kern.RBF(X.shape[1], ARD=True)
    gp = GPy.models.GPRegression(X, Y, kernel)
    # Constrain the noise variance to a wide bounded range and let it be
    # optimised together with the kernel hyperparameters. Fixing to 1e-2
    # silently mis-scales the GP for any objective whose Y is not O(1).
    gp.likelihood.variance.constrain_bounded(1e-6, 10.0, warning=False)
    gp.optimize(messages=False)
    return gp


# ---------------------------------------------------------------------------
# 1. Graph construction helpers
# ---------------------------------------------------------------------------

def build_full_dag(graph_name):
    """
    Build the full causal DAG (including latent nodes) for a named graph.
    """
    from .coarsening import get_dag_edges_from_sem

    edges, obs_nodes, hidden_nodes, manip_vars = get_dag_edges_from_sem(graph_name)
    observed = set(obs_nodes)
    latent = set(hidden_nodes or [])

    dag = nx.DiGraph()
    dag.add_nodes_from(observed | latent)

    for u, v in edges:
        dag.add_edge(u, v)

    if graph_name in ('CompleteGraph', 'CompleteGraph_NoCD', 'CompleteGraph_NoBC'):
        dag.add_edges_from([('U1', 'A'), ('U1', 'Y'),
                            ('U2', 'B'), ('U2', 'Y')])
    elif graph_name == 'ToyGraph':
        dag.add_edges_from([('U', 'X'), ('U', 'Y')])

    return dag, observed, latent, manip_vars


# ---------------------------------------------------------------------------
# 2. Backdoor and frontdoor criterion
# ---------------------------------------------------------------------------

def find_backdoor_set(dag, S, Y, observed):
    """
    Find a minimal valid backdoor adjustment set Z for do(S) on Y.
    """
    S_set = set(S)
    desc_S = set()
    for s in S_set:
        desc_S |= nx.descendants(dag, s)
    desc_S -= S_set

    candidates = sorted(observed - S_set - desc_S - {Y})

    dag_no_out = dag.copy()
    for s in S_set:
        for succ in list(dag.successors(s)):
            dag_no_out.remove_edge(s, succ)

    for size in range(len(candidates) + 1):
        for Z_tuple in combinations(candidates, size):
            Z_set = set(Z_tuple)
            separated = all(
                nx.is_d_separator(dag_no_out, {s}, {Y}, Z_set)
                for s in S_set
            )
            if separated:
                return sorted(Z_set)

    return None


def find_frontdoor_set(dag, S, Y, observed):
    """
    Find a valid frontdoor adjustment set M for do(S) on Y.
    """
    S_set = set(S)
    candidates = sorted(observed - S_set - {Y})

    for size in range(1, len(candidates) + 1):
        for M_tuple in combinations(candidates, size):
            M_set = set(M_tuple)

            # (i) All directed paths from S to Y pass through M
            all_intercepted = True
            for s in S_set:
                for path in nx.all_simple_paths(dag, s, Y):
                    intermediates = set(path[1:-1])
                    if not (intermediates & M_set):
                        all_intercepted = False
                        break
                if not all_intercepted:
                    break
            if not all_intercepted:
                continue

            # (ii) No unblocked backdoor path from S to M
            dag_no_out_S = dag.copy()
            for s in S_set:
                for succ in list(dag.successors(s)):
                    dag_no_out_S.remove_edge(s, succ)

            cond2 = all(
                nx.is_d_separator(dag_no_out_S, {s}, {m}, set())
                for s in S_set for m in M_set
            )
            if not cond2:
                continue

            # (iii) S blocks all backdoor paths from M to Y
            dag_no_out_M = dag.copy()
            for m in M_set:
                for succ in list(dag.successors(m)):
                    dag_no_out_M.remove_edge(m, succ)

            cond3 = all(
                nx.is_d_separator(dag_no_out_M, {m}, {Y}, S_set)
                for m in M_set
            )
            if not cond3:
                continue

            return sorted(M_set)

    return None


# ---------------------------------------------------------------------------
# 3. G-computation (chain propagation)
# ---------------------------------------------------------------------------

def find_gcomputation_plan(dag, S, Y, observed):
    """
    Build a g-computation plan: propagate the intervention through the
    graph in topological order using observable conditional distributions.
    """
    S_set = set(S)

    obs_dag = nx.DiGraph()
    for u, v in dag.edges():
        if u in observed and v in observed:
            obs_dag.add_edge(u, v)
    obs_dag.add_nodes_from(observed)

    topo = list(nx.topological_sort(obs_dag))

    affected = set()
    for s in S_set:
        if s in obs_dag:
            for desc in nx.descendants(obs_dag, s):
                affected.add(desc)

    propagation = []
    propagated = set(S_set)

    for v in topo:
        if v in S_set or v == Y or v not in affected:
            continue
        obs_parents = sorted(set(obs_dag.predecessors(v)))
        
        # Check against latent confounders between v and its observed parents
        v_latents = set(dag.predecessors(v)) - observed
        confounded = False
        for p in obs_parents:
            p_latents = set(dag.predecessors(p)) - observed
            if v_latents & p_latents:
                confounded = True
                break
        if confounded:
            return None

        can_propagate = all(
            p in propagated or p in S_set or p not in affected
            for p in obs_parents
        )
        if can_propagate:
            propagation.append((v, obs_parents))
            propagated.add(v)

    obs_parents_Y = sorted(set(obs_dag.predecessors(Y)))
    
    # Check against latent confounders between Y and its observed parents
    Y_latents = set(dag.predecessors(Y)) - observed
    confounded_Y = False
    for p in obs_parents_Y:
        p_latents = set(dag.predecessors(p)) - observed
        if Y_latents & p_latents:
            confounded_Y = True
            break
    if confounded_Y:
        return None

    parents_ready = all(
        p in propagated or p in S_set or p not in affected
        for p in obs_parents_Y
    )
    if not parents_ready:
        return None

    final_conditioning = sorted(
        set(obs_parents_Y) | (observed - S_set - {Y} - affected)
    )

    return {
        'method': 'gcomputation',
        'propagation_order': propagation,
        'conditioning_vars': final_conditioning,
        'affected_vars': sorted(affected - {Y}),
    }


# ---------------------------------------------------------------------------
# 4. Unified identification
# ---------------------------------------------------------------------------

def identify_adjustment(dag, S, Y, observed):
    """
    Find the best identification strategy for E[Y | do(S)].
    Priority: backdoor > frontdoor > g-computation.
    """
    S_set = set(S)

    Z = find_backdoor_set(dag, S_set, Y, observed)
    if Z is not None:
        return {'method': 'backdoor', 'adjustment_set': Z}

    M = find_frontdoor_set(dag, S_set, Y, observed)
    if M is not None:
        return {'method': 'frontdoor', 'adjustment_set': M}

    plan = find_gcomputation_plan(dag, S_set, Y, observed)
    if plan is not None:
        return plan

    return {'method': 'none', 'adjustment_set': []}


# ---------------------------------------------------------------------------
# 5. Pre-fitted causal effect estimation
# ---------------------------------------------------------------------------

def _build_backdoor_predictor(S_vars, Z_vars, Y_var, observational_samples):
    """
    Pre-fit a GP for backdoor adjustment and return a fast predictor.

    Returns a function: value -> (mean_do, var_do)
    """
    n = len(observational_samples)
    input_vars = S_vars + Z_vars
    X_obs = np.column_stack(
        [observational_samples[v].values for v in input_vars]
    )
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]

    gp = _fit_gp(X_obs, Y_obs)

    Z_obs = np.column_stack(
        [observational_samples[v].values for v in Z_vars]
    ) if Z_vars else np.empty((n, 0))

    def predict(value):
        value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
        S_repeated = np.tile(value, (n, 1))
        X_pred = np.hstack([S_repeated, Z_obs]) if Z_vars else S_repeated
        mu, var = gp.predict(X_pred)
        mean_do = float(np.mean(mu))
        var_do = float(np.mean(var) + np.var(mu))
        return mean_do, var_do

    return predict


def _build_frontdoor_predictor(S_vars, M_vars, Y_var, observational_samples):
    """
    Pre-fit GPs for frontdoor adjustment and return a fast predictor.
    """
    n = len(observational_samples)
    S_obs = np.column_stack(
        [observational_samples[v].values for v in S_vars]
    )

    # Stage 1 GPs: M_i | S
    m_gps = []
    for m_var in M_vars:
        M_obs = observational_samples[m_var].values[:, np.newaxis]
        m_gps.append(_fit_gp(S_obs, M_obs))

    # Stage 2 GP: Y | (M, S)
    input_vars_ym = M_vars + S_vars
    X_obs_ym = np.column_stack(
        [observational_samples[v].values for v in input_vars_ym]
    )
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]
    gp_y = _fit_gp(X_obs_ym, Y_obs)

    def predict(value):
        value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()

        # Stage 1: predict M
        m_preds = []
        for gp_m in m_gps:
            m_pred, _ = gp_m.predict(value.reshape(1, -1))
            m_preds.append(float(m_pred[0, 0]))
        m_hat = np.array(m_preds)

        # Stage 2: E_S'[E[Y | M=m_hat, S=s']]
        M_repeated = np.tile(m_hat, (n, 1))
        X_pred = np.hstack([M_repeated, S_obs])
        mu, var = gp_y.predict(X_pred)
        mean_do = float(np.mean(mu))
        var_do = float(np.mean(var) + np.var(mu))
        return mean_do, var_do

    return predict


def _build_gcomputation_predictor(S_vars, plan, Y_var, observational_samples):
    """
    Pre-fit all GPs for g-computation and return a fast predictor.
    """
    n = len(observational_samples)

    # Pre-fit GPs for each propagation step
    prop_gps = []
    for var, parents in plan['propagation_order']:
        if not parents:
            prop_gps.append(None)
            continue
        X_parent_obs = np.column_stack(
            [observational_samples[p].values for p in parents]
        )
        Y_var_obs = observational_samples[var].values[:, np.newaxis]
        prop_gps.append(_fit_gp(X_parent_obs, Y_var_obs))

    # Pre-fit GP for Y
    cond_vars = plan['conditioning_vars']
    X_cond_obs = np.column_stack(
        [observational_samples[v].values for v in cond_vars]
    )
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]
    gp_y = _fit_gp(X_cond_obs, Y_obs)

    # Cache observational values
    obs_values = {col: observational_samples[col].values.copy()
                  for col in observational_samples.columns}
    S_set = set(S_vars)
    affected_set = set(plan.get('affected_vars', []))

    def predict(value):
        value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()

        # Start with observational data
        current_values = {k: v.copy() for k, v in obs_values.items()}

        # Set intervention values
        for i, s_var in enumerate(S_vars):
            current_values[s_var] = np.full(n, value[i])

        # Propagate through affected variables
        for (var, parents), gp_var in zip(plan['propagation_order'], prop_gps):
            if gp_var is None:
                continue
            X_parent_current = np.column_stack(
                [current_values[p] for p in parents]
            )
            pred_mu, _ = gp_var.predict(X_parent_current)
            current_values[var] = pred_mu.ravel()

        # Predict Y
        X_cond_current = np.column_stack(
            [current_values[v] for v in cond_vars]
        )
        mu, var = gp_y.predict(X_cond_current)
        mean_do = float(np.mean(mu))
        var_do = float(np.mean(var) + np.var(mu))
        return mean_do, var_do

    return predict


# ---------------------------------------------------------------------------
# 6. Unified interface
# ---------------------------------------------------------------------------

def make_adjustment_do_function(graph_name, intervention_vars, observational_samples=None):
    """
    Create a do-calculus function using the best available identification
    strategy: backdoor > frontdoor > g-computation.

    If observational_samples is provided, GPs are pre-fitted and the returned
    function only does prediction (fast). Otherwise, the function fits GPs
    from scratch at each call (slow, for backward compatibility).

    The returned function has the signature expected by the CBO codebase:
        compute_do(observational_samples, functions, value) -> (mean, var)
    """
    dag, observed, latent, _ = build_full_dag(graph_name)
    ident = identify_adjustment(dag, intervention_vars, 'Y', observed)

    if ident['method'] == 'backdoor':
        Z_vars = ident['adjustment_set']

        if observational_samples is not None:
            predictor = _build_backdoor_predictor(
                intervention_vars, Z_vars, 'Y', observational_samples)

            def compute_do(observational_samples, functions, value):
                return predictor(value)
        else:
            def compute_do(observational_samples, functions, value):
                return _compute_backdoor_slow(
                    intervention_vars, Z_vars, 'Y', value,
                    observational_samples)

        return compute_do, ident

    elif ident['method'] == 'frontdoor':
        M_vars = ident['adjustment_set']

        if observational_samples is not None:
            predictor = _build_frontdoor_predictor(
                intervention_vars, M_vars, 'Y', observational_samples)

            def compute_do(observational_samples, functions, value):
                return predictor(value)
        else:
            def compute_do(observational_samples, functions, value):
                return _compute_frontdoor_slow(
                    intervention_vars, M_vars, 'Y', value,
                    observational_samples)

        return compute_do, ident

    elif ident['method'] == 'gcomputation':
        plan = ident

        if observational_samples is not None:
            predictor = _build_gcomputation_predictor(
                intervention_vars, plan, 'Y', observational_samples)

            def compute_do(observational_samples, functions, value):
                return predictor(value)
        else:
            def compute_do(observational_samples, functions, value):
                return _compute_gcomputation_slow(
                    intervention_vars, plan, 'Y', value,
                    observational_samples)

        return compute_do, ident

    else:
        return None, ident


# ---------------------------------------------------------------------------
# 7. C-DAG level identification helpers
# ---------------------------------------------------------------------------

def cdag_to_string_dag(cdag):
    """
    Convert a frozenset-node C-DAG to a string-node DiGraph.

    Frozenset nodes like frozenset({'B','C','D'}) become strings like '{B,C,D}'.
    String nodes are sortable, making them compatible with identify_adjustment.

    Returns
    -------
    string_dag : nx.DiGraph
        Same topology, string node labels.
    node_map : dict
        frozenset node -> string label.
    """
    def cluster_to_str(cluster):
        return '{' + ','.join(sorted(cluster)) + '}'

    node_map = {node: cluster_to_str(node) for node in cdag.nodes}
    string_dag = nx.DiGraph()
    for s in node_map.values():
        string_dag.add_node(s)
    for u, v in cdag.edges:
        string_dag.add_edge(node_map[u], node_map[v])
    return string_dag, node_map


def _cluster_str_to_vars(cluster_str):
    """Parse '{B,C,D}' → ['B', 'C', 'D']."""
    inner = cluster_str.strip('{}')
    return sorted(inner.split(',')) if inner else []


def _build_cdag_gcomputation_predictor(S_vars, plan, Y_var, observational_samples):
    """
    G-computation predictor for C-DAG level identification.

    The plan's propagation_order contains (cluster_str, parent_cluster_strs)
    tuples from the string C-DAG. Each cluster string is expanded to its
    fine-grained variables for GP fitting and prediction.
    """
    n = len(observational_samples)
    obs_cols = set(observational_samples.columns)

    # Pre-fit GPs for each propagation step.
    # For each intermediate cluster: one GP per fine var in that cluster,
    # conditioned on fine vars of parent clusters.
    prop_gps = []  # list of (fine_vars, parent_fine_vars, list_of_GPs)
    for cluster_str, parent_cluster_strs in plan['propagation_order']:
        fine_vars = [v for v in _cluster_str_to_vars(cluster_str) if v in obs_cols]
        parent_fine_vars = []
        for p_str in parent_cluster_strs:
            parent_fine_vars.extend(
                [v for v in _cluster_str_to_vars(p_str) if v in obs_cols]
            )

        if not parent_fine_vars:
            prop_gps.append((fine_vars, parent_fine_vars, []))
            continue

        X_parent_obs = np.column_stack(
            [observational_samples[v].values for v in parent_fine_vars]
        )
        cluster_gps = []
        for var in fine_vars:
            Y_var_obs = observational_samples[var].values[:, np.newaxis]
            cluster_gps.append(_fit_gp(X_parent_obs, Y_var_obs))
        prop_gps.append((fine_vars, parent_fine_vars, cluster_gps))

    # Pre-fit GP for Y: expand conditioning cluster strings to fine vars.
    cond_cluster_strs = plan['conditioning_vars']
    cond_fine_vars = []
    for c_str in cond_cluster_strs:
        cond_fine_vars.extend(
            [v for v in _cluster_str_to_vars(c_str)
             if v in obs_cols and v != Y_var]
        )

    if cond_fine_vars:
        X_cond_obs = np.column_stack(
            [observational_samples[v].values for v in cond_fine_vars]
        )
    else:
        X_cond_obs = np.ones((n, 1))
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]
    gp_y = _fit_gp(X_cond_obs, Y_obs)

    obs_values = {col: observational_samples[col].values.copy()
                  for col in observational_samples.columns}

    def predict(value):
        value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
        current_values = {k: v.copy() for k, v in obs_values.items()}

        # Set intervention values
        for i, s_var in enumerate(S_vars):
            current_values[s_var] = np.full(n, value[i])

        # Propagate through intermediate clusters
        for (fine_vars, parent_fine_vars, cluster_gps) in prop_gps:
            if not cluster_gps or not parent_fine_vars:
                continue
            X_parent_current = np.column_stack(
                [current_values[v] for v in parent_fine_vars]
            )
            for var, gp_var in zip(fine_vars, cluster_gps):
                pred_mu, _ = gp_var.predict(X_parent_current)
                current_values[var] = pred_mu.ravel()

        # Predict Y using updated current_values (intervention vars already set)
        if cond_fine_vars:
            X_cond_current = np.column_stack(
                [current_values[v] for v in cond_fine_vars]
            )
        else:
            X_cond_current = np.ones((n, 1))
        mu, var = gp_y.predict(X_cond_current)
        mean_do = float(np.mean(mu))
        var_do = float(np.mean(var) + np.var(mu))
        return mean_do, var_do

    return predict


# ---------------------------------------------------------------------------
# ADMG utilities: ananke gate + latent-expansion DAG for GP estimator reuse
# ---------------------------------------------------------------------------

def _is_admg_dict(obj):
    """Duck-type check for the internal ADMG dict from coarsening.py."""
    return (isinstance(obj, dict)
            and 'vertices' in obj and 'di' in obj and 'bi' in obj)


def _cdag_to_admg_dict(cdag):
    """Wrap a legacy nx.DiGraph C-DAG as an ADMG dict (no bidirected edges)."""
    return {
        'vertices': frozenset(cdag.nodes),
        'di': set((u, v) for u, v in cdag.edges),
        'bi': set(),
    }


def _admg_cluster_name(cluster):
    """Readable string name for a cluster frozenset."""
    return '{' + ','.join(sorted(cluster)) + '}'


def _ananke_id_check(admg_dict, X_vertices, Y_vertex):
    """
    Complete identifiability gate via ananke's ID algorithm.

    Returns
    -------
    identifiable : bool
    functional : str or None
        Symbolic Tian-Pearl factorisation as returned by ananke (diagnostic).
    deferred : str or None
        None when the gate ran; otherwise a short reason string recording
        that ananke crashed at runtime and the decision was deferred to the
        (sound but incomplete) estimator cascade.  Callers should surface
        this in their identification info.
    """
    try:
        from ananke.graphs import ADMG
        from ananke.identification import OneLineID
    except ImportError as e:
        # The two-tier prior is only principled if the identifiability gate
        # is complete. Running without ananke would silently treat every
        # query as identifiable, so make it a hard requirement.
        raise ImportError(
            "ananke-causal is required for the CCBO identifiability gate "
            "(pip install ananke-causal)") from e

    # Stringify the ADMG for ananke's interface.
    name_map = {v: _admg_cluster_name(v) if isinstance(v, frozenset) else str(v)
                for v in admg_dict['vertices']}
    vs = [name_map[v] for v in admg_dict['vertices']]
    di = [(name_map[u], name_map[v]) for u, v in admg_dict['di']]
    bi = [tuple(sorted(name_map[x] for x in tuple(e)))
          for e in admg_dict['bi']]
    try:
        aa = ADMG(vs, di, bi)
    except (ValueError, TypeError, KeyError, AssertionError) as e:
        reason = (f"ananke ADMG construction failed for X={X_vertices}, "
                  f"Y={Y_vertex} ({type(e).__name__}: {e})")
        warnings.warn(reason + "; deferring to estimator cascade.",
                      RuntimeWarning, stacklevel=2)
        return True, None, reason

    X_names = [name_map[x] for x in X_vertices if x in name_map]
    Y_name = name_map[Y_vertex]
    try:
        oid = OneLineID(aa, X_names, [Y_name])
        identifiable = bool(oid.id())
        functional = None
        if identifiable:
            try:
                functional = str(oid.functional())
            except Exception:
                # Functional extraction is best-effort; the id() result above
                # is what callers actually use. Silent fallback is correct.
                functional = None
        return identifiable, functional, None
    except (ValueError, TypeError, KeyError, AssertionError) as e:
        reason = (f"ananke OneLineID failed for X={X_names}, Y={Y_name} "
                  f"({type(e).__name__}: {e})")
        warnings.warn(reason + "; deferring to estimator cascade.",
                      RuntimeWarning, stacklevel=2)
        return True, None, reason


def _expand_admg_to_dag(admg_dict):
    """
    Convert an ADMG to a DAG with one explicit latent common-cause node per
    bidirected edge.  Used solely to reuse the BD/FD/g-computation solver.

    Returns
    -------
    dag : nx.DiGraph
    observed : set
        Observable (non-latent) vertices of the expanded DAG.
    latent_to_bi : dict
        name -> frozenset({u, v}) for each introduced latent.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(admg_dict['vertices'])
    for u, v in admg_dict['di']:
        dag.add_edge(u, v)
    observed = set(admg_dict['vertices'])
    latent_to_bi = {}
    for i, e in enumerate(sorted(admg_dict['bi'],
                                 key=lambda s: tuple(sorted(
                                     _admg_cluster_name(x) if isinstance(x, frozenset) else str(x)
                                     for x in tuple(s))))):
        a, b = tuple(e)
        latent = f'__U_bi_{i}__'
        dag.add_node(latent)
        dag.add_edge(latent, a)
        dag.add_edge(latent, b)
        latent_to_bi[latent] = frozenset({a, b})
    return dag, observed, latent_to_bi


def _cluster_to_fine_vars(cluster, obs_cols, exclude=()):
    """Fine-grained variable members of a cluster that exist in the data."""
    if isinstance(cluster, frozenset):
        members = sorted(cluster)
    else:
        members = [cluster]
    return [v for v in members if v in obs_cols and v not in exclude]


def make_cdag_do_function(cdag_or_admg, intervention_fine_vars, partition,
                          target_var, observational_samples):
    """
    Build a GP-based estimator of ``E[Y | do(X)]`` from the coarsened ADMG.

    Pipeline
    --------
    1. Map fine intervention variables to their cluster vertices ``X`` in
       the C-DAG; let the target cluster be ``Y_cluster`` (typically
       ``frozenset({target_var})``).
    2. Gate: :func:`_ananke_id_check` — run ananke's OneLineID on the
       cluster-level query.  If not identifiable, return ``(None, info)``
       so the CoarsenedGraph falls back to an uninformative prior.
    3. Estimator: expand each bidirected edge to an explicit latent common
       cause, run the legacy BD → FD → g-comp cascade on the resulting DAG,
       and build a GP predictor using fine-grained members of each cluster
       as the conditioning variables.

    Because Lee-2019 partitions are manipulable-only and a cluster
    intervention ``do(C_k)`` assigns all members of ``C_k``, the expansion
    of cluster-level adjustment sets to fine-grained GP inputs is exact,
    not heuristic (every member of an adjustment cluster has observational
    data and is not a descendant of any intervention cluster in the C-DAG).

    Parameters
    ----------
    cdag_or_admg : dict (ADMG) or nx.DiGraph
        Accepts either the new ADMG dict (vertices / di / bi) from
        :func:`coarsening.build_coarsened_admg`, or a legacy nx.DiGraph
        (treated as an ADMG with no bidirected edges).
    intervention_fine_vars : list of str
        Fine-grained manipulable variables to intervene on.
    partition : list of frozenset
        Partition used to build the C-DAG (clusters over observable
        manipulable variables; target ``{Y}`` is always a singleton).
    target_var : str
    observational_samples : pandas.DataFrame

    Returns
    -------
    compute_do : callable or None
    ident : dict
        Includes keys ``method``, ``identifiable``, ``functional`` (ananke
        symbolic expression) and strategy-specific fields.
    """
    # ---- Normalise input to an ADMG dict --------------------------------
    if _is_admg_dict(cdag_or_admg):
        admg = cdag_or_admg
    else:
        admg = _cdag_to_admg_dict(cdag_or_admg)

    node_to_cluster = {v: c for c in partition for v in c}

    # Intervention clusters (preserve fine-var order)
    seen = set()
    intervention_clusters = []
    for v in intervention_fine_vars:
        c = node_to_cluster.get(v)
        if c is not None and c not in seen:
            seen.add(c)
            intervention_clusters.append(c)

    target_cluster = node_to_cluster.get(target_var, frozenset({target_var}))
    if not intervention_clusters or target_cluster not in admg['vertices']:
        return None, {'method': 'none', 'identifiable': False,
                      'reason': 'target_or_intervention_not_in_cdag'}

    # ---- Ananke GID-PO identifiability gate -----------------------------
    identifiable, functional, gate_deferred = _ananke_id_check(
        admg, intervention_clusters, target_cluster)
    if not identifiable:
        return None, {
            'method': 'none',
            'identifiable': False,
            'functional': None,
            'reason': 'not_identifiable_on_cdag',
        }

    # ---- Build expanded DAG and reuse BD/FD/g-comp cascade --------------
    expanded_dag, observed, latent_to_bi = _expand_admg_to_dag(admg)
    ident = identify_adjustment(
        expanded_dag, list(intervention_clusters), target_cluster, observed)

    obs_cols = set(observational_samples.columns)

    if ident['method'] == 'backdoor':
        Z_clusters = ident['adjustment_set']
        Z_fine_vars = []
        for c in Z_clusters:
            Z_fine_vars.extend(_cluster_to_fine_vars(
                c, obs_cols, exclude=intervention_fine_vars + [target_var]))
        predictor = _build_backdoor_predictor(
            intervention_fine_vars, Z_fine_vars, target_var,
            observational_samples)

        def compute_do(obs, functions, value):
            return predictor(value)

        return compute_do, {
            'method': 'backdoor_cdag',
            'identifiable': True,
            'functional': functional,
            'gate_deferred': gate_deferred,
            'adjustment_clusters': [_admg_cluster_name(c) for c in Z_clusters],
            'adjustment_fine_vars': Z_fine_vars,
        }

    if ident['method'] == 'frontdoor':
        M_clusters = ident['adjustment_set']
        M_fine_vars = []
        for c in M_clusters:
            M_fine_vars.extend(_cluster_to_fine_vars(
                c, obs_cols, exclude=intervention_fine_vars + [target_var]))
        predictor = _build_frontdoor_predictor(
            intervention_fine_vars, M_fine_vars, target_var,
            observational_samples)

        def compute_do(obs, functions, value):
            return predictor(value)

        return compute_do, {
            'method': 'frontdoor_cdag',
            'identifiable': True,
            'functional': functional,
            'gate_deferred': gate_deferred,
            'mediator_clusters': [_admg_cluster_name(c) for c in M_clusters],
            'mediator_fine_vars': M_fine_vars,
        }

    if ident['method'] == 'gcomputation':
        # Cluster-level propagation order -> fine-grained GPs.
        predictor = _build_cdag_gcomputation_predictor_admg(
            intervention_fine_vars, ident, target_var,
            observational_samples, obs_cols)

        def compute_do(obs, functions, value):
            return predictor(value)

        return compute_do, {
            'method': 'gcomputation_cdag',
            'identifiable': True,
            'functional': functional,
            'gate_deferred': gate_deferred,
        }

    # Ananke said identifiable but our sound (non-complete) cascade missed
    # it — surface this as a diagnostic rather than silently failing.
    return None, {
        'method': 'none',
        'identifiable': True,
        'functional': functional,
        'gate_deferred': gate_deferred,
        'reason': 'ananke_identifiable_but_cascade_missed',
    }


def _build_cdag_gcomputation_predictor_admg(
        S_fine_vars, plan, Y_var, observational_samples, obs_cols):
    """
    G-computation predictor for cluster-level plans on the expanded DAG.

    ``plan['propagation_order']`` contains ``(cluster, parent_clusters)``
    pairs where each cluster is a frozenset of fine-grained observable
    variables (or a latent-expansion node, which we skip — latents have no
    data).  For each cluster we fit one GP per fine-grained member,
    conditioned on the fine-grained members of the parent clusters.
    """
    n = len(observational_samples)

    prop_gps = []
    for cluster, parent_clusters in plan['propagation_order']:
        if isinstance(cluster, str) and cluster.startswith('__'):
            continue  # expansion-latent node — no observational data
        fine_vars = _cluster_to_fine_vars(cluster, obs_cols)
        parent_fine_vars = []
        for p in parent_clusters:
            if isinstance(p, str) and p.startswith('__'):
                continue
            parent_fine_vars.extend(_cluster_to_fine_vars(p, obs_cols))
        if not parent_fine_vars or not fine_vars:
            prop_gps.append((fine_vars, parent_fine_vars, []))
            continue
        X_parent_obs = np.column_stack(
            [observational_samples[v].values for v in parent_fine_vars])
        cluster_gps = []
        for var in fine_vars:
            Y_var_obs = observational_samples[var].values[:, np.newaxis]
            cluster_gps.append(_fit_gp(X_parent_obs, Y_var_obs))
        prop_gps.append((fine_vars, parent_fine_vars, cluster_gps))

    # Conditioning set for Y
    cond_fine_vars = []
    for c in plan['conditioning_vars']:
        if isinstance(c, str) and c.startswith('__'):
            continue
        cond_fine_vars.extend(_cluster_to_fine_vars(
            c, obs_cols, exclude=[Y_var]))

    if cond_fine_vars:
        X_cond_obs = np.column_stack(
            [observational_samples[v].values for v in cond_fine_vars])
    else:
        X_cond_obs = np.ones((n, 1))
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]
    gp_y = _fit_gp(X_cond_obs, Y_obs)

    obs_values = {col: observational_samples[col].values.copy()
                  for col in observational_samples.columns}

    def predict(value):
        value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
        current_values = {k: v.copy() for k, v in obs_values.items()}
        for i, s_var in enumerate(S_fine_vars):
            current_values[s_var] = np.full(n, value[i])
        for (fine_vars, parent_fine_vars, cluster_gps) in prop_gps:
            if not cluster_gps or not parent_fine_vars:
                continue
            X_parent_current = np.column_stack(
                [current_values[v] for v in parent_fine_vars])
            for var, gp_var in zip(fine_vars, cluster_gps):
                pred_mu, _ = gp_var.predict(X_parent_current)
                current_values[var] = pred_mu.ravel()
        if cond_fine_vars:
            X_cond_current = np.column_stack(
                [current_values[v] for v in cond_fine_vars])
        else:
            X_cond_current = np.ones((n, 1))
        mu, var = gp_y.predict(X_cond_current)
        return float(np.mean(mu)), float(np.mean(var) + np.var(mu))

    return predict


# ---------------------------------------------------------------------------
# Slow versions (fit GP at each call, for backward compatibility)
# ---------------------------------------------------------------------------

def _compute_backdoor_slow(S_vars, Z_vars, Y_var, value, observational_samples):
    value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    n = len(observational_samples)
    input_vars = S_vars + Z_vars
    X_obs = np.column_stack([observational_samples[v].values for v in input_vars])
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]
    gp = _fit_gp(X_obs, Y_obs)
    Z_obs = np.column_stack([observational_samples[v].values for v in Z_vars]) if Z_vars else np.empty((n, 0))
    S_repeated = np.tile(value, (n, 1))
    X_pred = np.hstack([S_repeated, Z_obs]) if Z_vars else S_repeated
    mu, var = gp.predict(X_pred)
    return float(np.mean(mu)), float(np.mean(var) + np.var(mu))


def _compute_frontdoor_slow(S_vars, M_vars, Y_var, value, observational_samples):
    value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    n = len(observational_samples)
    S_obs = np.column_stack([observational_samples[v].values for v in S_vars])
    m_preds = []
    for m_var in M_vars:
        M_obs = observational_samples[m_var].values[:, np.newaxis]
        gp_m = _fit_gp(S_obs, M_obs)
        m_pred, _ = gp_m.predict(value.reshape(1, -1))
        m_preds.append(float(m_pred[0, 0]))
    m_hat = np.array(m_preds)
    input_vars_ym = M_vars + S_vars
    X_obs_ym = np.column_stack([observational_samples[v].values for v in input_vars_ym])
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]
    gp_y = _fit_gp(X_obs_ym, Y_obs)
    M_repeated = np.tile(m_hat, (n, 1))
    X_pred = np.hstack([M_repeated, S_obs])
    mu, var = gp_y.predict(X_pred)
    return float(np.mean(mu)), float(np.mean(var) + np.var(mu))


def _compute_gcomputation_slow(S_vars, plan, Y_var, value, observational_samples):
    value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    n = len(observational_samples)
    current_values = {col: observational_samples[col].values.copy()
                      for col in observational_samples.columns}
    for i, s_var in enumerate(S_vars):
        current_values[s_var] = np.full(n, value[i])
    for var, parents in plan['propagation_order']:
        if not parents:
            continue
        X_parent_obs = np.column_stack([observational_samples[p].values for p in parents])
        Y_var_obs = observational_samples[var].values[:, np.newaxis]
        gp_var = _fit_gp(X_parent_obs, Y_var_obs)
        X_parent_current = np.column_stack([current_values[p] for p in parents])
        pred_mu, _ = gp_var.predict(X_parent_current)
        current_values[var] = pred_mu.ravel()
    cond_vars = plan['conditioning_vars']
    X_cond_obs = np.column_stack([observational_samples[v].values for v in cond_vars])
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]
    gp_y = _fit_gp(X_cond_obs, Y_obs)
    X_cond_current = np.column_stack([current_values[v] for v in cond_vars])
    mu, var = gp_y.predict(X_cond_current)
    return float(np.mean(mu)), float(np.mean(var) + np.var(mu))
