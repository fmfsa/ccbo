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

(B) The legacy fine-grained DAG pipeline (``make_adjustment_do_function``
    and its ``_compute_*_slow`` duplicates) was removed in engine v3
    (2026-09): it had no callers and duplicated the estimators below with
    mean-substitution semantics.

Estimator semantics (engine v3): see the "Pre-fitted causal effect
estimation" section.  Intermediate conditionals are *integrated* (Gauss-
Hermite for front-door mediators, seeded common-random-number draws for
g-computation), never replaced by their mean; the prior variance is the
CBO-faithful ``Var[Y | do]`` with the GP's epistemic term excluded.

GPs are fitted ONCE when the do-function is created, not at each evaluation.
"""

import os
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
# 2. Backdoor and frontdoor criterion
# ---------------------------------------------------------------------------

def _vkey(v):
    """Canonical sort key for DAG / ADMG vertices (strings or frozenset
    clusters).  Every candidate enumeration and every returned variable
    order in this module goes through it, so identification and the GP
    input layout are independent of Python's per-process hash seed
    (``sorted()`` on frozensets is a partial order and is NOT canonical)."""
    if isinstance(v, frozenset):
        return (1, tuple(sorted(str(x) for x in v)))
    return (0, (str(v),))


def _sorted_v(vs):
    return sorted(vs, key=_vkey)


def find_backdoor_set(dag, S, Y, observed):
    """
    Find a minimal valid backdoor adjustment set Z for do(S) on Y.
    """
    S_set = set(S)
    desc_S = set()
    for s in S_set:
        desc_S |= nx.descendants(dag, s)
    desc_S -= S_set

    candidates = _sorted_v(observed - S_set - desc_S - {Y})

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
                return _sorted_v(Z_set)

    return None


def find_frontdoor_set(dag, S, Y, observed):
    """
    Find a valid frontdoor adjustment set M for do(S) on Y.
    """
    S_set = set(S)
    candidates = _sorted_v(observed - S_set - {Y})

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

            return _sorted_v(M_set)

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
    obs_dag.add_nodes_from(_sorted_v(observed))
    for u, v in _sorted_v((u, v) for u, v in dag.edges()
                          if u in observed and v in observed):
        obs_dag.add_edge(u, v)

    topo = list(nx.lexicographical_topological_sort(obs_dag, key=_vkey))

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
        obs_parents = _sorted_v(set(obs_dag.predecessors(v)))
        
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

    obs_parents_Y = _sorted_v(set(obs_dag.predecessors(Y)))
    
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

    final_conditioning = _sorted_v(
        set(obs_parents_Y) | (observed - S_set - {Y} - affected)
    )

    return {
        'method': 'gcomputation',
        'propagation_order': propagation,
        'conditioning_vars': final_conditioning,
        'affected_vars': _sorted_v(affected - {Y}),
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
#
# Estimator policy (engine v3, 2026-09).  Every predictor returns
# ``(mean, v)``: ``mean`` is the plug-in estimate of ``E[Y | do(A=a)]`` and
# ``v`` the prior variance handed to the arm surrogate's CausalRBF kernel.
# ``moments()`` exposes the full decomposition over the mixture of
# adjustment / mediator / propagation rows (weights ``w_i``):
#
#     spread     = sum_i w_i (mu_i - mean)^2        Var_mix(E[Y | pa])
#     lik_var    = fitted noise of the outcome GP   E_mix[Var(Y | pa)]
#     epistemic  = sum_i w_i var_f(pa_i)            GP posterior variance
#
# VARIANCE_POLICY selects ``v``:
#
#   "predictive" (default): v = lik_var + spread + epistemic -- the law of
#       total variance of Y under do(a) *under the fitted plug-in model*,
#       integrating the GP posterior over the outcome regression.  This is the
#       quantity the CBO reference implementation averages
#       (``np.mean(gp.predict(rows)[1])`` = epistemic + noise) plus the
#       between-row spread that a mixture over adjustment rows adds.
#   "total": v = lik_var + spread -- the plug-in Var[Y | do(a)] with the GP
#       epistemic term removed.  Kept for comparison runs: off the
#       observational support (or for an interpolating outcome fit) it
#       collapses to ~0, which makes the CausalRBF kernel identically zero
#       and the arm surrogate unable to learn from its own pulls (the
#       2026-09-09 preliminary family run: flat Coral trajectories).
#   "epistemic": diagnostic only.
#
# Override with the environment variable ``CCBO_VARIANCE_POLICY`` *before*
# import; the value is recorded in every decision-log sidecar.
# Conditional laws of intermediate variables are the fitted plug-in
# Gaussians ``N(mu_hat(pa), sigma_hat^2)``; they are integrated by
# Gauss-Hermite quadrature (front-door) or by seeded common-random-number
# draws (g-computation) -- never by substituting their mean, which is
# biased for any nonlinear outcome mechanism.

GH_NODES = 20            # Gauss-Hermite nodes per mediator (product grid <= 2)
MC_DRAWS = 400           # seeded draws when there are more than two mediators
GCOMP_REPS = 4           # replications of the observational rows in g-comp draws
ESTIMATOR_SEED = 0       # fixed: part of the quotient-contract coupling
VARIANCE_POLICIES = ("predictive", "total", "epistemic")
VARIANCE_POLICY = os.environ.get("CCBO_VARIANCE_POLICY", "predictive")
if VARIANCE_POLICY not in VARIANCE_POLICIES:
    raise ValueError(f"CCBO_VARIANCE_POLICY={VARIANCE_POLICY!r}; "
                     f"expected one of {VARIANCE_POLICIES}")

ESTIMATOR_INFO = {
    'gh_nodes': GH_NODES, 'mc_draws': MC_DRAWS, 'gcomp_reps': GCOMP_REPS,
    'seed': ESTIMATOR_SEED, 'variance_policy': VARIANCE_POLICY,
    'mediator_law': 'plug-in Gaussian (GP mean, likelihood variance)',
}


def _gp_moments(gp, X):
    """Latent posterior mean / variance (no likelihood) and the fitted
    likelihood variance of a GPy regression model."""
    mu, var_f = gp.predict(np.asarray(X, dtype=float), include_likelihood=False)
    return mu, var_f, float(gp.likelihood.variance)


def _gp_conditional(gp):
    """Plug-in conditional law ``N(mu_hat(x), sigma_hat^2)`` of a fitted GP,
    as a callable ``X -> (mu, var_total)`` with constant noise variance."""
    def cond(X):
        mu, _ = gp.predict(np.asarray(X, dtype=float), include_likelihood=False)
        return mu, np.full_like(mu, float(gp.likelihood.variance))
    return cond


def _gp_outcome(gp):
    """Outcome regression as a callable ``X -> (mu, var_f, lik_var)``."""
    return lambda X: _gp_moments(gp, X)


def _mixture_moments(mu, var_f, lik_var, weights, n_obs):
    """Moments of ``Y`` under an identified functional evaluated on a
    weighted mixture of conditioning rows (``weights`` sum to one)."""
    mu = np.ravel(np.asarray(mu, dtype=float))
    var_f = np.ravel(np.asarray(var_f, dtype=float))
    w = np.ravel(np.asarray(weights, dtype=float))
    mean = float(np.sum(w * mu))
    spread = float(np.sum(w * (mu - mean) ** 2))
    epistemic = float(np.sum(w * var_f))
    aleatoric = float(lik_var) + spread
    sampling = spread / max(int(n_obs), 1)
    if VARIANCE_POLICY == "predictive":
        v = aleatoric + epistemic
    elif VARIANCE_POLICY == "total":
        v = aleatoric
    elif VARIANCE_POLICY == "epistemic":
        v = epistemic + sampling
    else:
        raise ValueError(f"unknown VARIANCE_POLICY {VARIANCE_POLICY!r}")
    return {'mean': mean, 'v': float(max(v, 0.0)), 'aleatoric': aleatoric,
            'epistemic': epistemic, 'sampling': float(sampling),
            'spread': spread, 'lik_var': float(lik_var)}


def _attach(predict, moments):
    """Expose the full moment breakdown on the fast predictor."""
    predict.moments = moments
    return predict


def _mediator_grid(value, m_conditionals, n_nodes=GH_NODES,
                   mc_draws=MC_DRAWS, seed=ESTIMATOR_SEED):
    """Nodes / weights integrating the chain-rule mediator law
    ``prod_j p(m_j | value, m_<j)``: a Gauss-Hermite product grid for at
    most two mediators, seeded common-random-number draws beyond.
    ``n_nodes=1`` degenerates to mean substitution (diagnostic only)."""
    value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    n_med = len(m_conditionals)
    if n_med == 0:
        return np.zeros((1, 0)), np.ones(1)
    if n_med <= 2:
        xi, w = np.polynomial.hermite.hermgauss(int(n_nodes))
        w = w / np.sqrt(np.pi)
        nodes, weights = np.zeros((1, 0)), np.ones(1)
        for cond in m_conditionals:
            R = nodes.shape[0]
            mu_j, var_j = cond(np.hstack([np.tile(value, (R, 1)), nodes]))
            mu_j = np.ravel(mu_j)
            sd_j = np.sqrt(np.maximum(np.ravel(var_j), 0.0))
            m_col = (mu_j[:, None] + np.sqrt(2.0) * sd_j[:, None] * xi[None, :]
                     ).reshape(-1, 1)
            nodes = np.hstack([np.repeat(nodes, len(xi), axis=0), m_col])
            weights = (weights[:, None] * w[None, :]).reshape(-1)
        return nodes, weights
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((int(mc_draws), n_med))
    nodes = np.zeros((int(mc_draws), 0))
    for j, cond in enumerate(m_conditionals):
        mu_j, var_j = cond(np.hstack([np.tile(value, (int(mc_draws), 1)), nodes]))
        m_j = np.ravel(mu_j) + np.sqrt(np.maximum(np.ravel(var_j), 0.0)) * Z[:, j]
        nodes = np.hstack([nodes, m_j[:, None]])
    return nodes, np.full(int(mc_draws), 1.0 / int(mc_draws))


def frontdoor_integrate(value, m_conditionals, y_conditional, S_obs,
                        n_nodes=GH_NODES, mc_draws=MC_DRAWS,
                        seed=ESTIMATOR_SEED):
    """Front-door functional with the mediator law integrated out:

        E[Y | do(S=s)] = int p(m | s) [ (1/n) sum_j E[Y | m, S=s_j] ] dm .

    ``m_conditionals`` : list of ``X -> (mu, var_total)``; the j-th receives
    ``[value, m_<j]`` rows (chain rule for vector mediators).
    ``y_conditional``  : ``X -> (mu, var_f, lik_var)`` on ``[M, S]`` rows.
    ``S_obs``          : ``(n, |S|)`` observational rows for the outer average.
    Returns the moment dict of :func:`_mixture_moments`.
    """
    value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    S_obs = np.asarray(S_obs, dtype=float)
    n = S_obs.shape[0]
    nodes, weights = _mediator_grid(value, m_conditionals, n_nodes, mc_draws, seed)
    K = nodes.shape[0]
    X_pred = np.hstack([np.repeat(nodes, n, axis=0), np.tile(S_obs, (K, 1))])
    mu, var_f, lik = y_conditional(X_pred)
    return _mixture_moments(mu, var_f, lik, np.repeat(weights, n) / n, n)


def gcomp_propagate(value, S_vars, steps, y_conditional, cond_vars,
                    base_rows, Z):
    """G-computation by conditional draws with common random numbers.

    ``steps`` is a topologically ordered list ``(var, parent_vars, cond)``
    with ``cond : X -> (mu, var_total)`` the plug-in conditional of ``var``
    given ``parent_vars`` (earlier members of the same cluster appear among
    the parents of later members: chain rule, so within-cluster dependence
    is preserved).  ``base_rows`` maps every observational column to its
    ``(n,)`` array; ``Z`` has shape ``(reps * n, len(steps))`` and is drawn
    once, so the estimator is deterministic and continuous in ``value``.
    """
    value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    n = len(next(iter(base_rows.values())))
    R = int(Z.shape[0])
    reps = R // n
    current = {k: np.tile(np.asarray(v, dtype=float), reps)
               for k, v in base_rows.items()}
    for i, s in enumerate(S_vars):
        current[s] = np.full(R, value[i])
    for k, (var, parents, cond) in enumerate(steps):
        mu, var_t = cond(np.column_stack([current[p] for p in parents]))
        current[var] = (np.ravel(mu)
                        + np.sqrt(np.maximum(np.ravel(var_t), 0.0)) * Z[:, k])
    X_cond = (np.column_stack([current[v] for v in cond_vars])
              if cond_vars else np.ones((R, 1)))
    mu, var_f, lik = y_conditional(X_cond)
    return _mixture_moments(mu, var_f, lik, np.full(R, 1.0 / R), n)


def _build_backdoor_predictor(S_vars, Z_vars, Y_var, observational_samples):
    """Pre-fit one GP ``Y | S, Z`` and return ``value -> (mean, v)``."""
    n = len(observational_samples)
    X_obs = np.column_stack(
        [observational_samples[v].values for v in S_vars + Z_vars])
    Y_obs = observational_samples[Y_var].values[:, np.newaxis]
    gp = _fit_gp(X_obs, Y_obs)
    Z_obs = (np.column_stack([observational_samples[v].values for v in Z_vars])
             if Z_vars else np.empty((n, 0)))
    y_cond = _gp_outcome(gp)

    def moments(value):
        value = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
        mu, var_f, lik = y_cond(np.hstack([np.tile(value, (n, 1)), Z_obs]))
        return _mixture_moments(mu, var_f, lik, np.full(n, 1.0 / n), n)

    def predict(value):
        m = moments(value)
        return m['mean'], m['v']

    return _attach(predict, moments)


def _build_frontdoor_predictor(S_vars, M_vars, Y_var, observational_samples,
                               n_nodes=GH_NODES):
    """Pre-fit the chain-rule mediator GPs ``M_j | S, M_<j`` and the outcome
    GP ``Y | M, S``; return ``value -> (mean, v)`` integrating over the
    fitted mediator law by Gauss-Hermite quadrature.  ``n_nodes=1``
    reproduces the legacy mean-substitution estimator (diagnostics only)."""
    S_obs = np.column_stack([observational_samples[v].values for v in S_vars])
    m_conditionals = []
    for j, m_var in enumerate(M_vars):
        X_in = np.column_stack(
            [observational_samples[v].values for v in S_vars + M_vars[:j]])
        m_conditionals.append(_gp_conditional(
            _fit_gp(X_in, observational_samples[m_var].values[:, np.newaxis])))
    X_obs_ym = np.column_stack(
        [observational_samples[v].values for v in M_vars + S_vars])
    gp_y = _fit_gp(X_obs_ym, observational_samples[Y_var].values[:, np.newaxis])
    y_cond = _gp_outcome(gp_y)

    def moments(value):
        return frontdoor_integrate(value, m_conditionals, y_cond, S_obs,
                                   n_nodes=n_nodes)

    def predict(value):
        m = moments(value)
        return m['mean'], m['v']

    return _attach(predict, moments)

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
    status : {"identified", "not_identified", "error"}
        Tri-state verdict.  ``"error"`` means ananke raised at runtime; the
        caller must treat it as *not usable* (fail-closed) and record it
        separately from a negative identifiability result.
    functional : str or None
        Symbolic Tian-Pearl factorisation as returned by ananke (diagnostic).
    reason : str or None
        Diagnostic text for the ``"error"`` state, else ``None``.
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
        warnings.warn(reason + "; gate status = error (fail-closed).",
                      RuntimeWarning, stacklevel=2)
        return "error", None, reason

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
                # is what callers actually use.
                functional = None
        return ("identified" if identifiable else "not_identified"), functional, None
    except (ValueError, TypeError, KeyError, AssertionError) as e:
        reason = (f"ananke OneLineID failed for X={X_names}, Y={Y_name} "
                  f"({type(e).__name__}: {e})")
        warnings.warn(reason + "; gate status = error (fail-closed).",
                      RuntimeWarning, stacklevel=2)
        return "error", None, reason

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
    dag.add_nodes_from(_sorted_v(admg_dict['vertices']))
    for u, v in _sorted_v(admg_dict['di']):
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
       cluster-level query.  ``not_identified`` and ``error`` both return
       ``(None, info)`` so the CoarsenedGraph falls back to the constant
       observational prior; the two states are recorded separately in
       ``info['gate_status']`` (an ``error`` is a fail-closed diagnostic,
       never a positive identification).
    3. Estimator: expand each bidirected edge to an explicit latent common
       cause, run the BD → FD → g-comp cascade on the resulting DAG, and
       build a GP predictor using fine-grained members of each cluster as
       the conditioning variables (front-door / g-comp conditionals are
       integrated, see the estimator section).

    Returns
    -------
    compute_do : callable or None
    ident : dict
        Keys ``method``, ``identifiable``, ``gate_status``, ``functional``,
        ``estimator`` and strategy-specific fields.
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
                      'gate_status': 'not_run',
                      'reason': 'target_or_intervention_not_in_cdag'}

    # ---- Ananke GID-PO identifiability gate (tri-state, fail-closed) ----
    status, functional, gate_reason = _ananke_id_check(
        admg, intervention_clusters, target_cluster)
    if status == "error":
        return None, {
            'method': 'none', 'identifiable': False, 'functional': None,
            'gate_status': 'error', 'reason': gate_reason,
        }
    if status != "identified":
        return None, {
            'method': 'none', 'identifiable': False, 'functional': None,
            'gate_status': 'not_identified',
            'reason': 'not_identifiable_on_cdag',
        }

    # ---- Build expanded DAG and reuse BD/FD/g-comp cascade --------------
    expanded_dag, observed, latent_to_bi = _expand_admg_to_dag(admg)
    ident = identify_adjustment(
        expanded_dag, list(intervention_clusters), target_cluster, observed)

    obs_cols = set(observational_samples.columns)
    base = {'identifiable': True, 'functional': functional,
            'gate_status': 'identified', 'estimator': dict(ESTIMATOR_INFO)}

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
        compute_do.moments = predictor.moments

        return compute_do, dict(base, method='backdoor_cdag',
                                adjustment_clusters=[_admg_cluster_name(c) for c in Z_clusters],
                                adjustment_fine_vars=Z_fine_vars)

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
        compute_do.moments = predictor.moments

        return compute_do, dict(base, method='frontdoor_cdag',
                                mediator_clusters=[_admg_cluster_name(c) for c in M_clusters],
                                mediator_fine_vars=M_fine_vars)

    if ident['method'] == 'gcomputation':
        predictor = _build_cdag_gcomputation_predictor_admg(
            intervention_fine_vars, ident, target_var,
            observational_samples, obs_cols)

        def compute_do(obs, functions, value):
            return predictor(value)
        compute_do.moments = predictor.moments

        return compute_do, dict(base, method='gcomputation_cdag')

    # Ananke said identifiable but our sound (non-complete) cascade missed
    # it — surface this as a diagnostic rather than silently failing.
    return None, {
        'method': 'none', 'identifiable': True, 'functional': functional,
        'gate_status': 'identified',
        'reason': 'ananke_identifiable_but_cascade_missed',
    }


def _build_cdag_gcomputation_predictor_admg(
        S_fine_vars, plan, Y_var, observational_samples, obs_cols,
        reps=GCOMP_REPS, seed=ESTIMATOR_SEED):
    """
    G-computation predictor for cluster-level plans on the expanded DAG.

    ``plan['propagation_order']`` contains ``(cluster, parent_clusters)``
    pairs where each cluster is a frozenset of fine-grained observable
    variables (or a latent-expansion node, which is skipped — latents have
    no data).  Every propagated variable gets a plug-in conditional GP given
    the fine-grained members of its parent clusters *and* the earlier
    members of its own cluster (chain rule).  Prediction integrates those
    conditionals with seeded common-random-number draws
    (:func:`gcomp_propagate`) instead of propagating means.
    """
    n = len(observational_samples)
    steps = []
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
            continue
        for j, var in enumerate(fine_vars):
            parents = parent_fine_vars + fine_vars[:j]
            X_in = np.column_stack(
                [observational_samples[v].values for v in parents])
            gp = _fit_gp(X_in, observational_samples[var].values[:, np.newaxis])
            steps.append((var, parents, _gp_conditional(gp)))

    cond_fine_vars = []
    for c in plan['conditioning_vars']:
        if isinstance(c, str) and c.startswith('__'):
            continue
        cond_fine_vars.extend(_cluster_to_fine_vars(
            c, obs_cols, exclude=[Y_var]))
    X_cond_obs = (np.column_stack(
        [observational_samples[v].values for v in cond_fine_vars])
        if cond_fine_vars else np.ones((n, 1)))
    gp_y = _fit_gp(X_cond_obs, observational_samples[Y_var].values[:, np.newaxis])
    y_cond = _gp_outcome(gp_y)

    base_rows = {col: observational_samples[col].values.copy()
                 for col in observational_samples.columns}
    Z = np.random.default_rng(seed).standard_normal(
        (int(reps) * n, max(len(steps), 1)))

    def moments(value):
        return gcomp_propagate(value, S_fine_vars, steps, y_cond,
                               cond_fine_vars, base_rows, Z)

    def predict(value):
        m = moments(value)
        return m['mean'], m['v']

    predict.steps = [(var, list(parents)) for var, parents, _ in steps]
    return _attach(predict, moments)
