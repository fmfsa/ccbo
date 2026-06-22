"""
CoarsenedGraph: a wrapper that presents a coarsened view of a causal DAG
to the CBO algorithm.

Implements the GraphStructure interface so it plugs directly into CBO().
"""

import os
import itertools
import numpy as np
import pandas as pd
from collections import OrderedDict

from ccbo.cbo.graphs.graph import GraphStructure
from ccbo.cbo.utils import fit_single_GP_model

from .adjustment import make_cdag_do_function, _ananke_id_check
from .coarsening import (get_dag_edges_from_sem, get_hidden_confounders,
                          project_out_hidden, build_coarsened_admg,
                          compute_POMIS, _admg_is_acyclic)


class CoarsenedGraph(GraphStructure):
    """
    A coarsened version of an original CBO graph.

    Given a partition of the DAG's nodes, this class derives the
    exploration set, do-calculus functions, and other CBO components
    from the coarsened structure.

    Parameters
    ----------
    original_graph : GraphStructure
        The fine-grained graph instance (e.g., ToyGraph, CompleteGraph).
    partition : list of frozenset
        The partition of observable nodes (including {Y} as a singleton).
    graph_name : str
        Name of the graph ('ToyGraph' or 'CompleteGraph').
    observational_samples : pd.DataFrame
        Observational data.
    max_intervention_size : int, optional
        Maximum number of coarsened manipulative nodes to intervene on
        simultaneously. Defaults to 3.
    num_mc_samples : int, optional
        Number of MC samples for do-calculus. Defaults to 10000.
    assumed_graph_name : str, optional
        Name of the assumed graph structure for adjustment formulas.
        If None, uses graph_name (correct graph). Set to a different
        name (e.g., 'CompleteGraph_NoCD') to simulate graph misspecification.
    """

    def __init__(self, original_graph, partition, graph_name,
                 observational_samples, max_intervention_size=3,
                 num_mc_samples=10000, assumed_graph_name=None):
        self.original_graph = original_graph
        self.partition = partition
        self.graph_name = graph_name
        self.assumed_graph_name = assumed_graph_name or graph_name
        self.num_mc_samples = num_mc_samples
        self.max_intervention_size = max_intervention_size

        # Store observational data columns
        self._obs_samples = observational_samples
        for col in observational_samples.columns:
            setattr(self, col, np.asarray(observational_samples[col])[:, np.newaxis])

        # Get graph metadata (from assumed graph for structural decisions)
        _, _, _, self._fine_manip_vars = get_dag_edges_from_sem(self.assumed_graph_name)
        self._fine_ranges = original_graph.get_interventional_ranges()

        # Build coarsened structure
        self._build_coarsened_structure()

    def _build_coarsened_structure(self):
        """
        Derive the exploration set from a manipulable-only partition via the
        Lee-2019 latent projection + POMIS pipeline.

        Steps
        -----
        1. **Validate** that every non-singleton cluster contains only
           manipulable variables (mixing manipulable and non-manipulable
           vertices in one cluster is forbidden — see Plan §4.1).
        2. Build the projected ADMG ``G* = proj(G, V\\U)`` over all
           observable vertices by projecting out hidden confounders.
        3. Build the coarsened ADMG ``G^π`` as the quotient of ``G*`` under
           ``π``, lifting non-manipulable observables to singleton atomic
           vertices.  Verify acyclicity.
        4. Compute POMIS on ``G^π`` with the manipulable cluster vertices
           as the action space.  Each POMIS element is a set of cluster
           vertices that are *jointly* intervened on.
        5. Flatten each POMIS element to its fine-grained manipulable
           members (``do(C_k) =`` joint assignment over all of ``C_k``).
        """
        # (1) Validate: non-singleton clusters must be manipulable-only
        self._manip_coarsened_nodes = []
        self._node_to_manip_vars = {}
        fine_manip_set = set(self._fine_manip_vars)
        for part in self.partition:
            if part == frozenset({'Y'}):
                continue
            if not part.issubset(fine_manip_set):
                # Either a non-manip singleton atom (fine) or a mixed cluster
                # (forbidden).  Detect the latter.
                if len(part) > 1:
                    raise ValueError(
                        f"Partition cluster {set(part)} mixes manipulable "
                        f"and non-manipulable variables; Lee-2019 CCBO "
                        f"requires manipulable-only clusters.")
                continue  # valid singleton non-manip atom
            # Pure manipulable cluster
            self._manip_coarsened_nodes.append(part)
            self._node_to_manip_vars[part] = sorted(part)

        # (2) Lee-2019 projected ADMG (observables only; U absorbed as bi)
        projected_admg = project_out_hidden(self.assumed_graph_name)

        # (3) Coarsened ADMG quotient; non-manipulable observables become
        # singleton atomic vertices alongside manipulable clusters.
        _, all_nodes, hidden, _ = get_dag_edges_from_sem(self.assumed_graph_name)
        hidden = set(hidden or [])
        nonmanip_observables = (set(all_nodes) - hidden
                                 - fine_manip_set - {'Y'})
        self._coarsened_admg = build_coarsened_admg(
            self.partition, projected_admg,
            atomic_vertices=nonmanip_observables)
        # Backwards-compat alias (callers previously accessed ._coarsened_dag)
        self._coarsened_dag = self._coarsened_admg
        if not _admg_is_acyclic(self._coarsened_admg):
            raise ValueError(
                f"Coarsening {self.partition} induces a cyclic C-DAG.")

        # (4) POMIS on the coarsened ADMG (for theoretical guarantees)
        target_node = frozenset({'Y'})
        self._pomis_sets = compute_POMIS(
            self._coarsened_admg, self._manip_coarsened_nodes, target_node)

        # (5) Exploration set = all identifiable non-empty subsets of
        #     manipulable clusters (up to max_intervention_size).
        #     POMIS gives the optimality guarantee; the broader identifiable
        #     set speeds up convergence by also exploring cheaper 1-cluster
        #     and 2-cluster interventions that ananke confirms are identified.
        seen = set()
        exploration_tuples = []

        def _add_combo(combo):
            key = frozenset(combo)
            if key in seen or len(key) == 0 or len(key) > self.max_intervention_size:
                return
            ok, _, _ = _ananke_id_check(self._coarsened_admg, set(combo), target_node)
            if ok:
                seen.add(key)
                exploration_tuples.append(combo)

        M = self._manip_coarsened_nodes
        for r in range(1, min(len(M), self.max_intervention_size) + 1):
            for combo in itertools.combinations(M, r):
                _add_combo(frozenset(combo))

        # Always include every POMIS element as well (it is identifiable by
        # definition, but may exceed max_intervention_size limit above).
        for combo in self._pomis_sets:
            _add_combo(combo)

        self._exploration_set = []
        for combo in exploration_tuples:
            fine_vars = []
            for node in combo:
                fine_vars.extend(self._node_to_manip_vars[node])
            fine_vars = sorted(fine_vars)
            # Invariant: every exploration-set entry intervenes on FULL
            # clusters (do(C_k) assigns all members of C_k). The cluster-
            # level identification produced by make_cdag_do_function is
            # exact for these queries; no sub-cluster heuristic exists.
            assert fine_vars == sorted(v for node in combo for v in node), (
                f"ES entry {fine_vars} is not a union of full clusters "
                f"{[sorted(c) for c in combo]}")
            self._exploration_set.append(fine_vars)

        # Full manipulable variable list for standard BO
        all_manip = set()
        for part in self._manip_coarsened_nodes:
            all_manip.update(self._node_to_manip_vars[part])
        self._manipulative_variables = sorted(all_manip)

    def define_SEM(self):
        """Return the original fine-grained SEM."""
        return self.original_graph.define_SEM()

    def get_sets(self):
        """Return (MIS, POMIS, manipulative_variables) for the coarsened graph."""
        # MIS computed on the coarsened DAG; POMIS ≈ MIS (safe over-approximation)
        return self._exploration_set, self._exploration_set, self._manipulative_variables

    def get_set_BO(self):
        """Return manipulative variables for standard BO."""
        return self._manipulative_variables

    def get_interventional_ranges(self):
        """Return interventional ranges for the manipulative variables."""
        # Return the original ranges for manipulative variables
        ranges = OrderedDict()
        for var in self._manipulative_variables:
            if var in self._fine_ranges:
                ranges[var] = self._fine_ranges[var]
        return ranges

    def fit_all_models(self):
        """Fit conditional GP models (delegate to original graph)."""
        return self.original_graph.fit_all_models()

    def refit_models(self, observational_samples):
        """Refit GP models on new observational data."""
        return self.original_graph.refit_models(observational_samples)

    def get_cost_structure(self, type_cost):
        """Return cost structure (delegate to original graph)."""
        return self.original_graph.get_cost_structure(type_cost)

    def get_all_do(self):
        """
        Return do-calculus functions for each exploration set element.

        For each exploration set entry, attempts C-DAG level identification
        (backdoor / frontdoor / g-computation on the coarsened DAG with
        bidirected edges for hidden confounders).  This makes the GP prior
        **invariant to intra-cluster edge misspecification**.

        When the causal effect is **not identifiable** from the C-DAG (e.g.
        because a hidden confounder spans the intervention cluster and the
        target, and the frontdoor mediator is internal), the GP prior is set
        to an uninformative constant (observational mean of Y).  The BO loop
        then learns the interventional effect purely from experimental data.

        No oracle / true-SEM knowledge is used.  This is the key design
        choice: CCBO operates with C-DAG knowledge + experiments only, with
        the possibility of refining the C-DAG via future interventional data
        (e.g. using the RePaRe algorithm from Madaleno et al. 2026).

        The result is memoized: it is a pure function of the (fixed) C-DAG,
        partition, and initial observational sample, so it never changes over a
        BO run. Callers such as the benchmark CBO loop invoke ``get_all_do``
        once per trial; without the cache this re-fits every cluster GP each
        time, which dominates wall-clock. Build once, reuse.
        """
        if getattr(self, '_do_cache', None) is not None:
            return self._do_cache

        do_dict = {}
        self._identification_info = {}

        # Uninformative prior: observational mean/variance of Y
        y_mean = float(self._obs_samples['Y'].mean())
        y_var = float(self._obs_samples['Y'].var())

        def _make_uninformative_do(y_mu, y_sigma2):
            """Return a do-function with constant (uninformative) prior."""
            def compute_do(obs, functions, value):
                return y_mu, y_sigma2
            return compute_do

        for es_entry in self._exploration_set:
            name = 'compute_do_' + ''.join(es_entry)

            # Identify at the C-DAG level — invariant to intra-cluster edges.
            do_fn, ident = make_cdag_do_function(
                self._coarsened_dag, es_entry, self.partition,
                'Y', self._obs_samples
            )
            self._identification_info[name] = ident

            if do_fn is not None:
                do_dict[name] = do_fn
            else:
                # Non-identifiable from C-DAG: uninformative prior.
                # The BO loop will learn the effect from interventional data.
                ident['_uninformative'] = True
                do_dict[name] = _make_uninformative_do(y_mean, y_var)

        # Also add do function for the full BO set
        full_name = 'compute_do_' + ''.join(self._manipulative_variables)
        if full_name not in do_dict:
            do_fn, ident = make_cdag_do_function(
                self._coarsened_dag, self._manipulative_variables,
                self.partition, 'Y', self._obs_samples
            )
            self._identification_info[full_name] = ident

            if do_fn is not None:
                do_dict[full_name] = do_fn
            else:
                ident['_uninformative'] = True
                do_dict[full_name] = _make_uninformative_do(y_mean, y_var)

        self._do_cache = do_dict
        return do_dict

    def get_identifiability_summary(self):
        """
        Return a summary of which exploration set elements are identifiable
        from the C-DAG and which use uninformative priors.

        Useful for diagnostics and for deciding which clusters to refine
        (e.g. via the RePaRe algorithm from Madaleno et al. 2026).
        """
        if not hasattr(self, '_identification_info'):
            return {}
        summary = {}
        for name, info in self._identification_info.items():
            summary[name] = {
                'method': info.get('method', 'unknown'),
                'identifiable': info.get('method', 'none') != 'none',
                'uninformative': info.get('_uninformative', False),
            }
        return summary

    def marginal_log_likelihood(self):
        """
        Observational-data log-likelihood of the Y-predictor under this
        coarsening's C-DAG identification.

        Used as the RCCBO partition-acceptance criterion (Plan §6.1): a
        refinement ``π → π'`` is accepted iff
        ``marginal_log_likelihood(π') > marginal_log_likelihood(π)``.

        Implementation: fit one GP per POMIS / exploration-set entry
        ``X`` using the cluster-level adjustment set returned by
        :func:`make_cdag_do_function`, and sum the GP marginal
        log-likelihoods.  Falls back to 0 for non-identifiable entries
        (uninformative prior carries no structural information).
        """
        import GPy
        from .adjustment import _fit_gp
        total = 0.0
        obs_cols = set(self._obs_samples.columns)
        Y_obs = np.asarray(self._obs_samples['Y'])[:, np.newaxis]
        for es_entry in self._exploration_set:
            do_fn, ident = make_cdag_do_function(
                self._coarsened_admg, es_entry, self.partition,
                'Y', self._obs_samples)
            if do_fn is None:
                continue
            # Assemble input columns: intervention vars + adjustment vars
            cols = list(es_entry)
            cols += [v for v in ident.get('adjustment_fine_vars', []) if v != 'Y']
            cols += [v for v in ident.get('mediator_fine_vars', []) if v != 'Y']
            cols = [c for c in cols if c in obs_cols]
            if not cols:
                continue
            try:
                X = np.column_stack(
                    [np.asarray(self._obs_samples[c]) for c in cols])
                gp = _fit_gp(X, Y_obs)
                total += float(gp.log_likelihood())
            except Exception:
                continue
        return total

    def get_partition_description(self):
        """Return a human-readable description of the coarsening."""
        parts = []
        for part in self.partition:
            parts.append('{' + ','.join(sorted(part)) + '}')
        return ' | '.join(parts)

    def get_exploration_set_description(self):
        """Return a human-readable description of the exploration set."""
        entries = []
        for es in self._exploration_set:
            entries.append('{' + ','.join(es) + '}')
        return '[' + ', '.join(entries) + ']'
