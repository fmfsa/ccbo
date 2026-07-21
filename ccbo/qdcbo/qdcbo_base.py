"""QDCBO: quotient Dynamic Causal Bayesian Optimisation.

Subclasses the vendored DCBO (third_party/CausalBO_Benchmark/baselines/DCBO)
so that EVERY graph-dependent model component reads only the quotient
temporal graph of a :class:`~ccbo.qdcbo.quotient_dbn.QuotientDBNSpec`:

* parent resolution (``node_parents``) answers with the member-node tuples of
  a cluster's quotient parents;
* arc fitting (:func:`fit_quotient_arcs`, replacing the stock ``fit_arcs``)
  fits ONE mechanism per cluster per arc type — a multi-output GP (GPy ICM /
  ``GPCoregionalizedRegression``) whose inputs are the concatenated
  parent-cluster member columns and whose outputs are the cluster's member
  columns; source clusters get a joint (multivariate) KDE marginal;
* the SEM-hat (:func:`make_qsem_hat`) dispatches per *cluster* and samples all
  member coordinates jointly from the shared multi-output posterior;
* the sufficient-statistics update walks the quotient with
  :func:`sequential_sample_from_qsem_hat` (cluster-major, jointly per
  cluster).

Optimisation-side objects (blankets, target functions, exploration domains)
stay in member coordinates: exploration sets are lifted to whole-cluster
patterns, so the stock acquisition / BO-model machinery runs unchanged on
(possibly multivariate) cluster arms — arms DCBO itself already supports.

Reduction guarantees (tested in ccbo/tests/test_qdcbo.py):
* finest partition (all singletons): the quotient graph, the fitted
  mechanisms (``fit_joint_gp`` delegates to the stock ``fit_gp`` for
  single-output clusters, same internal ``np.random.seed(0)`` and restart
  count) and the samplers reproduce stock DCBO's trajectory exactly;
* intra-cluster edge edits of the ASSUMED graph leave the quotient — and
  hence the whole QDCBO trajectory — byte-identical (E2 invariance).

Faithfully mirrored stock quirks (needed for the exact finest-partition
reduction; both are properties of the authors' code, not choices of ours):
* transition mechanisms are fitted on the parent variables' columns at the
  CHILD's time slice (``fit_arcs`` uses ``data[pa][:, t]`` with ``t`` the
  child's index) even though they are evaluated at the t-1 values downstream;
* intervention clamping uses truthiness (``if interventions[var][t]:``), so a
  0.0-valued intervention falls through to the mechanism.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Sequence

import numpy as np

from ccbo.qdcbo.quotient_dbn import (QuotientDBNSpec, build_quotient_dbn,
                                     ensure_dcbo_on_path)

ensure_dcbo_on_path()

import GPy  # noqa: E402
from sklearn.neighbors import KernelDensity  # noqa: E402

from dcbo.bases.root import Root  # noqa: E402
from dcbo.methods.dcbo import DCBO  # noqa: E402
from dcbo.utils.gp_utils import fit_gp  # noqa: E402
from dcbo.utils.sequential_intervention_functions import (  # noqa: E402
    evaluate_target_function)
from dcbo.utils.utilities import (  # noqa: E402
    convert_to_dict_of_temporal_lists, select_sample)


# ---------------------------------------------------------------------------
# Joint cluster mechanisms
# ---------------------------------------------------------------------------

class JointClusterGP:
    """ICM multi-output GP over a cluster's member coordinates.

    ``GPy.models.GPCoregionalizedRegression`` with an RBF x Coregionalize
    (ICM) kernel: one latent input-space process shared across the m member
    outputs, mixed through a learned m x m task matrix, per-output Gaussian
    noise.  ``predict(Xnew) -> (mean (n, m), var (n, m))`` mirrors the
    two-moment contract of a stock ``GPRegression.predict``, so the SEM-hat
    closures treat scalar and joint mechanisms uniformly — every member's
    moments come from the SAME fitted joint posterior (coherent cluster-level
    sampling in DCBO's moment-propagation scheme).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, n_restart: int = 10,
                 seed: int = 0):
        assert x.ndim == 2 and y.ndim == 2 and y.shape[1] > 1
        self.n_outputs = y.shape[1]
        np.random.seed(seed)  # same determinism contract as stock fit_gp
        kernel = GPy.util.multioutput.ICM(
            input_dim=x.shape[1], num_outputs=self.n_outputs,
            kernel=GPy.kern.RBF(x.shape[1]))
        self.model = GPy.models.GPCoregionalizedRegression(
            X_list=[x] * self.n_outputs,
            Y_list=[y[:, j:j + 1] for j in range(self.n_outputs)],
            kernel=kernel)
        self.model.optimize_restarts(n_restart, verbose=False, robust=True)

    def predict(self, x_new: np.ndarray):
        n = x_new.shape[0]
        means, variances = [], []
        for j in range(self.n_outputs):
            idx = np.full((n, 1), j, dtype=float)
            mu, va = self.model.predict(
                np.hstack([x_new, idx]),
                Y_metadata={"output_index": idx.astype(int)})
            means.append(mu)
            variances.append(va)
        # GPy can return variances a hair below zero on near-interpolated
        # data; DCBO's acquisition clips at zero anyway — do it here so the
        # joint moments are valid variances.
        return np.hstack(means), np.clip(np.hstack(variances), 0.0, None)


def fit_joint_gp(x: np.ndarray, y: np.ndarray):
    """One mechanism for one cluster arc. Single-output clusters delegate to
    the stock ``fit_gp`` (identical seed/restart schedule), so singleton
    clusters reduce EXACTLY to the stock scalar fit."""
    if y.shape[1] == 1:
        return fit_gp(x=x, y=y)
    return JointClusterGP(x, y)


def _columns(data: dict, member_nodes: Sequence[str]) -> np.ndarray:
    """(n, k) matrix of the (base var, slice) columns named by member nodes."""
    cols = []
    for node in member_nodes:
        var, t = node.rsplit("_", 1)
        cols.append(np.asarray(data[var])[:, int(t)].reshape(-1, 1))
    return np.hstack(cols)


def fit_quotient_arcs(spec: QuotientDBNSpec, data: dict,
                      emissions: bool) -> dict:
    """Quotient replacement for the stock ``fit_arcs``.

    Returns ``{t: {cluster: {"inputs": member-node tuple, "model": obj}}}``
    keyed by the CHILD cluster (collision-free even under forks; the stock
    parent-tuple/fork-key gymnastics disappear because each cluster owns
    exactly one mechanism per arc type).

    data is the DCBO observational dict {base var: (n, T) array}; only the
    member columns named by the spec are read — cluster membership metadata
    is the sole structural input.
    """
    T = spec.T
    fncs = {t: {} for t in range(T)}
    if emissions:
        for t in range(T):
            for C in spec.cluster_order:
                pars = spec.emit_cluster_parents[t][C]
                members = spec.member_nodes([C], t)
                if not pars:
                    # Source cluster in the slice: joint marginal (KDE), same
                    # estimator family as stock's per-variable marginal.
                    xx = _columns(data, members)
                    fncs[t][C] = {"inputs": (),
                                  "model": KernelDensity(
                                      kernel="gaussian").fit(xx)}
                else:
                    inputs = spec.member_nodes(pars, t)
                    fncs[t][C] = {"inputs": inputs,
                                  "model": fit_joint_gp(
                                      _columns(data, inputs),
                                      _columns(data, members))}
    else:
        if T == 1:
            return fncs
        for t in range(1, T):
            for C in spec.cluster_order:
                pars = spec.trans_cluster_parents[t][C]
                if not pars:
                    continue
                inputs = spec.member_nodes(pars, t - 1)
                # Stock-quirk mirror: regressors are the parent VARIABLES'
                # columns at the child's slice t (see module docstring).
                inputs_at_t = spec.member_nodes(pars, t)
                fncs[t][C] = {"inputs": inputs,
                              "model": fit_joint_gp(
                                  _columns(data, inputs_at_t),
                                  _columns(data,
                                           spec.member_nodes([C], t)))}
    return fncs


# ---------------------------------------------------------------------------
# Quotient SEM-hat + sampling
# ---------------------------------------------------------------------------

def make_qsem_hat(spec: QuotientDBNSpec) -> Callable:
    """Builder handed to DCBO as ``make_sem_estimator``: called as
    ``make(G=self.G, emission_fncs=..., transition_fncs=...)`` and returns a
    SEM-hat class whose ``static``/``dynamic`` OrderedDicts are keyed by
    CLUSTER and whose functions return all member coordinates at once from
    the cluster's joint posterior (mirrors the stock ``build_sem_hat``
    dispatch: marginal / emit / trans / emit-plus-trans, decided on the
    quotient graph's slice-0 and slice-1 connectivity)."""

    def make(G=None, emission_fncs=None, transition_fncs=None):

        class QSEMHat:
            @staticmethod
            def _make_marginal() -> Callable:
                return lambda t, margin_id: emission_fncs[t][
                    margin_id[1].rsplit("_", 1)[0]]["model"].sample()

            @staticmethod
            def _make_emit_fnc(C: str, moment: int) -> Callable:
                return lambda t, _, emit_input_vars, sample: (
                    emission_fncs[t][C]["model"].predict(
                        select_sample(sample, emit_input_vars, t))[moment])

            @staticmethod
            def _make_trans_fnc(C: str, moment: int) -> Callable:
                return lambda t, transfer_input_vars, _, sample: (
                    transition_fncs[t][C]["model"].predict(
                        select_sample(sample, transfer_input_vars,
                                      t - 1))[moment])

            @staticmethod
            def _make_emit_plus_trans_fnc(C: str, moment: int) -> Callable:
                return (
                    lambda t, transfer_input_vars, emit_input_vars, sample:
                    transition_fncs[t][C]["model"].predict(
                        select_sample(sample, transfer_input_vars,
                                      t - 1))[moment]
                    + emission_fncs[t][C]["model"].predict(
                        select_sample(sample, emit_input_vars, t))[moment])

            def static(self, moment: int) -> OrderedDict:
                assert moment in (0, 1), moment
                f = OrderedDict()
                for C in spec.cluster_order:
                    if not spec.emit_cluster_parents[0][C]:
                        f[C] = self._make_marginal()
                    else:
                        f[C] = self._make_emit_fnc(C, moment)
                return f

            def dynamic(self, moment: int) -> OrderedDict:
                assert moment in (0, 1), moment
                assert spec.T > 1
                f = OrderedDict()
                for C in spec.cluster_order:
                    emit_p = spec.emit_cluster_parents[1][C]
                    trans_p = spec.trans_cluster_parents[1][C]
                    if not emit_p and not trans_p:
                        f[C] = self._make_marginal()
                    elif not emit_p:
                        f[C] = self._make_trans_fnc(C, moment)
                    elif not trans_p:
                        f[C] = self._make_emit_fnc(C, moment)
                    else:
                        f[C] = self._make_emit_plus_trans_fnc(C, moment)
                return f

        return QSEMHat

    return make


def sequential_sample_from_qsem_hat(static_sem: OrderedDict,
                                    dynamic_sem: OrderedDict,
                                    timesteps: int,
                                    spec: QuotientDBNSpec,
                                    initial_values: dict = None,
                                    interventions: dict = None,
                                    seed: int = None) -> OrderedDict:
    """Cluster-major mirror of the stock ``sequential_sample_from_SEM_hat``.

    The returned sample dict stays keyed by BASE variables (so blankets,
    ``select_sample`` and the target bookkeeping are untouched), but each
    cluster's members are filled in one shot from the cluster's joint
    mechanism.  At the finest partition every cluster is a singleton and this
    walk reproduces the stock sampler operation-for-operation (including its
    truthiness-based clamping and its RNG consumption pattern: one KDE draw
    per source cluster, GP moments elsewhere).
    """
    if seed:
        np.random.seed(seed)
    sample = OrderedDict((v, np.zeros(timesteps)) for v in spec.base_order)
    if initial_values:
        assert set(sample.keys()) == set(initial_values.keys())

    for t in range(timesteps):
        static_slice = t == 0 or dynamic_sem is None
        sems = static_sem if static_slice else dynamic_sem
        for C, function in sems.items():
            members = spec.members(C)
            if t == 0 and interventions and initial_values:
                for m in members:
                    if interventions[m][t] is not None and \
                            initial_values.get(m) is not None:
                        raise ValueError(
                            "Cannot provide both an initial value and an "
                            f"intervention for {m} at t=0.")
            # Whole-cluster clamp: skip the mechanism entirely (stock skips
            # evaluation for clamped variables; truthiness mirrored).
            if interventions and all(interventions[m][t] for m in members):
                for m in members:
                    sample[m][t] = interventions[m][t]
                continue
            if static_slice and initial_values and not any(
                    interventions and interventions[m][t] for m in members):
                for m in members:
                    sample[m][t] = initial_values[m]
                continue

            V = f"{C}_{t}"
            emit_pars = spec.emit_input_nodes(C, t)
            if static_slice:
                if emit_pars:
                    out = function(t, None, emit_pars, sample)
                else:
                    out = function(t, (None, V))
            else:
                trans_pars = spec.trans_input_nodes(C, t)
                if not trans_pars and not emit_pars:
                    out = function(t, (None, V))
                else:
                    out = function(t, trans_pars, emit_pars, sample)

            if len(members) == 1:
                # Keep the stock assignment form exactly (predict returns a
                # (1, 1) array; numpy coerces it into the scalar slot).
                sample[members[0]][t] = out
            else:
                vals = np.asarray(out).reshape(-1)
                assert vals.shape[0] == len(members), (C, vals.shape)
                for j, m in enumerate(members):
                    sample[m][t] = vals[j]
            # Partial clamp (only reachable for multi-member clusters where
            # some members carry truthy blanket values): override members.
            if interventions:
                for m in members:
                    if interventions[m][t]:
                        sample[m][t] = interventions[m][t]
    return sample


def update_sufficient_statistics_qhat(
    temporal_index: int,
    target_variable: str,
    exploration_set: tuple,
    sem_hat,
    spec: QuotientDBNSpec,
    dynamic: bool,
    assigned_blanket: dict,
    mean_dict_store: dict,
    var_dict_store: dict,
    seed: int = 1,
):
    """Quotient mirror of ``dcbo.utils.gp_utils.update_sufficient_statistics_hat``
    (same closures, caches and seeds; only the graph walk is replaced by the
    cluster-major sampler)."""
    if dynamic:
        intervention_blanket = deepcopy(assigned_blanket)
        dynamic_sem_mean = sem_hat().dynamic(moment=0)
        dynamic_sem_var = sem_hat().dynamic(moment=1)
    else:
        intervention_blanket = deepcopy(assigned_blanket)
        assert [all(intervention_blanket[key] is None
                    for key in intervention_blanket.keys())]
        dynamic_sem_mean = None
        dynamic_sem_var = None

    kwargs1 = {
        "static_sem": sem_hat().static(moment=0),
        "dynamic_sem": dynamic_sem_mean,
        "spec": spec,
        "timesteps": temporal_index + 1,
    }
    kwargs2 = {
        "static_sem": sem_hat().static(moment=1),
        "dynamic_sem": dynamic_sem_var,
        "spec": spec,
        "timesteps": temporal_index + 1,
    }

    def _cache_key(x: np.ndarray) -> tuple:
        x_arr = np.asarray(x)
        return (x_arr.shape, x_arr.tobytes())

    def mean_function(x_vals) -> np.ndarray:
        samples = []
        for x in x_vals:
            k = _cache_key(x)
            store = mean_dict_store[temporal_index][exploration_set]
            if k in store:
                samples.append(store[k])
            else:
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][
                        temporal_index] = xx
                sample = sequential_sample_from_qsem_hat(
                    interventions=intervention_blanket, seed=seed, **kwargs1)
                samples.append(sample[target_variable][temporal_index])
                store[k] = sample[target_variable][temporal_index]
        return np.vstack(samples)

    def variance_function(x_vals) -> np.ndarray:
        out = []
        for x in x_vals:
            k = _cache_key(x)
            store = var_dict_store[temporal_index][exploration_set]
            if k in store:
                out.append(store[k])
            else:
                for intervention_variable, xx in zip(exploration_set, x):
                    intervention_blanket[intervention_variable][
                        temporal_index] = xx
                sample = sequential_sample_from_qsem_hat(
                    interventions=intervention_blanket, seed=seed, **kwargs2)
                out.append(sample[target_variable][temporal_index])
                store[k] = sample[target_variable][temporal_index]
        return np.vstack(out)

    return mean_function, variance_function


# ---------------------------------------------------------------------------
# The method class
# ---------------------------------------------------------------------------

class QDCBO(DCBO):
    """Stock DCBO driven entirely by the quotient view of its temporal graph.

    Constructor takes the FINE (assumed) graph plus a ``partition`` over the
    manipulable base variables; everything model-side is immediately
    quotiented (``self.G`` IS the quotient graph) while the environment-side
    bookkeeping (blankets, target functions) is rebuilt in base-variable
    coordinates.  Unsupported stock options (online refits, data integration,
    hyperparameter transfer) are rejected explicitly — their update paths
    read the stock per-edge function dictionaries.
    """

    def __init__(
        self,
        G,
        sem,
        make_sem_estimator: Callable = None,   # ignored; quotient builder used
        observation_samples: dict = None,
        intervention_domain: dict = None,
        intervention_samples: dict = None,
        exploration_sets: list = None,
        partition: Sequence[Sequence[str]] = None,
        number_of_trials: int = 10,
        base_target_variable: str = "Y",
        task: str = "min",
        estimate_sem: bool = True,
        cost_type: int = 1,
        ground_truth: list = None,
        n_restart: int = 1,
        use_mc: bool = False,
        debug_mode: bool = False,
        online: bool = False,
        optimal_assigned_blankets: dict = None,
        use_di: bool = False,
        transfer_hp_o: bool = False,
        transfer_hp_i: bool = False,
        hp_i_prior: bool = True,
        n_obs_t=None,
        num_anchor_points=100,
        seed: int = 1,
        sample_anchor_points: bool = False,
        seed_anchor_points=None,
        args_sem=None,
        manipulative_variables: list = None,
        change_points: list = None,
    ):
        assert partition is not None, "QDCBO requires a partition"
        assert estimate_sem, "QDCBO is defined for the estimated-SEM regime"
        if online or use_di or transfer_hp_o or transfer_hp_i or \
                isinstance(n_obs_t, list):
            raise NotImplementedError(
                "online/use_di/transfer_hp refit paths read stock per-edge "
                "function dictionaries and are not supported by QDCBO")

        spec = build_quotient_dbn(G, partition, base_target_variable)
        self._spec = spec

        if exploration_sets is None:
            exploration_sets = [tuple(
                k for k in observation_samples.keys()
                if base_target_variable not in k)]
        lifted_es = spec.lift_exploration_sets(exploration_sets)
        spec.lift_intervention_domain(intervention_domain)

        root_args = {
            "G": spec.G_quotient,
            "sem": sem,
            "make_sem_estimator": make_qsem_hat(spec),
            "observation_samples": observation_samples,
            "intervention_domain": intervention_domain,
            "intervention_samples": intervention_samples,
            "exploration_sets": lifted_es,
            "estimate_sem": estimate_sem,
            "base_target_variable": base_target_variable,
            "task": task,
            "cost_type": cost_type,
            "use_mc": use_mc,
            "number_of_trials": number_of_trials,
            "ground_truth": ground_truth,
            "n_restart": n_restart,
            "debug_mode": debug_mode,
            "online": online,
            "num_anchor_points": num_anchor_points,
            "args_sem": args_sem,
            "manipulative_variables": manipulative_variables,
            "change_points": change_points,
        }
        # Bypass BaseClassDCBO.__init__ (its fit_arcs assumes scalar,
        # fine-graph nodes); Root does everything else.
        Root.__init__(self, **root_args)

        # --- environment-side rebuild in base-variable coordinates ---------
        self._rebuild_env_side()

        # --- quotient mechanism fits (before temporal-list conversion) ------
        self.sem_emit_fncs = fit_quotient_arcs(
            spec, self.observational_samples, emissions=True)
        self.sem_trans_fncs = fit_quotient_arcs(
            spec, self.observational_samples, emissions=False)

        # --- stock DCBO.__init__ tail ---------------------------------------
        self.optimal_assigned_blankets = optimal_assigned_blankets
        self.use_di = use_di
        self.transfer_hp_o = transfer_hp_o
        self.transfer_hp_i = transfer_hp_i
        self.hp_i_prior = hp_i_prior
        self.hyperparam_obs_emit = {}
        self.hyperparam_obs_transf = {}
        self.n_obs_t = n_obs_t
        self.seed = seed
        self.sample_anchor_points = sample_anchor_points
        self.seed_anchor_points = seed_anchor_points
        self.observational_samples = convert_to_dict_of_temporal_lists(
            self.observational_samples)

    # ------------------------------------------------------------------
    def _rebuild_env_side(self) -> None:
        """Root built blankets / target functions from quotient node names;
        the environment operates in base-variable coordinates, so rebuild
        them (sorted keys, mirroring the stock blanket constructor)."""
        import networkx as nx
        spec = self._spec
        blanket = {v: self.T * [None] for v in sorted(spec.base_order)}
        self.optimal_blanket = deepcopy(blanket)
        self.assigned_blanket = deepcopy(blanket)
        self.empty_intervention_blanket = deepcopy(blanket)
        if self.estimate_sem:
            self.assigned_blanket_hat = deepcopy(blanket)

        # Minimal base-named graph: evaluate_target_function only reads node
        # names to key its intervention blanket (environment-side object).
        env_key_graph = nx.MultiDiGraph()
        for v in spec.base_order:
            env_key_graph.add_node(f"{v}_0")
        for temporal_index in range(self.T):
            for es in self.exploration_sets:
                self.target_functions[temporal_index][es] = \
                    evaluate_target_function(
                        self.true_initial_sem, self.true_sem, env_key_graph,
                        es, self.observational_samples.keys(), self.T)

    # ------------------------------------------------------------------
    # Quotient parent resolution (replaces the stock string-parsing lookup)
    # ------------------------------------------------------------------
    def node_parents(self, V: str, t: int = None) -> tuple:
        """Member-node tuples of the quotient parents of V's cluster.

        ``t == slice(V) - 1`` selects transition parents (at t), anything
        else the within-slice emission parents (at t) — the stock contract.
        Accepts cluster nodes and (for singleton clusters) base-var nodes.
        """
        name, vt = V.rsplit("_", 1)
        vt = int(vt)
        if t is not None and vt - 1 == t:
            return self._spec.trans_input_nodes(name, vt)
        return self._spec.emit_input_nodes(name, vt if t is None else t)

    def _filter_on_time_index(self, node: str, temporal_index: int) -> tuple:
        """Quotient predecessors of a quotient node, filtered on slice."""
        name = node.rsplit("_", 1)[0]
        C = self._spec.cluster_of(name)
        preds = self.G.predecessors(f"{C}_{node.rsplit('_', 1)[1]}")
        return tuple(sorted(
            (p for p in preds if p.endswith(str(temporal_index))),
            key=self.sorted_nodes.get))

    # ------------------------------------------------------------------
    def _update_sufficient_statistics(
            self, target: str, temporal_index: int, dynamic: bool,
            assigned_blanket: dict, updated_sem) -> None:
        target_variable, target_temporal_index = target.split("_")
        assert int(target_temporal_index) == temporal_index
        for es in self.exploration_sets:
            (self.mean_function[temporal_index][es],
             self.variance_function[temporal_index][es],
             ) = update_sufficient_statistics_qhat(
                temporal_index=temporal_index,
                target_variable=target_variable,
                exploration_set=es,
                sem_hat=updated_sem,
                spec=self._spec,
                dynamic=dynamic,
                assigned_blanket=assigned_blanket,
                mean_dict_store=self.mean_dict_store,
                var_dict_store=self.var_dict_store,
            )

    # ------------------------------------------------------------------
    def _post_optimisation_assignments(self, target: tuple, t: int,
                                       DCBO: bool = False) -> None:
        """Stock bookkeeping, but ``assign_blanket``'s child fill-in (an
        environment-side true-SEM sample) resolves successors on the
        member-coordinate view of the quotient graph, whose node names match
        the base-variable blanket keys."""
        G_quotient = self.G
        try:
            self.G = self._spec.G_member_view
            super()._post_optimisation_assignments(target, t, DCBO=DCBO)
        finally:
            self.G = G_quotient
