"""Matched controlled experiments: population feedback and learner accounting.

The protocol follows the CBO reference implementation (Aglietti et al., 2020):

* every measurement returns the exact population expectation E[Y | do(X = x)]
  of the true SCM (the reference target averages 100,000 SEM draws; here the
  closed forms in ``minibench`` replace that average);
* each arm starts with ``n_init`` uniformly drawn initial interventions that are
  given, not charged, so cost is counted from 0 over sequential interventions;
* the optimizer starts from 100 observational rows and, at every trial, either
  observes (20 further rows, up to 150, via CBO's coverage-based epsilon rule)
  or intervenes; observing costs no intervention budget;
* runs stop at a fixed intervention-cost budget (the reference uses a fixed
  number of trials; trial indices are logged so both views are available).

In addition, the null intervention (observational mean of Y over the initial
rows, zero cost) is an eligible recommendation. Population scoring lives in a
separate, post-run function.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
from ccbo import minibench as mb

# v4: CBO-aligned (exact feedback, free initial design, observe-or-intervene)
# with a null-eligible recommendation. Observational data and initial levels are
# keyed by the unchanged v2 namespace, so protocol versions share them.
PROTOCOL_ID = "matched-controlled-cbo-v4"
SEED_NAMESPACE = "matched-controlled-noisy-v2"
FEEDBACK_MODE = "population_expectation"
DEFAULT_BUDGET = {mb.PP_NAME: 100, mb.FD_NAME: 100, mb.MC_NAME: 120}
DEFAULT_N_INIT = {scm: 3 for scm in DEFAULT_BUDGET}   # free initial points per arm
N_OBS_INITIAL, N_OBS_POOL, N_OBS_BATCH = 100, 150, 20   # CBO reference: 100, +20, cap 100+50
NODES = {scm: tuple(mb.NODES[scm]) for scm in DEFAULT_BUDGET}
SCALAR_METHODS = ("CBO", "QCBO", "BO-S", "BO", "CBO-NP")


class PilotComplete(Exception):
    """Explicit pilot-only sequential purchase cap reached."""


class BudgetExhausted(Exception):
    """Normal termination when no allowed arm can be purchased."""


def keyed_seed(*parts):
    encoded = json.dumps([SEED_NAMESPACE, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:4], "little")


def canonical_arm(arm):
    return tuple(sorted(arm))


def domains(scm):
    return {v: mb.domain(scm, v) for v in NODES[scm] if v != "Y"}


def all_arms(scm):
    names = sorted(domains(scm))
    return [a for n in range(1, len(names) + 1) for a in combinations(names, n)]


def sample_true(scm, iv, z):
    """One true SCM row, driven by ``mb.NOISE_DIM[scm]`` supplied standard normals."""
    if scm == mb.PP_NAME:
        u = mb.PP_SIGMA_U * z[0]
        a = iv.get("X1", u + mb.PP_SIGMA_IN*z[1])
        b = iv.get("X2", u + mb.PP_SIGMA_IN*z[2])
        y = mb.PP_LAM*(a-mb.PP_A)**2 + (b-mb.PP_B)**2 + mb.SIGMA_Y*z[3]
        return dict(X1=float(a), X2=float(b), Y=float(y))
    if scm == mb.FD_NAME:
        u = mb.FD_SIGMA_U*z[0]
        a = iv.get("X1", u + mb.FD_SIGMA_1*z[1])
        m = iv.get("M", mb.FD_B*a + mb.FD_SIGMA_M*z[2])
        y = mb.FD_AMP*(1-np.exp(-(m-mb.FD_C)**2/(2*mb.FD_W**2))) + mb.FD_G*u + mb.SIGMA_Y*z[3]
        return dict(X1=float(a), M=float(m), Y=float(y))
    if scm == mb.MC_NAME:
        a = iv.get("X1", mb.MC_SIGMA_1*z[1])
        b = iv.get("X2", mb.MC_BETA*a + mb.MC_SIGMA_2*z[2])
        return dict(X1=float(a), X2=float(b), Y=float((b-mb.MC_C)**2 + mb.SIGMA_Y*z[3]))
    raise ValueError(scm)


def natural_means(scm, iv):
    """Population means of the non-intervened manipulable variables under do(iv)."""
    row = dict(iv)
    if scm == mb.PP_NAME:
        row.setdefault("X1", 0.)
        row.setdefault("X2", 0.)
    elif scm == mb.FD_NAME:
        row.setdefault("X1", 0.)
        row.setdefault("M", mb.FD_B * row["X1"])
    elif scm == mb.MC_NAME:
        row.setdefault("X1", 0.)
        row.setdefault("X2", mb.MC_BETA * row["X1"])
    else:
        raise ValueError(scm)
    return row


def population_row(scm, iv):
    """One measurement: intervened coordinates, natural means of the remaining
    manipulable variables, and the exact target expectation E[Y | do(iv)]."""
    row = natural_means(scm, iv)
    row["Y"] = mb.population_do(scm, list(iv), [iv[v] for v in iv])
    return {v: float(row[v]) for v in NODES[scm]}


def observational_data(scm, seed, n=N_OBS_INITIAL):
    """Replicate-specific true-SCM observational rows. The first rows of a larger
    pool equal the smaller dataset, so the initial 100 rows are the same in
    every protocol version and the pool only appends rows revealed by observing."""
    import pandas as pd
    rng = np.random.RandomState(keyed_seed(scm, seed, "observational"))
    return pd.DataFrame([sample_true(scm, {}, rng.randn(mb.NOISE_DIM[scm])) for _ in range(n)],
                        columns=NODES[scm])


class Experiment:
    """Measurement-only learner interface; every purchased row is an event."""
    def __init__(self, scm, seed, arms, budget=None, n_init=None, max_purchases=None,
                 null_estimate=None):
        self.scm, self.seed = scm, int(seed)
        self.max_purchases = max_purchases
        n_init = DEFAULT_N_INIT[scm] if n_init is None else n_init
        # Observational mean of Y: the zero-cost value of recommending no
        # intervention. ``None`` keeps the v2 rule (executed interventions only).
        self.null_estimate = None if null_estimate is None else float(null_estimate)
        self.arms = sorted({canonical_arm(a) for a in arms}, key=lambda a: (len(a), a))
        self.budget = int(DEFAULT_BUDGET[scm] if budget is None else budget)
        self.n_init, self.cost, self.sequential_index = int(n_init), 0, 0
        self.trial = 0              # optimizer trials so far (observe or intervene)
        self.observation_log = []   # one entry per observe trial
        if not self.arms or any(a not in all_arms(scm) for a in self.arms):
            raise ValueError("invalid arm family")
        if self.n_init < 0:
            raise ValueError("negative initial design")
        self.events = []
        self.initial = {}
        for arm in self.arms:
            rng = np.random.RandomState(keyed_seed(scm, seed, "init-levels", arm))
            self.initial[arm] = []
            if not self.n_init:
                continue
            xs = np.column_stack([rng.uniform(*domains(scm)[v], self.n_init) for v in arm])
            for x in xs:
                # Initial interventions are given, as in CBO: not charged.
                self.initial[arm].append(deepcopy(self._buy(arm, x, "init", charged=False)))

    @property
    def remaining(self):
        return self.budget-self.cost

    def affordable(self, arm):
        return len(arm) <= self.remaining

    def check_available(self):
        if self.max_purchases is not None and self.sequential_index >= self.max_purchases:
            raise PilotComplete()
        if not any(self.affordable(a) for a in self.arms):
            raise BudgetExhausted()

    def _buy(self, arm, values, phase, charged=True):
        supplied = tuple(arm)
        x = np.asarray(values, dtype=float).ravel()
        if len(supplied) != len(x) or len(set(supplied)) != len(supplied):
            raise ValueError("intervention coordinates do not match arm")
        arm = canonical_arm(supplied)
        iv = dict(zip(supplied, map(float, x)))
        if arm not in self.arms:
            raise ValueError("unavailable arm")
        if charged and not self.affordable(arm):
            raise BudgetExhausted()
        if any(not np.isfinite(v) or not domains(self.scm)[k][0]-1e-9 <= v <= domains(self.scm)[k][1]+1e-9 for k,v in iv.items()):
            raise ValueError("intervention outside declared domain")
        row = population_row(self.scm, iv)
        cost = len(arm) if charged else 0
        self.cost += cost
        event = dict(event_id=len(self.events), phase=phase, arm=list(arm),
                     x=[iv[v] for v in arm], measured=row,
                     cost=cost, cum_cost=self.cost, trial=self.trial if charged else 0)
        self.events.append(event)
        event["recommendation_event_id"] = recommend(self.events, self.null_estimate)
        return deepcopy(row)

    def purchase(self, arm, values):
        self.check_available()
        self.trial += 1
        row = self._buy(arm, values, "sequential")
        self.sequential_index += 1
        return row

    def observed(self, n_rows_after):
        """Record an observe trial (no intervention cost)."""
        self.trial += 1
        self.observation_log.append(dict(trial=self.trial, cum_cost=self.cost, n_rows_after=int(n_rows_after)))

    def target(self, arm, values):
        return self.purchase(arm, values)["Y"]

    def scalar_initial(self, arms):
        xs, ys = [], []
        for a in arms:
            rows = self.initial[canonical_arm(a)]
            xs.append(np.array([[r[v] for v in a] for r in rows]).reshape(len(rows), len(a)))
            ys.append(np.array([[r["Y"]] for r in rows]).reshape(len(rows), 1))
        if not self.events:
            # No initial design: the starting incumbent is the null intervention,
            # credited to the first arm's label only so the backend's incumbent
            # bookkeeping (min over arms) sees its value.
            if self.null_estimate is None:
                raise ValueError("no initial design requires the null estimate as starting incumbent")
            first = list(arms[0])
            return xs, ys, np.zeros(len(first)), self.null_estimate, "".join(first)
        best = min(self.events, key=lambda e: (e["measured"]["Y"], tuple(e["arm"]), e["event_id"]))
        # Reorder best point to the backend's variable ordering.
        backend_arm = next(a for a in arms if canonical_arm(a) == tuple(best["arm"]))
        iv = dict(zip(best["arm"], best["x"]))
        return xs, ys, np.array([iv[v] for v in backend_arm]), best["measured"]["Y"], "".join(backend_arm)


def recommend(events, null_estimate=None):
    """Recommendation after ``events``: the executed intervention with the
    smallest measured Y (ties: canonical arm, then execution index), or the
    null intervention (``None``) when its observational value is strictly
    smaller. The rule uses measurements only, never population values."""
    best = min(events, key=lambda e: (e["measured"]["Y"], tuple(e["arm"]), e["event_id"]))
    if null_estimate is not None and null_estimate < best["measured"]["Y"]:
        return None
    return best["event_id"]


OPTIMUM = {mb.PP_NAME: 0., mb.FD_NAME: 0., mb.MC_NAME: mb.MC_SIGMA_2**2}


def regret_start(scm):
    """First cost of the common cost grid. Without an initial design every
    method has a recommendation (the null intervention) from cost 0."""
    return 0


def score_events(scm, events, budget, null_estimate=None):
    """Post-run oracle evaluation of data-selected recommendations; never cummin μ."""
    oracle = mb.population_evaluator(scm)
    optimum = OPTIMUM[scm]
    result, oracle_best = [], float("inf")
    if null_estimate is not None:
        # State before the first purchase: only the null intervention is available.
        null_population = float(mb.null_value(scm))
        result.append(dict(event_id=None, cum_cost=0, recommendation_event_id=None,
                           recommendation_population=null_population,
                           recommendation_regret=null_population-optimum,
                           oracle_best_visited=null_population))
        oracle_best = null_population
    for index, event in enumerate(events):
        chosen = recommend(events[:index+1], null_estimate)
        if "recommendation_event_id" in event and event["recommendation_event_id"] != chosen:
            raise ValueError("logged recommendation disagrees with declared measured-data rule")
        if chosen is None:
            value = float(mb.null_value(scm))
        else:
            value = float(oracle(events[chosen]["arm"], events[chosen]["x"]))
        oracle_best = min(oracle_best, float(oracle(event["arm"], event["x"])))
        result.append(dict(event_id=event["event_id"], cum_cost=event["cum_cost"],
                           recommendation_event_id=chosen, recommendation_population=value,
                           recommendation_regret=value-optimum, oracle_best_visited=oracle_best))
    if not events:
        raise ValueError("empty experiment")
    checkpoints = []
    for c in range(regret_start(scm), int(budget)+1):
        eligible = [r for r in result if r["cum_cost"] <= c]
        if eligible:
            checkpoints.append(dict(cost=c, **{k:v for k,v in eligible[-1].items() if k != "cum_cost"}))
    return dict(final=result[-1], scored_events=[r for r in result if r["event_id"] is not None],
                initial=result[0] if result[0]["event_id"] is None else None, checkpoints=checkpoints,
                cost_integrated_recommendation_regret=sum(r["recommendation_regret"] for r in checkpoints))


@contextmanager
def scalar_runtime(experiment):
    """Scoped affordable-arm and budget hooks around the vendored CBO loop.

    Surrogates are left to the engine: targets are exact population
    expectations, so the interventional likelihood variance stays fixed at the
    engine default (``BO_functions.NOISE_VAR``). Arms without data use the
    engine's prior-only surrogate. Every built surrogate and every acquisition
    call is audited so the fixed-noise policy is verifiable post hoc.
    """
    import importlib
    cbo = importlib.import_module("ccbo.cbo.cbo")
    bo = importlib.import_module("ccbo.cbo.bo")
    old_build, old_next, old_fix = cbo.update_BO_models, cbo.find_next_y_point, bo.fix_noise
    old_observe = cbo.observe

    def observe(complete_dataset, start, batch_size, stop):
        rows, cursor = old_observe(complete_dataset, start, batch_size, stop)
        experiment.observed(cursor)
        return rows, cursor

    def record(gpy_model, prior_only=False):
        if not hasattr(experiment, "gp_noise_audit"):
            experiment.gp_noise_audit = []
        if prior_only:
            audit = dict(variance=None, fixed=True, policy="prior_only", n_rows=0, acquisition_snapshots=[])
        else:
            audit = dict(variance=float(gpy_model.likelihood.variance),
                         fixed=bool(gpy_model.likelihood.variance.is_fixed),
                         policy="fixed", n_rows=len(gpy_model.Y), acquisition_snapshots=[])
        experiment.gp_noise_audit.append(audit)
        return audit

    def build(*args, **kwargs):
        wrapper = old_build(*args, **kwargs)
        wrapper._matched_noise_audit = record(wrapper.model, prior_only=wrapper.model is None)
        return wrapper

    def fix_noise_bo(gpy_model, *args, **kwargs):
        # The joint BO baseline builds its surrogate in ccbo.cbo.bo.
        gpy_model = old_fix(gpy_model, *args, **kwargs)
        record(gpy_model)
        return gpy_model

    def next_point(space, model, current, arm, costs, task="min"):
        experiment.check_available()
        if not experiment.affordable(arm):
            return np.full((1, 1), -np.inf), np.zeros((1, len(arm)))
        audit = getattr(model, "_matched_noise_audit", None)
        if audit is None:
            raise RuntimeError("acquisition received a surrogate built outside the matched runtime")
        gpy_model = model.model
        if gpy_model is not None and not gpy_model.likelihood.variance.is_fixed:
            raise RuntimeError("interventional likelihood variance must stay fixed under population feedback")
        audit["acquisition_snapshots"].append(dict(
            arm=list(arm), n_rows=0 if gpy_model is None else len(gpy_model.Y),
            variance=None if gpy_model is None else float(gpy_model.likelihood.variance),
            parameters=None if gpy_model is None else gpy_model.param_array.tolist()))
        return old_next(space, model, current, arm, costs, task=task)

    cbo.update_BO_models, cbo.find_next_y_point, bo.fix_noise = build, next_point, fix_noise_bo
    cbo.observe = observe
    try:
        yield
    finally:
        cbo.update_BO_models, cbo.find_next_y_point, bo.fix_noise = old_build, old_next, old_fix
        cbo.observe = old_observe


def method_partition(scm, method, partition_id=None):
    """Partition used to build a method's arms (``fine`` for graph-only CBO)."""
    if method in ("CBO", "CBO-NP", "BO-S", "BO"):
        if partition_id not in (None, "fine"):
            raise ValueError(f"{method} uses the fine partition")
        return "fine"
    pid = partition_id or "coarse"
    if pid not in mb.partition_ids(scm) or pid == "fine":
        raise ValueError(f"{method} needs a coarse partition of {scm}, got {pid!r}")
    return pid


def build_graph(scm, cond, method, obs, partition_id=None):
    """Arms, manipulables and the CoarsenedGraph supplying priors for ``method``."""
    from ccbo.scm_graphs import get_original_graph
    from ccbo.coarsened_graph import CoarsenedGraph
    graph = get_original_graph(scm, obs)
    pid = method_partition(scm, method, partition_id)
    assumed = mb.variant_name(cond)
    cg = CoarsenedGraph(graph, mb.partition(scm, pid), scm, obs,
                        num_mc_samples=2000, assumed_graph_name=assumed)
    arms, _, manip = cg.get_sets()
    arms = [list(a) for a in arms]
    if method == "BO-S":
        arms = [list(a) for a in all_arms(scm)]
    if method == "BO":
        arms = [sorted(manip)]
    return cg, arms, manip, pid


def run_scalar(scm, cond, method, seed, budget=None, n_init=None, max_purchases=None,
               partition_id=None):
    """Run the existing CBO or joint BO backend on the matched environment."""
    from ccbo.cbo.utils import compute_coverage
    from ccbo.cbo.cbo import CBO
    from ccbo.cbo.bo import NonCausal_BO
    if method not in SCALAR_METHODS:
        raise ValueError(method)
    mb.register_variants()
    algorithm_seed = keyed_seed(scm, seed, "algorithm")
    np.random.seed(algorithm_seed)
    pool = observational_data(scm, seed, N_OBS_POOL)
    obs = pool.iloc[:N_OBS_INITIAL].copy()
    cg, arms, manip, pid = build_graph(scm, cond, method, obs, partition_id)
    functions = cg.fit_all_models()
    ranges, costs = cg.get_interventional_ranges(), cg.get_cost_structure(1)
    _, _, coverage = compute_coverage(obs, manip, ranges)
    experiment = Experiment(scm, seed, arms, budget, n_init, max_purchases,
                            null_estimate=float(obs["Y"].mean()))
    xs, ys, bestx, besty, bestarm = experiment.scalar_initial(arms)
    with scalar_runtime(experiment):
        try:
            if method == "BO":
                NonCausal_BO(experiment.budget, cg, ranges, xs[0], ys[0], costs,
                            obs, functions, bestx, besty, arms[0], Causal_prior=False,
                            target_evaluator=experiment.target)
            else:
                # One trial per unit of budget plus every possible observe trial.
                num_trials = experiment.budget + (N_OBS_POOL - N_OBS_INITIAL) // N_OBS_BATCH + 2
                CBO(num_trials, arms, manip, xs, ys, bestx, besty, bestarm,
                    ranges, functions, obs, coverage, cg, N_OBS_BATCH, costs, pool,
                    "min", N_OBS_POOL, N_OBS_INITIAL, experiment.n_init,
                    Causal_prior=method in ("CBO", "QCBO"),
                    target_evaluator=experiment.target, force_observe_on_entry=None)
        except (BudgetExhausted, PilotComplete):
            pass
    experiment.check_available_if_complete = not any(experiment.affordable(a) for a in experiment.arms)
    if not experiment.check_available_if_complete and not (max_purchases is not None and experiment.sequential_index == max_purchases):
        raise RuntimeError("backend stopped with affordable actions remaining")
    return experiment, dict(backend="vendored-CBO" if method != "BO" else "vendored-BO",
                            algorithm_seed=algorithm_seed,
                            observation_log=experiment.observation_log,
                            new_observation_rows=(experiment.observation_log[-1]["n_rows_after"] - N_OBS_INITIAL
                                                  if experiment.observation_log else 0),
                            partition=pid, arms=[list(a) for a in experiment.arms],
                            gp_noise_audit=getattr(experiment, "gp_noise_audit", []),
                            feedback_mode=FEEDBACK_MODE,
                            noise_policy="fixed interventional likelihood variance (engine NOISE_VAR); exact population feedback")
