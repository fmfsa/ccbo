"""Matched controlled experiments: learner-visible stochastic data and accounting.

Population scoring lives in a separate, post-run function. No oracle values are
computed during a learner run. Historical exact experiment defaults are untouched.
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

PROTOCOL_ID = "matched-controlled-noisy-v2"
DEFAULT_BUDGET = {mb.PP_NAME: 100, mb.FD_NAME: 100, mb.MC_NAME: 120}
NODES = {mb.PP_NAME: ("X1", "X2", "Y"), mb.FD_NAME: ("X1", "M", "Y"),
         mb.MC_NAME: ("X1", "X2", "Y")}


class PilotComplete(Exception):
    """Explicit pilot-only sequential purchase cap reached."""


class BudgetExhausted(Exception):
    """Normal termination when no allowed arm can be purchased."""


def keyed_seed(*parts):
    encoded = json.dumps([PROTOCOL_ID, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:4], "little")


def canonical_arm(arm):
    return tuple(sorted(arm))


def domains(scm):
    return {v: (mb.MC_X2_BOX if scm == mb.MC_NAME and v == "X2" else mb.DOMAIN)
            for v in NODES[scm] if v != "Y"}


def all_arms(scm):
    names = sorted(domains(scm))
    return [a for n in range(1, len(names) + 1) for a in combinations(names, n)]


def sample_true(scm, iv, z):
    """One true SCM row, driven by four explicitly supplied standard normals."""
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


def observational_data(scm, seed):
    """A replicate-specific shared 100-row true-SCM observational dataset."""
    import pandas as pd
    rng = np.random.RandomState(keyed_seed(scm, seed, "observational"))
    return pd.DataFrame([sample_true(scm, {}, rng.randn(4)) for _ in range(100)],
                        columns=NODES[scm])


class Experiment:
    """Measurement-only learner interface; every purchased row is an event."""
    def __init__(self, scm, seed, arms, budget=None, n_init=3, max_purchases=None):
        self.scm, self.seed = scm, int(seed)
        self.max_purchases = max_purchases
        self.arms = sorted({canonical_arm(a) for a in arms}, key=lambda a: (len(a), a))
        self.budget = int(DEFAULT_BUDGET[scm] if budget is None else budget)
        self.n_init, self.cost, self.sequential_index = int(n_init), 0, 0
        if not self.arms or any(a not in all_arms(scm) for a in self.arms):
            raise ValueError("invalid arm family")
        if self.n_init < 1 or self.budget < self.n_init*sum(map(len, self.arms)):
            raise ValueError("budget cannot purchase the full declared initial design")
        self.events = []
        self.initial = {}
        for arm in self.arms:
            rng = np.random.RandomState(keyed_seed(scm, seed, "init-levels", arm))
            xs = np.column_stack([rng.uniform(*domains(scm)[v], self.n_init) for v in arm])
            self.initial[arm] = []
            for i, x in enumerate(xs):
                row = self._buy(arm, x, "init", keyed_seed(scm, seed, "init-noise", arm, i))
                self.initial[arm].append(deepcopy(row))

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

    def _buy(self, arm, values, phase, noise_seed):
        supplied = tuple(arm)
        x = np.asarray(values, dtype=float).ravel()
        if len(supplied) != len(x) or len(set(supplied)) != len(supplied):
            raise ValueError("intervention coordinates do not match arm")
        arm = canonical_arm(supplied)
        iv = dict(zip(supplied, map(float, x)))
        if arm not in self.arms:
            raise ValueError("unavailable arm")
        if not self.affordable(arm):
            raise BudgetExhausted()
        if any(not np.isfinite(v) or not domains(self.scm)[k][0]-1e-9 <= v <= domains(self.scm)[k][1]+1e-9 for k,v in iv.items()):
            raise ValueError("intervention outside declared domain")
        z = np.random.RandomState(noise_seed).randn(4)
        row = sample_true(self.scm, iv, z)
        self.cost += len(arm)
        event = dict(event_id=len(self.events), phase=phase, arm=list(arm),
                     x=[iv[v] for v in arm], measured=row, noise_seed=noise_seed,
                     cost=len(arm), cum_cost=self.cost)
        self.events.append(event)
        recommendation = min(self.events, key=lambda e: (e["measured"]["Y"], tuple(e["arm"]), e["event_id"]))
        event["recommendation_event_id"] = recommendation["event_id"]
        return deepcopy(row)

    def purchase(self, arm, values):
        self.check_available()
        row = self._buy(arm, values, "sequential",
                        keyed_seed(self.scm, self.seed, "sequential-noise", self.sequential_index))
        self.sequential_index += 1
        return row

    def target(self, arm, values):
        return self.purchase(arm, values)["Y"]

    def scalar_initial(self, arms):
        xs, ys = [], []
        for a in arms:
            rows = self.initial[canonical_arm(a)]
            xs.append(np.array([[r[v] for v in a] for r in rows]))
            ys.append(np.array([[r["Y"]] for r in rows]))
        best = min(self.events, key=lambda e: (e["measured"]["Y"], tuple(e["arm"]), e["event_id"]))
        # Reorder best point to the backend's variable ordering.
        backend_arm = next(a for a in arms if canonical_arm(a) == tuple(best["arm"]))
        iv = dict(zip(best["arm"], best["x"]))
        return xs, ys, np.array([iv[v] for v in backend_arm]), best["measured"]["Y"], "".join(backend_arm)


def score_events(scm, events, budget):
    """Post-run oracle evaluation of data-selected recommendations; never cummin μ."""
    oracle = mb.population_evaluator(scm)
    optimum = {mb.PP_NAME: 0., mb.FD_NAME: 0., mb.MC_NAME: mb.MC_SIGMA_2**2}[scm]
    result, best, oracle_best = [], None, float("inf")
    for event in events:
        key = lambda e: (e["measured"]["Y"], tuple(e["arm"]), e["event_id"])
        if best is None or key(event) < key(best):
            best = event
        if "recommendation_event_id" in event and event["recommendation_event_id"] != best["event_id"]:
            raise ValueError("logged recommendation disagrees with declared measured-data rule")
        value = float(oracle(best["arm"], best["x"]))
        oracle_best = min(oracle_best, float(oracle(event["arm"], event["x"])))
        result.append(dict(event_id=event["event_id"], cum_cost=event["cum_cost"],
                           recommendation_event_id=best["event_id"], recommendation_population=value,
                           recommendation_regret=value-optimum, oracle_best_visited=oracle_best))
    if not result:
        raise ValueError("empty experiment")
    checkpoints = []
    for c in range(12, int(budget)+1):
        eligible = [r for r in result if r["cum_cost"] <= c]
        if eligible:
            checkpoints.append(dict(cost=c, **{k:v for k,v in eligible[-1].items() if k != "cum_cost"}))
    return dict(final=result[-1], scored_events=result, checkpoints=checkpoints,
                cost_integrated_recommendation_regret=sum(r["recommendation_regret"] for r in checkpoints))


@contextmanager
def noisy_scalar_runtime(experiment):
    """Scoped opt-in likelihood and affordable-arm hooks; exact defaults preserved.

    Raw-target Gaussian variance is estimated, initialized at max(.01,.1 VarY),
    bounded [1e-6,1e6]. This homoscedastic approximation is explicitly recorded.
    """
    import importlib
    cbo = importlib.import_module("ccbo.cbo.cbo")
    bo = importlib.import_module("ccbo.cbo.bo")
    old_build, old_next, old_fix = cbo.update_BO_models, cbo.find_next_y_point, bo.fix_noise

    def learn_noise(model, value=None):
        model.likelihood.variance.unfix()
        model.likelihood.variance[:] = max(.01, .1*float(np.var(model.Y)))
        model.likelihood.variance.constrain_bounded(1e-6, 1e6, warning=False)
        if not hasattr(experiment, "gp_noise_audit"):
            experiment.gp_noise_audit = []
            experiment._noise_models = []
        experiment._noise_models.append(model)
        experiment.gp_noise_audit.append(dict(initial_variance=float(model.likelihood.variance), fixed=bool(model.likelihood.variance.is_fixed), policy="learned", n_rows=len(model.Y)))
        return model

    def build(*args, **kwargs):
        wrapper = old_build(*args, **kwargs)
        learn_noise(wrapper.model)
        # Vendored CBO rebuilds the selected GP before acquisition, discarding
        # its previous post-response fit. Fit the newly rebuilt noisy GP here.
        wrapper.optimize()
        audit = experiment.gp_noise_audit[-1]
        audit["fitted_before_acquisition"] = True
        audit["fitted_variance"] = float(wrapper.model.likelihood.variance)
        audit["fitted_parameters"] = wrapper.model.param_array.tolist()
        audit["acquisition_snapshots"] = []
        wrapper._matched_noise_audit = audit
        return wrapper

    def next_point(space, model, current, arm, costs, task="min"):
        experiment.check_available()
        if not experiment.affordable(arm):
            return np.full((1, 1), -np.inf), np.zeros((1, len(arm)))
        audit = getattr(model, "_matched_noise_audit", None)
        if audit is None or not audit.get("fitted_before_acquisition"):
            raise RuntimeError("noisy CBO acquisition received an unfitted model")
        parameters = model.model.param_array.tolist()
        if parameters != audit["fitted_parameters"]:
            raise RuntimeError("CBO acquisition parameters differ from pre-acquisition fit")
        audit["acquisition_snapshots"].append(dict(
            arm=list(arm), n_rows=len(model.model.Y),
            variance=float(model.model.likelihood.variance), parameters=parameters))
        return old_next(space, model, current, arm, costs, task=task)

    cbo.update_BO_models, cbo.find_next_y_point, bo.fix_noise = build, next_point, learn_noise
    try:
        yield
    finally:
        cbo.update_BO_models, cbo.find_next_y_point, bo.fix_noise = old_build, old_next, old_fix


def run_scalar(scm, cond, method, seed, budget=None, n_init=3, max_purchases=None):
    """Run the existing CBO or joint BO backend on the matched environment."""
    from ccbo.scm_graphs import get_original_graph
    from ccbo.coarsened_graph import CoarsenedGraph
    from ccbo.cbo.utils import compute_coverage
    from ccbo.cbo.cbo import CBO
    from ccbo.cbo.bo import NonCausal_BO
    if method not in ("CBO", "QCBO", "BO-S", "BO", "CBO-NP"):
        raise ValueError(method)
    mb.register_variants()
    algorithm_seed = keyed_seed(scm, seed, "algorithm")
    np.random.seed(algorithm_seed)
    obs = observational_data(scm, seed)
    graph = get_original_graph(scm, obs)
    partition = mb.fine_partition(scm) if method in ("CBO", "CBO-NP") else mb.coarse_partition(scm)
    cg = CoarsenedGraph(graph, partition, scm, obs, num_mc_samples=2000,
                        assumed_graph_name=mb.variant_name(cond))
    arms, _, manip = cg.get_sets()
    if method == "BO-S":
        arms = [list(a) for a in all_arms(scm)]
    if method == "BO":
        arms = [sorted(manip)]
    functions = cg.fit_all_models()
    ranges, costs = cg.get_interventional_ranges(), cg.get_cost_structure(1)
    _, _, coverage = compute_coverage(obs, manip, ranges)
    experiment = Experiment(scm, seed, arms, budget, n_init, max_purchases)
    xs, ys, bestx, besty, bestarm = experiment.scalar_initial(arms)
    with noisy_scalar_runtime(experiment):
        try:
            if method == "BO":
                NonCausal_BO(experiment.budget, cg, ranges, xs[0], ys[0], costs,
                            obs, functions, bestx, besty, arms[0], Causal_prior=False,
                            target_evaluator=experiment.target)
            else:
                CBO(experiment.budget, arms, manip, xs, ys, bestx, besty, bestarm,
                    ranges, functions, obs, coverage, cg, 20, costs, obs,
                    "min", 100, 100, n_init, Causal_prior=method in ("CBO", "QCBO"),
                    target_evaluator=experiment.target, force_observe_on_entry=False)
        except (BudgetExhausted, PilotComplete):
            pass
    for model, audit in zip(getattr(experiment, "_noise_models", []), getattr(experiment, "gp_noise_audit", [])):
        audit["final_variance"] = float(model.likelihood.variance)
        audit["fixed_after_fit"] = bool(model.likelihood.variance.is_fixed)
    if hasattr(experiment, "_noise_models"):
        del experiment._noise_models
    experiment.check_available_if_complete = not any(experiment.affordable(a) for a in experiment.arms)
    if not experiment.check_available_if_complete and not (max_purchases is not None and experiment.sequential_index == max_purchases):
        raise RuntimeError("backend stopped with affordable actions remaining")
    return experiment, dict(backend="vendored-CBO" if method != "BO" else "vendored-BO",
                            algorithm_seed=algorithm_seed, new_observation_rows=0,
                            gp_noise_audit=getattr(experiment, "gp_noise_audit", []),
                            noise_policy="learned raw-target homoscedastic variance; init=max(.01,.1*VarY), bounds=[1e-6,1e6]")
