"""Isolated opt-in bridge to pinned CEO: one measurement, no hidden oracle.

Committed singleton graphs are used here. This module never imports or calls
population scorers. The upstream second ('noiseless') callback replays the same
purchased row, so all internal incumbent/acquisition state is learner observable.
"""
from copy import deepcopy
import os
import random
import types

import numpy as np
from ccbo.matched_protocol import Experiment, BudgetExhausted, PilotComplete, observational_data, all_arms, canonical_arm, keyed_seed, domains


def run_ceo(scm, cond, seed, budget=None, n_init=3, anchors=35, ceo_root=None, max_purchases=None):
    from ccbo.baselines import ceo as ceo_adapter
    if ceo_root:
        ceo_adapter.CEO_ROOT = os.path.abspath(ceo_root)
    ceo_adapter.ensure_ceo()
    # Reuse graph/SEM declarations only, never its initialization/target/scorer.
    from ccbo.baselines import ceo as definitions
    import src.bases.ceo_base as base
    from src.methods.ceo import CEO
    from src.utils.sem_utils.sem_estimate import build_sem_hat
    spec = definitions.SCMS[scm]
    canonical_cond = spec["aliases"].get(cond, cond)
    graph = definitions.variant_graph(scm, canonical_cond)
    experiment = Experiment(scm, seed, all_arms(scm), budget, n_init, max_purchases)
    algorithm_seed = keyed_seed(scm, seed, "algorithm")
    np.random.seed(algorithm_seed)
    random.seed(algorithm_seed)
    pending = {}
    audit = dict(measurement_calls=0, bookkeeping_replays=0)

    def factory(noisy, random_state, initial_structural_equation_model,
                structural_equation_model, graph, exploration_set, all_vars, T):
        arm = tuple(exploration_set)
        def target(current_target, intervention_levels, assigned_blanket):
            levels = np.asarray(intervention_levels, dtype=float).ravel()
            key = (arm, levels.tobytes())
            if noisy:
                if pending:
                    raise RuntimeError("CEO requested a second measurement before replay")
                sample = experiment.purchase(arm, levels)
                pending[key] = deepcopy(sample)
                audit["measurement_calls"] += 1
            else:
                if key not in pending:
                    raise RuntimeError("unpaired CEO bookkeeping callback")
                sample = pending.pop(key)
                audit["bookkeeping_replays"] += 1
            return {k: np.array([v]) for k,v in sample.items()}
        return target

    initial = {a: {v: np.array([[r[v]] for r in rows]) for v in spec["nodes"]}
               for a, rows in experiment.initial.items()}
    class SEM:
        @staticmethod
        def static(): return spec["static"]
        @staticmethod
        def dynamic(): return None

    obs = observational_data(scm, seed)
    d_obs = {v:obs[v].to_numpy()[:,None] for v in spec["nodes"]}
    old_factory = base.evaluate_target_function_all_for_ceo
    base.evaluate_target_function_all_for_ceo = factory
    try:
        ceo = CEO(graphs=[graph], init_posterior=[1.], sem=SEM,
                  make_sem_estimator=build_sem_hat, observation_samples=d_obs,
                  intervention_domain={k:list(v) for k,v in domains(scm).items()},
                  intervention_samples=deepcopy(initial), intervention_samples_noiseless=deepcopy(initial),
                  exploration_sets=list(experiment.arms), number_of_trials=experiment.budget+1,
                  base_target_variable="Y", task="min", estimate_sem=True,
                  num_anchor_points=anchors, sample_anchor_points=True,
                  seed_anchor_points=1+keyed_seed(scm, seed, "ceo-anchors") % 1_000_000, seed=algorithm_seed,
                  manipulative_variables=list(spec["manip"]),
                  random_state=np.random.RandomState(keyed_seed(scm, seed, "ceo-internal")))
        original_trial = ceo._per_trial_computations
        original_acq = ceo._evaluate_acquisition_functions
        def trial(self, *args, **kwargs):
            experiment.check_available()
            return original_trial(*args, **kwargs)
        def acquisition(self, *args, **kwargs):
            result = original_acq(*args, **kwargs)
            for arm in self.y_acquired:
                if not experiment.affordable(arm):
                    self.y_acquired[arm] = -np.inf
            return result
        ceo._per_trial_computations = types.MethodType(trial, ceo)
        ceo._evaluate_acquisition_functions = types.MethodType(acquisition, ceo)
        try:
            ceo.run()
        except (BudgetExhausted, PilotComplete):
            pass
        if pending or audit["measurement_calls"] != audit["bookkeeping_replays"]:
            raise RuntimeError("CEO measurement replay mismatch")
        if any(experiment.affordable(a) for a in experiment.arms) and not (max_purchases is not None and experiment.sequential_index == max_purchases):
            raise RuntimeError("CEO stopped before exhausting affordable budget")
        # An integration canary checks actual final GP training labels, not just callbacks.
        gp_y = {}
        for arm in experiment.arms:
            expected = [e["measured"]["Y"] for e in experiment.events if tuple(e["arm"]) == arm]
            actual = np.asarray(ceo.interventional_data_y[0][arm]).ravel()
            if not np.array_equal(actual, np.array(expected)):
                raise RuntimeError(f"CEO training-label mismatch on {arm}")
            gp_y["+".join(arm)] = actual.tolist()
        return experiment, dict(backend="pinned-CEO-committed", algorithm_seed=algorithm_seed,
                                pool=[canonical_cond], asserted_cond=cond,
                                observable_alias_of=canonical_cond if canonical_cond != cond else None,
                                true_graph_representable=scm == "MediatedChain",
                                graph_edges=[list(e) for e in graph.edges()],
                                noise_policy="native CEO learned Gaussian likelihood",
                                num_anchor_points=anchors, new_observation_rows=0,
                                feedback_audit=audit, final_gp_y=gp_y,
                                gp_noise_audit={"+".join(a): {"variance":float(ceo.bo_model[0][a].model.likelihood.variance), "fixed":bool(ceo.bo_model[0][a].model.likelihood.variance.is_fixed)} for a in experiment.arms})
    finally:
        base.evaluate_target_function_all_for_ceo = old_factory
