"""Graph-free HQCBO for MediatedChain, using resumed vendored CBO.

The original quotient's cached priors survive; newly exposed singleton arms use
plain GPs. After exposure the backend receives a graph-access guard. Refinement
uses measured incumbents only and pays the complete split design before use.
"""
import numpy as np
from ccbo import minibench as mb
from ccbo.matched_protocol import (Experiment, BudgetExhausted, PilotComplete,
    canonical_arm, domains, keyed_seed, scalar_runtime, observational_data, FEEDBACK_MODE)


class NoGraphAccess:
    def __getattr__(self, name):
        raise AssertionError(f"post-split graph access forbidden: {name}")


class RefinementExperiment(Experiment):
    def expose(self, new_arms):
        """Atomic affordability check, then independently keyed paid design rows."""
        new_arms = sorted({canonical_arm(a) for a in new_arms} - set(self.arms), key=lambda a:(len(a),a))
        needed = self.n_init*sum(map(len,new_arms))
        if needed > self.remaining:
            return None
        # Validate before changing the allowed menu or charging any rows.
        from ccbo.matched_protocol import all_arms
        if any(a not in all_arms(self.scm) for a in new_arms):
            raise ValueError("invalid exposed arm")
        self.arms = sorted(set(self.arms) | set(new_arms), key=lambda a:(len(a),a))
        rows = {}
        for arm in new_arms:
            rng=np.random.RandomState(keyed_seed(self.scm,self.seed,"split-levels",arm))
            xs=np.column_stack([rng.uniform(*domains(self.scm)[v],self.n_init) for v in arm])
            rows[arm]=[]
            for x in xs:
                rows[arm].append(self._buy(arm,x,"split_init"))
        return rows


def extend_state(state, arms, experiment, new_rows, ranges):
    """Append only genuinely purchased arm data; retain old GP/prior history."""
    from emukit.core import ParameterSpace, ContinuousParameter
    old_arms=[list(a) for a in arms]
    for arm,rows in new_rows.items():
        label="".join(arm)
        x=np.array([[r[v] for v in arm] for r in rows])
        y=np.array([[r["Y"]] for r in rows])
        arms.append(list(arm))
        state["dict_interventions"].append(label)
        state["current_best_x"][label]=[x[int(np.argmin(y))].copy()]
        state["current_best_y"][label]=y.ravel().tolist()
        state["x_dict_mean"][label]={}
        state["x_dict_var"][label]={}
        state["data_x_list"].append(x)
        state["data_y_list"].append(y)
        state["target_function_list"].append(lambda value,a=arm: np.asarray(experiment.target(a,value),dtype=float)[None,None])
        state["space_list"].append(ParameterSpace([ContinuousParameter(v,*ranges[v]) for v in arm]))
        state["model_list"].append(None)
        state["mean_functions_list"].append(None)
        state["var_functions_list"].append(None)
        state["prior_mask"].append(False)
    state["global_opt"][-1]=min(e["measured"]["Y"] for e in experiment.events)
    state["cumulative_cost"] += experiment.n_init*sum(map(len,new_rows))
    state["current_cost"][-1]=state["cumulative_cost"]
    state["force_rebuild_all"]=True
    state["models_fresh"]=False
    return old_arms


def run_hqcbo(scm, cond, seed, budget=None, n_init=3, max_purchases=None):
    if scm != mb.MC_NAME or cond != "C0":
        raise ValueError("matched HQCBO currently supports only MediatedChain/C0")
    from ccbo.run_experiment import get_original_graph
    from ccbo.coarsened_graph import CoarsenedGraph
    from ccbo.cbo.utils import compute_coverage
    from ccbo.cbo.cbo import CBO
    mb.register_variants()
    algorithm_seed=keyed_seed(scm,seed,"algorithm")
    np.random.seed(algorithm_seed)
    obs=observational_data(scm,seed)
    graph=get_original_graph(scm,obs)
    cg=CoarsenedGraph(graph,mb.coarse_partition(scm),scm,obs,num_mc_samples=2000,assumed_graph_name=mb.variant_name(cond))
    arms,_,manip=cg.get_sets()
    arms=[list(a) for a in arms]
    functions=cg.fit_all_models()
    ranges,costs=cg.get_interventional_ranges(),cg.get_cost_structure(1)
    _,_,coverage=compute_coverage(obs,manip,ranges)
    experiment=RefinementExperiment(scm,seed,arms,budget,n_init,max_purchases)
    xs,ys,bestx,besty,bestarm=experiment.scalar_initial(arms)
    history=[besty]
    state=None
    split=None
    prior_mask=[True]*len(arms)
    backend_graph=cg
    with scalar_runtime(experiment):
        try:
            while True:
                experiment.check_available()
                _,state=CBO(1,arms,manip,xs,ys,bestx,besty,bestarm,ranges,functions,
                    obs,coverage,backend_graph,20,costs,obs,"min",100,100,n_init,
                    Causal_prior=prior_mask,target_evaluator=experiment.target,
                    force_observe_on_entry=False,state=state,return_state=True)
                history.append(min(e["measured"]["Y"] for e in experiment.events))
                experiment.check_available()  # do not expand after an explicit pilot cap
                if split is None and _plateau(history,5,1e-3,"min"):
                    # Supplied hierarchy alone defines children; no graph MIS query.
                    hierarchy=mb.MC_REFINE_MAP[frozenset({"X1","X2"})]
                    from itertools import combinations
                    new_arms=[tuple(sorted(set().union(*combo))) for k in range(1,len(hierarchy)+1) for combo in combinations(hierarchy,k)]
                    before=len(experiment.events)
                    new_rows=experiment.expose(new_arms)
                    split=dict(trigger_sequential_index=experiment.sequential_index,
                               trigger_event_id=before-1, plateau_k=5,plateau_delta=1e-3,
                               trigger_measurements=list(history[-6:]),
                               accepted=new_rows is not None,split_cost=0,split_arms=[])
                    if new_rows is not None:
                        extend_state(state,arms,experiment,new_rows,ranges)
                        prior_mask=list(state["prior_mask"])
                        split.update(split_cost=sum(e["cost"] for e in experiment.events[before:]),
                                     split_arms=[list(a) for a in new_rows],
                                     split_event_ids=list(range(before,len(experiment.events))),
                                     recommendation_after_split=experiment.events[-1]["recommendation_event_id"])
                        backend_graph=NoGraphAccess()
                        # The graph object is not consulted again, even to rebuild priors.
        except (BudgetExhausted,PilotComplete):
            pass
    if any(experiment.affordable(a) for a in experiment.arms) and not (max_purchases is not None and experiment.sequential_index==max_purchases):
        raise RuntimeError("HQCBO stopped with affordable actions remaining")
    return experiment,dict(backend="vendored-CBO-resumed-graphfree-refinement",
        algorithm_seed=algorithm_seed,new_observation_rows=0,refinement=split,
        prior_mask=prior_mask,final_arms=arms,gp_noise_audit=getattr(experiment,"gp_noise_audit",[]),
        feedback_mode=FEEDBACK_MODE,
        noise_policy="fixed interventional likelihood variance (engine NOISE_VAR); exact population feedback",
        post_split_graph_access="guarded; cached original-quotient priors only")


def _plateau(traj, k, delta, task):
    """True when the best-so-far trajectory improved < ``delta`` over the last
    ``k`` steps."""
    if len(traj) < k + 1:
        return False
    gain = (traj[-1 - k] - traj[-1]) if task == 'min' else (traj[-1] - traj[-1 - k])
    return gain < delta
