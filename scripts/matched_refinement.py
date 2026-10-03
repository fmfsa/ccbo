"""Graph-free noisy HQCBO (staged hierarchy, mixed unions), using resumed vendored CBO.

The original quotient's cached priors survive; newly exposed singleton arms use
plain GPs. After exposure the backend receives a graph-access guard. Refinement
uses measured incumbents only and pays the complete split design before use.
"""
import numpy as np
from ccbo import minibench as mb
from ccbo.matched_protocol import (Experiment, BudgetExhausted, PilotComplete,
    canonical_arm, domains, keyed_seed, scalar_runtime, observational_data)


class RefinementExperiment(Experiment):
    def expose(self, new_arms, design_points=0, charged=True):
        """Add newly exposed arms. By default they start from a plain prior with
        no data (refinement adds actions, not free information); a paid design
        of ``design_points`` rows per arm is bought atomically if requested."""
        new_arms = sorted({canonical_arm(a) for a in new_arms} - set(self.arms), key=lambda a:(len(a),a))
        needed = design_points*sum(map(len,new_arms)) if charged else 0
        if needed > self.remaining:
            return None
        # Validate before changing the allowed menu or charging any rows.
        from ccbo.matched_protocol import all_arms
        if any(a not in all_arms(self.scm) for a in new_arms):
            raise ValueError("invalid exposed arm")
        self.arms = sorted(set(self.arms) | set(new_arms), key=lambda a:(len(a),a))
        rows = {}
        for arm in new_arms:
            rows[arm]=[]
            if not design_points:
                continue
            rng=np.random.RandomState(keyed_seed(self.scm,self.seed,"split-levels",arm))
            xs=np.column_stack([rng.uniform(*domains(self.scm)[v],design_points) for v in arm])
            for x in xs:
                rows[arm].append(self._buy(arm,x,"split_init",charged=charged))
        return rows


def extend_state(state, arms, experiment, new_rows, ranges, prior_centre=None, plain=False):
    """Append only genuinely purchased arm data; retain old GP/prior history."""
    from emukit.core import ParameterSpace, ContinuousParameter
    old_arms=[list(a) for a in arms]
    for arm,rows in new_rows.items():
        label="".join(arm)
        x=np.array([[r[v] for v in arm] for r in rows]).reshape(len(rows),len(arm))
        y=np.array([[r["Y"]] for r in rows]).reshape(len(rows),1)
        arms.append(list(arm))
        state["dict_interventions"].append(label)
        # Without a split design the new arm starts from its plain prior.
        state["current_best_x"][label]=[x[int(np.argmin(y))].copy()] if len(rows) else []
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
        # Graph-free uninformative prior: the observational mean and variance of
        # Y (the fallback the paper uses for arms without identified effects).
        if plain:
            # Engine's plain GP prior (zero mean, unit RBF), no graph and no data.
            state["prior_mask"].append(False)
            continue
        y=np.asarray(state["observational_samples"]["Y"],dtype=float)
        mu,var=float(y.mean()),float(y.var())
        if prior_centre is not None: mu=float(prior_centre)
        state.setdefault("fixed_priors",{})[len(arms)-1]=(
            lambda X,mu=mu:np.full((np.atleast_2d(X).shape[0],1),mu),
            lambda X,var=var:np.full((np.atleast_2d(X).shape[0],1),var))
        state["prior_mask"].append(True)
    state["global_opt"][-1]=min(e["measured"]["Y"] for e in experiment.events)
    state["cumulative_cost"] += experiment.n_init*sum(map(len,new_rows))
    state["current_cost"][-1]=state["cumulative_cost"]
    state["force_rebuild_all"]=True
    state["models_fresh"]=False
    return old_arms


def apply_stage(clusters, stage):
    """Split every current cluster named in ``stage`` into its children."""
    out=[]
    for c in clusters:
        out.extend(stage[c] if c in stage else [c])
    return out


def mixed_unions(clusters):
    """All nonempty unions of the given (disjoint) manipulable clusters, as
    canonical arms; mixed unions across different split clusters included."""
    from itertools import combinations
    return [canonical_arm(set().union(*combo)) for k in range(1,len(clusters)+1)
            for combo in combinations(clusters,k)]


NEW_ARM_MODES=("fallback","free_design","incumbent","plain")


def run_hqcbo(scm, cond, seed, budget=None, n_init=None, max_purchases=None, partition_id="coarse",
              plateau_k=5, plateau_delta=1e-3, new_arm_mode="plain", max_stages=None):
    """Graph-free staged refinement from ``partition_id`` along ``mb.REFINE_STAGES``.

    A stage fires when the measured incumbent improves by less than
    ``plateau_delta`` over the last ``plateau_k`` sequential measurements since
    the previous stage. It splits the stage's clusters and exposes every
    previously unavailable nonempty union of the *current* clusters (mixed
    unions included). New arms get no data; ``new_arm_mode`` sets their prior:
    "plain" (default; the engine's zero-mean plain GP, as for QCBO-NP and BO-S),
    "fallback" (observational mean and variance of Y), "incumbent" (fallback
    variance centred at the incumbent), or "free_design" (fallback prior plus
    ``n_init`` uncharged points). Original arms keep their quotient priors,
    refreshed from the initial quotient when observing.
    """
    if scm not in mb.REFINE_STAGES:
        raise ValueError(f"no refinement hierarchy declared for {scm}")
    from ccbo.cbo.utils import compute_coverage
    from ccbo.cbo.cbo import CBO
    from ccbo.matched_protocol import build_graph
    mb.register_variants()
    algorithm_seed=keyed_seed(scm,seed,"algorithm")
    np.random.seed(algorithm_seed)
    from ccbo.matched_protocol import N_OBS_INITIAL, N_OBS_POOL, N_OBS_BATCH
    pool=observational_data(scm,seed,N_OBS_POOL)
    obs=pool.iloc[:N_OBS_INITIAL].copy()
    cg,arms,manip,pid=build_graph(scm,cond,"QCBO",obs,partition_id)
    functions=cg.fit_all_models()
    ranges,costs=cg.get_interventional_ranges(),cg.get_cost_structure(1)
    _,_,coverage=compute_coverage(obs,manip,ranges)
    experiment=RefinementExperiment(scm,seed,arms,budget,n_init,max_purchases,
                                    null_estimate=float(obs["Y"].mean()))
    xs,ys,bestx,besty,bestarm=experiment.scalar_initial(arms)
    clusters=[c for c in mb.partition(scm,pid) if c!=frozenset({"Y"})]
    stages=list(mb.REFINE_STAGES[scm])[:max_stages]
    history=[besty]
    since=0                       # history index where the current stage began
    refinements=[]
    state=None
    prior_mask=[True]*len(arms)
    prior_mask_initial=list(prior_mask)
    backend_graph=cg
    with scalar_runtime(experiment):
        try:
            while True:
                experiment.check_available()
                purchases=experiment.sequential_index
                _,state=CBO(1,arms,manip,xs,ys,bestx,besty,bestarm,ranges,functions,
                    obs,coverage,backend_graph,N_OBS_BATCH,costs,pool,"min",N_OBS_POOL,N_OBS_INITIAL,experiment.n_init,
                    Causal_prior=prior_mask,target_evaluator=experiment.target,
                    force_observe_on_entry=None,state=state,return_state=True)
                if experiment.sequential_index==purchases:
                    continue   # an observe trial: no new measurement for the plateau rule
                # Incumbent value: best measurement, or the null intervention's estimate.
                history.append(min([e["measured"]["Y"] for e in experiment.events]+
                                   ([experiment.null_estimate] if experiment.null_estimate is not None else [])))
                experiment.check_available()  # do not expand after an explicit pilot cap
                declined=refinements and not refinements[-1]["accepted"]
                if stages and not declined and _plateau(history[since:],plateau_k,plateau_delta,"min"):
                    # Supplied hierarchy alone defines children; no graph query.
                    stage=stages.pop(0)
                    clusters=apply_stage(clusters,stage)
                    new_arms=[a for a in mixed_unions(clusters) if a not in experiment.arms]
                    before=len(experiment.events)
                    if new_arm_mode=="free_design":
                        new_rows=experiment.expose(new_arms,design_points=experiment.n_init,charged=False) if new_arms else {}
                    else:
                        new_rows=experiment.expose(new_arms) if new_arms else {}
                    record=dict(stage=len(refinements),trigger_sequential_index=experiment.sequential_index,
                                trigger_event_id=before-1,plateau_k=plateau_k,plateau_delta=plateau_delta,
                                trigger_measurements=list(history[-(plateau_k+1):]),
                                split_clusters=[sorted(c) for c in stage],
                                partition_after=[sorted(c) for c in clusters],
                                accepted=new_rows is not None,split_cost=0,split_arms=[])
                    if new_rows:
                        centre=(min(e["measured"]["Y"] for e in experiment.events)
                                if new_arm_mode=="incumbent" else None)
                        extend_state(state,arms,experiment,new_rows,ranges,prior_centre=centre,
                                     plain=new_arm_mode=="plain")
                        prior_mask=list(state["prior_mask"])
                        record.update(split_cost=sum(e["cost"] for e in experiment.events[before:]),
                                      split_arms=[list(a) for a in new_rows],
                                      split_event_ids=list(range(before,len(experiment.events))),
                                      recommendation_after_split=experiment.events[-1]["recommendation_event_id"])
                        # Original arms keep their quotient priors (refreshed from the
                        # initial quotient when observing); new arms never touch the graph.
                    refinements.append(record)
                    since=len(history)-1
        except (BudgetExhausted,PilotComplete):
            pass
    if any(experiment.affordable(a) for a in experiment.arms) and not (max_purchases is not None and experiment.sequential_index==max_purchases):
        raise RuntimeError("HQCBO stopped with affordable actions remaining")
    return experiment,dict(backend="vendored-CBO-resumed-graphfree-refinement",
        algorithm_seed=algorithm_seed,partition=pid,observation_log=experiment.observation_log,
        new_observation_rows=(experiment.observation_log[-1]["n_rows_after"]-N_OBS_INITIAL
                              if experiment.observation_log else 0),
        refinement=refinements[0] if refinements else None,refinements=refinements,
        prior_mask=prior_mask,final_arms=arms,gp_noise_audit=getattr(experiment,"gp_noise_audit",[]),
        arms=[list(a) for a in arms[:len(prior_mask_initial)]],
        noise_policy="fixed interventional likelihood variance (engine NOISE_VAR); exact population feedback",
        post_split_graph_access="initial quotient only, for original arms; new arms use the observational fallback prior")


def _plateau(traj, k, delta, task):
    """True when the best-so-far trajectory improved < ``delta`` over the last
    ``k`` steps."""
    if len(traj) < k + 1:
        return False
    gain = (traj[-1 - k] - traj[-1]) if task == 'min' else (traj[-1] - traj[-1 - k])
    return gain < delta
