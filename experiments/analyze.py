#!/usr/bin/env python3
"""Audit portable experiment results and summarize complete paper matrices.

Subsets are descriptive only. Primary outcomes use learner-selected recommendations,
never oracle best-visited values. No server paths or frozen deployment manifests.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b):
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9)
    except (ValueError,TypeError):
        return False


def finite(value, nonnegative=False):
    return isinstance(value,(int,float)) and math.isfinite(value) and (not nonnegative or value>=0)

def population(scm, arm, x):
    """Independent closed forms for the frozen three SCMs, including natural variance."""
    iv = dict(zip(arm,x))
    if scm == "ParallelParent":
        return ((iv["X1"]-2)**2 if "X1" in iv else 4.25) + ((iv["X2"]+2)**2 if "X2" in iv else 4.25)
    if scm == "FrontDoor":
        mu = iv["M"] if "M" in iv else 2*iv.get("X1",0.)
        variance = 0. if "M" in iv else .04 if "X1" in iv else 1.04
        return 2*(1-.3/math.sqrt(.09+variance)*math.exp(-(mu-1)**2/(2*(.09+variance))))
    if scm == "MediatedChain":
        return (iv["X2"]-4)**2 if "X2" in iv else (2*iv.get("X1",0.)-4)**2 + (.04 if "X1" in iv else 1.04)
    raise ValueError("unsupported SCM "+scm)

def static_events(unit, config, events, scores, summary, replication, nominal_budget=False):
    scm=unit["scm"]; budget=config["budget"]
    require(events and isinstance(events,list), "empty event log")
    cost=0; best=None; init_cost=0; split_cost=0; sequential=0; scored=[]; oracle_best=math.inf
    initial_counts={}; split_counts={}; split_ids=[]; sequential_started=False
    nodes = {"X1","M"} if scm == "FrontDoor" else {"X1","X2"}
    optimum=.04 if scm == "MediatedChain" else 0.
    for index,e in enumerate(events):
        arm=e["arm"]; x=e["x"]
        require(e["event_id"] == index, "event IDs must be consecutive")
        require(arm == sorted(set(arm)) and set(arm) <= nodes and 0 < len(arm)==len(x), "invalid canonical arm")
        require(e["phase"] in ("init","split_init","sequential"), "invalid event phase")
        for v,z in zip(arm,x):
            bound=2 if scm=="MediatedChain" and v=="X2" else 3
            require(math.isfinite(z) and -bound-1e-9 <= z <= bound+1e-9, "invalid intervention domain")
            require(close(e["measured"][v],z), "measured clamped coordinate mismatch")
        require(set(e["measured"]) == nodes|{"Y"} and all(math.isfinite(v) for v in e["measured"].values()), "invalid measured row")
        require(e["cost"] == len(arm), "purchase cost != arm size")
        cost += len(arm)
        require(e["cum_cost"] == cost and cost <= budget, "cumulative cost mismatch or overshoot")
        if e["phase"]=="init":
            require(not sequential_started, "initialization after sequential purchase")
            init_cost+=len(arm); initial_counts[tuple(arm)]=initial_counts.get(tuple(arm),0)+1
        elif e["phase"]=="split_init":
            require(unit["method"]=="HQCBO" and scm=="MediatedChain" and unit["cond"]=="C0",
                    "split initialization outside declared refinement method")
            require(sequential_started and tuple(arm) not in initial_counts,"split arm was already initialized")
            require(len(arm)==1,"refinement exposes singleton arms only")
            split_cost+=len(arm); split_counts[tuple(arm)]=split_counts.get(tuple(arm),0)+1; split_ids.append(index)
        else:
            sequential_started=True; sequential+=1
            require(initial_counts.get(tuple(arm),split_counts.get(tuple(arm),0))==config["n_init"],
                    "sequential arm lacks complete purchased initial design")
        key=lambda r:(r["measured"]["Y"],tuple(r["arm"]),r["event_id"])
        if best is None or key(e)<key(best): best=e
        require(e.get("recommendation_event_id") == best["event_id"], "recommendation is not measured-data winner")
        value=population(scm,best["arm"],best["x"])
        oracle_best=min(oracle_best,population(scm,arm,x))
        scored.append(dict(event_id=index,cum_cost=cost,recommendation_event_id=best["event_id"],
                           recommendation_population=value,recommendation_regret=value-optimum,
                           oracle_best_visited=oracle_best))
    require(all(n==config["n_init"] for n in initial_counts.values()), "initial count does not match protocol")
    require(summary.get("split_init_cost",0)==split_cost,"split initialization cost mismatch")
    if split_ids:
        require(split_ids==list(range(split_ids[0],split_ids[-1]+1)) and split_cost==6 and
                split_counts=={("X1",):3,("X2",):3},"split design is not the complete contiguous atomic design")
    refinement=summary.get("backend",{}).get("refinement")
    if unit["method"]=="HQCBO" and refinement is not None:
        trigger=refinement["trigger_event_id"]
        require(0<=trigger<len(events) and events[trigger]["phase"]=="sequential","invalid split trigger")
        require(refinement["trigger_sequential_index"]==sum(e["phase"]=="sequential" for e in events[:trigger+1]),"split trigger index mismatch")
        if refinement["accepted"]:
            require(bool(split_ids) and split_ids[0]==trigger+1 and refinement["split_event_ids"]==split_ids and
                    refinement["split_cost"]==6 and sorted(refinement["split_arms"])==[["X1"],["X2"]],"accepted split metadata mismatch")
            require(refinement["recommendation_after_split"]==events[split_ids[-1]]["recommendation_event_id"],"post-split recommendation mismatch")
        else:
            require(not split_ids and refinement["split_cost"]==0 and budget-events[trigger]["cum_cost"]<6,
                    "declined split must be unaffordable and uncharged")
    else:
        require(not split_ids,"split events missing refinement metadata")
    require(summary["actual_cost"]==cost and summary["init_cost"]==init_cost and summary["n_purchases"]==len(events)
            and summary["sequential_purchases"]==sequential, "summary ledger counts mismatch")
    if replication: require(cost==budget, "replication has not reached common budget B")
    elif nominal_budget:
        cheapest=min(len(arm) for arm in set(initial_counts)|set(split_counts))
        require(0 <= budget-cost < cheapest, "nominal-budget pilot still has affordable actions")
    else: require(sequential==config["max_purchases"], "pilot purchase count mismatch")
    def same_row(a,b):
        require(set(a)==set(b), "score row schema mismatch")
        for k,v in b.items(): require(close(a[k],v), "score/recommendation mismatch: "+k)
    require(len(scores["scored_events"])==len(scored), "score event count mismatch")
    for a,b in zip(scores["scored_events"],scored): same_row(a,b)
    same_row(scores["final"],scored[-1]); same_row(summary["final"],scored[-1])
    checkpoints=[]
    for c in range(12,cost+1):
        row=next(r for r in reversed(scored) if r["cum_cost"]<=c)
        checkpoints.append(dict(cost=c,**{k:v for k,v in row.items() if k!="cum_cost"}))
    require(len(scores["checkpoints"])==len(checkpoints), "cost checkpoints mismatch")
    for a,b in zip(scores["checkpoints"],checkpoints): same_row(a,b)
    area=sum(r["recommendation_regret"] for r in checkpoints)
    require(close(scores["cost_integrated_recommendation_regret"],area) and
            close(summary["cost_integrated_recommendation_regret"],area), "cost area mismatch")
    return dict(final_recommendation_regret=scored[-1]["recommendation_regret"],
                cost_integrated_recommendation_regret=area, actual_cost=cost,
                purchases=len(events), sequential_purchases=sequential)

def targets(env,menu):
    if env=='ToyGraph':
        return [[1,0,0],[0,1,0],[1,1,0],[0,0,0]] if menu=='full' else [[1,1,0],[0,0,0]]
    return [[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,1,1,0,0]] if menu=='full' else [[0,0,1,1,0,0]]

def mcbo_events(info,events,unit):
    env=unit['env_name']; n=3 if env=='ToyGraph' else 6
    allowed=targets(env,unit['menu']); allowed_keys={tuple(t) for t in allowed}
    require(info['targets']==allowed and info['n_targets']==len(allowed),'Wrong target order/menu')
    counts={}; sequential=[]; init={}; eligible=[]; best=None; cost=0; init_cost=0; means=[]
    sequential_started=False
    for idx,e in enumerate(events):
        target=e['target']; x=e['X']; physical=e['physical_values']; y=e['network_observation']
        require(len(target)==n and all(t in (0,1) for t in target),'Invalid target mask')
        require(len(x)==2*n and x[:n]==target and len(physical)==n and len(y)==n,'Action/data shape mismatch')
        require(all(math.isfinite(v) for v in x+physical+y),'Nonfinite action/data')
        require(all(-1e-9<=v<=1+1e-9 for v in x[n:]),'Action outside normalized domain')
        mapped=x[n:].copy()
        if env=='ToyGraph': mapped[0]=10*mapped[0]-5; mapped[1]=25*mapped[1]-5
        require(all(close(a,b) for a,b in zip(physical,mapped)),'Physical mapping mismatch')
        require(all(close(y[k],physical[k]) for k,t in enumerate(target) if t),'Environment clamp mismatch')
        require(close(e['measured_y'],y[-1]),'Measured objective mismatch')
        require(e['cost']==sum(target),'Cost mismatch'); cost+=e['cost']
        is_eligible=tuple(target) in allowed_keys
        require(e['recommendation_eligible'] is is_eligible,'Recommendation eligibility mask mismatch')
        eligible.append(is_eligible); means.append(e['population_estimate'])
        require(math.isfinite(means[-1]) and math.isfinite(e['score_se']) and e['score_se']>=0,'Invalid score or SE')
        require(e['score_samples']==(1 if env=='ToyGraph' else 100000),'Wrong scoring precision')
        if env=='ToyGraph':
            natural_x=physical[0] if target[0] else 0.
            z=physical[1] if target[1] else math.exp(-natural_x)
            exact=math.exp(-z/20)-math.cos(z)
            require(close(means[-1],exact) and close(e['measured_y'],exact) and e['score_se']==0,'Toy score differs from exact SCM')
        if e['phase']=='init':
            require(not sequential_started,'Initial data after sequential actions')
            key=tuple(target); j=counts.get(key,0)
            require(e['index']==j,'Initial target row index mismatch'); counts[key]=j+1
            init.setdefault(key,[]).append({k:e[k] for k in ('X','physical_values','network_observation','population_estimate','score_se')})
            init_cost+=e['cost']
        else:
            require(e['phase']=='sequential' and is_eligible,'Sequential action outside menu')
            sequential_started=True
            require(e['index']==len(sequential),'Sequential index mismatch')
            winner=max((j for j in range(idx+1) if eligible[j]),key=lambda j:(events[j]['measured_y'],-j))
            require(e['recommendation_event']==winner,'Recommendation is not eligible measured argmax/first tie')
            require(close(e['recommendation_population_estimate'],means[winner]),'Recommendation score lookup mismatch')
            require(close(e['oracle_best_visited'],max(means[j] for j in range(idx+1) if eligible[j])),'Oracle diagnostic mismatch')
            require(e['usable_node_rows']==[sum(not row['target'][k] for row in events[:idx+1]) for k in range(n)],'Fit data counts mismatch')
            require(e.get('acquisition_diagnostics'),'Missing actual acquisition diagnostics')
            sequential.append(e)
    expected_counts={tuple([0]*n):5,**{tuple(t):2 for t in allowed if any(t)}}
    require(counts==expected_counts,'Initialization counts mismatch')
    require(info['initial_rows']==sum(counts.values()) and info['initial_cost']==init_cost,'Initialization accounting mismatch')
    require(close(info['initial_best_population_estimate'],max(e['population_estimate'] for e in events if e['phase']=='init' and e['recommendation_eligible'])),'Initial score mismatch')
    require(len(sequential)==unit['num_trials'],'Incomplete round count')
    rewards=[e['recommendation_population_estimate'] for e in sequential]
    return dict(endpoint=rewards[-1],equal_round_sum=sum(rewards),total_cost=cost,initial_cost=init_cost,
                initial_rows=sum(counts.values()),rounds=len(rewards),_initial=init,_events=events)

def flatten(value):
    if isinstance(value,list):
        return [x for item in value for x in flatten(item)]
    return [value]

def dynamic_events(payload, rows, info, settings, unit):
    meta=payload['unit']; T=settings['T']; trials=settings['trials']
    for key in ('setup','algo','seed','action_menu'):
        require(meta[key]==unit[key] and info[key]==unit[key], 'Unit identity mismatch: '+key)
    for key in ('T','trials','predictive_samples','feedback_samples'):
        require(meta[key]==settings[key] and info[key]==settings[key], 'Protocol mismatch: '+key)
    require(info['n_obs']==settings['n_obs'] and info['score_samples']==settings['score_samples'], 'Observation/scoring settings mismatch')
    require(meta['mechanism_protocol']=='coherent-union-parent-v1' and info['mechanism_protocol']=='coherent', 'Unexpected estimator')
    require(meta['objective_protocol']==info['objective_protocol']=='population-mc', 'Wrong estimand')
    require(not meta['stock_quirks'] and not info['stock_quirks'] and meta['misspec']==info['misspec']=='', 'Unexpected baseline protocol')
    require(finite(info['secs'],True), 'Invalid wall seconds')
    events=payload['population_events']; commits=payload['policy_commits']; blocks=payload['per_t']
    require(len(events)==T*(trials-1) and len(commits)==len(blocks)==T, 'Incomplete event matrix')
    require(len(rows)==T*trials, 'CSV count mismatch')
    arms=payload['exploration_sets']
    require(all(a and set(a)<= {'X','Z'} and len(set(a))==len(a) for a in arms), 'Invalid action menu')
    require(len({tuple(a) for a in arms})==len(arms), 'Duplicate action menu')
    if unit['algo']=='QDCBO' or unit['action_menu']=='coarse':
        require(arms==[['X','Z']], 'Coarse menu mismatch')
    else:
        require({tuple(a) for a in arms}=={('X',),('Z',),('X','Z')}, 'Fine menu mismatch')
    history={v:[None]*T for v in ('X','Z','Y')}; totalcost=0; training_draws=0; scoring_draws=0
    prefix_assignments=0
    for t,block in enumerate(blocks):
        require(block['t']==t and block['trial_types']==['o']+['i']*(trials-1), 'Unsupported trial schedule')
        require(block['eligible_recommendation']==[False]+[True]*(trials-1), 'Eligibility mismatch')
        require(len(block['best_so_far'])==len(block['per_trial_cost'])==len(block['recommended_values'])==trials, 'Trial vector count mismatch')
        require(len(block['outcome_values'])==trials+1 and len(block['chosen_sets'])==trials-1, 'Outcome offset/count mismatch')
        require(block['per_trial_cost'][0]==0 and block['recommended_values'][0] is None, 'Observation is not eligible')
        require(block['outcome_values'][0]==block['outcome_values'][1]==block['best_so_far'][0]==1e7, 'Unknown sentinel convention')
        keys={'+'.join(a) for a in arms}
        require(set(block['levels_by_set'])==keys and all(len(v)==trials+(t==0) for v in block['levels_by_set'].values()), 'Level table dimensions')
        require(all(v[0] is None for v in block['levels_by_set'].values()), 'Observation contains an intervention')
        if t==0: require(all(v[-1] is None for v in block['levels_by_set'].values()), 'Unused initial storage slot must be empty')
        best=None
        for trial in range(trials):
            row=rows[t*trials+trial]
            label=unit['algo']+('-coarse-menu' if unit['action_menu']=='coarse' else '')+'-coherent-population'
            require(row['method']==label and int(row['seed'])==unit['seed'] and int(row['time_index'])==t and int(row['trial_index'])==trial, 'CSV row identity/order mismatch')
            if trial==0:
                require(row['eligible_recommendation']=='False' and close(row['cost'],0), 'Observation cost/eligibility mismatch')
                require(all(row[k]=='' for k in ('population_recommendation_mean','population_mc_se','training_incumbent','recommendation_event')), 'CSV exposes unearned sentinel')
                continue
            index=t*(trials-1)+trial-1; event=events[index]; arm=event['arm']; x=event['x']
            require(event['event_id']==index and event['t']==t and event['trial']==trial, 'Event order mismatch')
            require(arm in arms and len(x)==len(arm), 'Event arm/coordinate mismatch')
            for variable,value in zip(arm,x):
                lo,hi=(-4,1) if variable=='X' else (-3,3)
                require(finite(value) and lo-1e-9<=value<=hi+1e-9, 'Intervention outside domain')
            expected=deepcopy(history)
            for variable,value in zip(arm,x): expected[variable][t]=value
            require(event['history']==expected, 'History contains uncommitted action, target clamp or wrong prefix')
            require(event['cost']==len(arm)==block['per_trial_cost'][trial], 'Cost mismatch')
            require(block['chosen_sets'][trial-1]==arm, 'Trial/action alignment mismatch')
            active=[k for k,v in block['levels_by_set'].items() if v[trial] is not None]
            require(active==['+'.join(arm)] and flatten(block['levels_by_set'][active[0]][trial])==x, 'Trial/level alignment mismatch')
            require(finite(event['training_mean']) and finite(event['training_mc_se'],True), 'Invalid training response')
            require(close(block['outcome_values'][trial+1],event['training_mean']), 'Trial/outcome alignment mismatch')
            # Earliest executed event breaks exact ties; no scored values select actions.
            if best is None or event['training_mean']<best['training_mean']: best=event
            require(event['recommendation_event']==best['event_id'], 'Recommendation is not measured argmin')
            require(close(block['best_so_far'][trial],best['training_mean']) and close(block['recommended_values'][trial],best['training_mean']), 'Incumbent mismatch/sentinel leakage')
            require(event['training_samples']==settings['feedback_samples'] and event['scoring_samples']==settings['score_samples'], 'Oracle precision mismatch')
            require(finite(event['population_mean']) and finite(event['population_mc_se'],True), 'Invalid postrun score')
            require(close(event['recommendation_population_mean'],best['population_mean']) and close(event['recommendation_population_mc_se'],best['population_mc_se']), 'Recommendation score mismatch')
            require(row['eligible_recommendation']=='True' and int(row['recommendation_event'])==best['event_id'], 'CSV recommendation mismatch')
            for key,value in [('population_recommendation_mean',best['population_mean']),('population_mc_se',best['population_mc_se']),('training_incumbent',best['training_mean']),('cost',event['cost'])]:
                require(close(row[key],value), 'CSV value mismatch: '+key)
            totalcost+=len(arm); training_draws+=event['training_samples']; scoring_draws+=event['scoring_samples']
            prefix_assignments+=sum(value is not None for values in expected.values() for value in values)*event['training_samples']
        for variable,value in zip(best['arm'],best['x']): history[variable][t]=value
        commit=commits[t]
        require(commit['t']==t and commit['recommendation_event']==best['event_id'] and commit['arm']==best['arm'] and commit['x']==best['x'], 'Committed recommendation mismatch')
        require(commit['history']==history and block['optimal_intervention_set']==best['arm'], 'Committed history/arm mismatch')
        for key in ('training_mean','population_mean','population_mc_se'):
            require(close(commit[key],best[key]), 'Commit value mismatch')
        require(close(info['final_best_so_far_per_t'][t],best['population_mean']), 'Info endpoint mismatch')
    require(payload['assigned_blanket']==history, 'Final blanket mismatch')
    return dict(slice_values=[c['population_mean'] for c in commits],
        score_se=[c['population_mc_se'] for c in commits],
        committed_slice_sum=sum(c['population_mean'] for c in commits),
        # Common scoring streams correlate slices: no independence assumption.
        sum_score_se_upper_bound=sum(c['population_mc_se'] for c in commits),
        intervention_coordinate_cost=totalcost, training_draws=training_draws,
        scoring_draws=scoring_draws, training_prefix_coordinate_assignments=prefix_assignments,
        total_runtime_seconds=info['secs'])


def runner():
    # Supports both `python experiments/analyze.py` and namespace imports.
    import importlib.util
    path=Path(__file__).with_name('run.py')
    spec=importlib.util.spec_from_file_location('paper_experiment_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def typed_files(folder):
    result=folder/'result';paths={}
    for kind,pattern in [('csv','*.csv'),('decisions','*.decisions.json'),('info','*_info.json')]:
        matches=list(result.glob(pattern))
        require(len(matches)==1,'Expected exactly one '+kind+' output')
        paths[kind]=matches[0]
    return paths


def static_arms(scm,cond,method):
    joint=('M','X1') if scm=='FrontDoor' else ('X1','X2')
    singles=[(v,) for v in joint]
    if method in ('QCBO','HQCBO','BO'):return {joint}
    if method=='BO-S':return set(singles+[joint])
    if scm=='ParallelParent':return {('X2',)} if cond=='A1' else set(singles+[joint])
    return set(singles)


def parse_unit(folder,unit):
    status=load(folder/'unit.json');suite=unit['suite'];options=unit['options']
    require(status.get('schema')==1 and status.get('unit')==unit,'Unit declaration differs from selected matrix')
    require(status.get('status')=='complete' and status.get('returncode')==0,'Execution incomplete or failed')
    identity=status.get('code_identity','')
    require(len(identity)==64 and all(c in '0123456789abcdef' for c in identity),'Missing source identity')
    expected=runner().output_hashes(folder,suite)
    require(status.get('output_sha256')==expected,'Output hashes do not match unit record')
    record=dict(unit_id=unit['id'],seed=unit['seed'],code_identity=identity)
    if suite=='static':
        result=folder/'result';config=load(result/'config.json');summary=load(result/'summary.json')
        native={k:options['--'+k] for k in ('scm','cond','method')};native['seed']=unit['seed']
        for k,v in native.items():require(config[k]==summary[k]==v,'Static config identity mismatch: '+k)
        for k,v in config.items():require(summary.get(k)==v,'Summary/config mismatch: '+k)
        require(config['protocol_id']=='matched-controlled-noisy-v2' and config['n_obs']==100 and config['n_init']==3,'Unsupported static protocol')
        require(config['feedback_mode']=='single_true_SCM_draw' and config['recommendation_rule']=='minimum_measured_Y;ties_canonical_arm_then_execution_index','Wrong feedback/recommendation protocol')
        require(config['stage']==options['--stage'],'Wrong stage')
        require(config['budget']==(120 if native['scm']=='MediatedChain' else 100),'Wrong budget')
        require(config.get('max_purchases')==(int(options['--max-purchases']) if '--max-purchases' in options else None),'Wrong purchase cap')
        require(summary['status']==('budget_complete' if unit['mode']=='paper' else 'pilot_complete'),'Backend incomplete')
        events=load(result/'events.json');scores=load(result/'scores.json')
        require(digest(result/'events.json')==summary['events_sha256'],'Backend event hash mismatch')
        require(digest(result/'observations.json')==config['observational_sha256'],'Observational hash mismatch')
        obs=load(result/'observations.json');columns=['X1','M','Y'] if native['scm']=='FrontDoor' else ['X1','X2','Y']
        require(obs['columns']==columns and len(obs['rows'])==100 and all(len(row)==3 and all(finite(v) for v in row) for row in obs['rows']),'Invalid observations')
        require({tuple(e['arm']) for e in events if e['phase']=='init'}==static_arms(native['scm'],native['cond'],native['method']),'Wrong initial action menu')
        metrics=static_events(native,config,events,scores,summary,unit['mode']=='paper')
        record.update(native,**metrics,observational_sha256=config['observational_sha256'],wall_seconds=summary['wall_seconds'],
                      _events=events,_initial=[{k:e[k] for k in ('arm','x','measured','noise_seed')} for e in events if e['phase']=='init'])
        return record
    paths=typed_files(folder);info=load(paths['info']);data=load(paths['decisions'])
    with paths['csv'].open(newline='') as f:csv_rows=list(csv.DictReader(f))
    require(Path(info['csv']).name==paths['csv'].name,'CSV/info name mismatch')
    if suite=='mcbo':
        native=dict(env_name=options['--env'],algo=options['--algo'],menu=options['--menu'],seed=unit['seed'],
                    num_trials=int(options['--num-trials']),score_samples=int(options['--score-samples']),misspec=options.get('--misspec',''))
        require(data.get('schema')==2 and data['unit']==info,'Decision/info mismatch')
        for field,key in (('env','env_name'),('algo','algo'),('menu','menu'),('seed','seed'),('num_trials','num_trials'),('score_samples','score_samples'),('misspec','misspec')):
            require(info[field]==native[key],'MCBO configuration mismatch: '+field)
        require(info['protocol']=='mcbo-corrected-v2' and info['beta']==10 and info['noise_scale']==0,'Wrong MCBO protocol')
        metrics=mcbo_events(info,data['events'],native)
        sequential=[e for e in data['events'] if e['phase']=='sequential']
        require(len(csv_rows)==len(sequential),'CSV row count mismatch')
        for i,(row,event) in enumerate(zip(csv_rows,sequential)):
            require(set(row)=={'trial_number','recommendation_population_estimate','oracle_best_visited'},'Wrong CSV columns')
            require(int(row['trial_number'])==i and close(row['recommendation_population_estimate'],event['recommendation_population_estimate']) and close(row['oracle_best_visited'],event['oracle_best_visited']),'CSV score mismatch')
        record.update(native,**metrics,wall_seconds=info['secs'])
        return record
    settings={k:int(options['--'+k.replace('_','-')]) for k in ('T','trials','n_obs','predictive_samples','feedback_samples','score_samples','num_anchor_points')}
    native={k:options['--'+k.replace('_','-')] for k in ('setup','algo','action_menu')};native['seed']=unit['seed']
    require(settings['T']==3 and settings['n_obs']==100 and settings['predictive_samples']==1024 and settings['feedback_samples']==2048 and settings['score_samples']==100000 and settings['num_anchor_points']==100,'Wrong dynamic numerical settings')
    require(data['unit']['engine_sha']==info['engine_sha']==identity,'Recorded dynamic source differs from runner')
    record.update(native,**dynamic_events(data,csv_rows,info,settings,native))
    return record


def paired(rows,left,right,metric,seeds):
    pairs=[]
    for seed in seeds:
        a=next(r for r in rows if r['seed']==seed and left(r));b=next(r for r in rows if r['seed']==seed and right(r))
        pairs.append(dict(seed=seed,quotient=a[metric],fine=b[metric],difference=a[metric]-b[metric]))
    values=[p['difference'] for p in pairs];n=len(values)
    require(n in (20,30),'Unsupported inferential sample size')
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(n)
    critical=2.093024054408263 if n==20 else 2.045229642132703
    return dict(metric=metric,n=n,mean_paired_difference=mean,ci95=[mean-critical*se,mean+critical*se],
                interval=f'pointwise two-sided95% Student t, df{n-1}; no multiplicity correction',seed_values=pairs)


def compare_protected(suite,rows):
    """Exact executed traces, excluding MCBO model-fit diagnostics only."""
    checks=[]
    if suite=='dynamic':return checks
    if suite=='mcbo':
        for row in rows:
            if not row['misspec']:continue
            base=next((r for r in rows if not r['misspec'] and (r['env_name'],r['algo'],r['menu'],r['seed'])==(row['env_name'],row['algo'],row['menu'],row['seed'])),None)
            if base is None:continue
            clean=lambda r:[{k:v for k,v in e.items() if k!='acquisition_diagnostics'} for e in r['_events']]
            checks.append(dict(unit_id=row['unit_id'],required=row['algo']=='QMCBO',exact_equal=clean(row)==clean(base)))
    else:
        for row in rows:
            if row['method']!='QCBO' or row['cond'] not in ('A1','B1'):continue
            baseline='A0' if row['cond']=='A1' else 'B0'
            base=next((r for r in rows if (r['scm'],r['method'],r['cond'],r['seed'])==(row['scm'],'QCBO',baseline,row['seed'])),None)
            if base is not None:checks.append(dict(unit_id=row['unit_id'],required=True,exact_equal=row['_events']==base['_events']))
    return checks


def summarize(results,suite,smoke=False):
    api=runner();expected=api.units(suite,smoke=smoke)
    if not smoke:
        require(len(expected)=={'static':630,'mcbo':200,'dynamic':180}[suite],
                'Paper configuration matrix size changed')
        require({u['seed'] for u in expected}==set(range(2000,2030 if suite=='static' else 2020)),
                'Paper replication seed set changed')
        if suite=='mcbo':
            require(all(u['options']['--num-trials']=='100' and u['options']['--score-samples']=='100000' for u in expected),
                    'Paper MCBO horizon or scorer changed')
        if suite=='dynamic':
            require(all(u['options']['--trials']=='10' for u in expected),'Paper dynamic horizon changed')
        case_fields={'static':('--scm','--cond','--method'),
                     'mcbo':('--env','--algo','--menu','--misspec'),
                     'dynamic':('--setup','--algo','--action-menu')}[suite]
        keys=[tuple(u['options'].get(k,'') for k in case_fields)+(u['seed'],) for u in expected]
        require(len(set(keys))==len(keys),'Duplicate scientific configuration/seed')
        if suite=='static':
            cases={(s,c,m) for s,c in [('ParallelParent','A0'),('ParallelParent','A1'),('FrontDoor','B0'),('FrontDoor','B1'),('MediatedChain','C0')] for m in ('CBO','QCBO')}
            cases.update([('MediatedChain','C0','HQCBO'),('ParallelParent','A1','CBO-NP'),('FrontDoor','B0','CBO-FALLBACK')])
            cases.update((s,c,m) for s,c in [('ParallelParent','A0'),('FrontDoor','B0')] for m in ('CBO-NP','BO-S','BO'))
            cases.update(('MediatedChain','C0',m) for m in ('BO-S','BO'))
        elif suite=='mcbo':
            cases={(s,a,m,'') for s in ('ToyGraph','PSAGraph') for a,m in [('MCBO','full'),('MCBO','coarse'),('QMCBO','coarse')]}
            cases.update((s,a,m,edit) for s,edit in [('ToyGraph','del:0:1'),('PSAGraph','add:2:3')] for a,m in [('MCBO','full'),('QMCBO','coarse')])
        else:
            cases={(s,a,m) for s in ('stat','ind','nonstat_regularized') for a,m in [('DCBO','native'),('DCBO','coarse'),('QDCBO','native')]}
        require({k[:-1] for k in keys}==cases,'Paper scientific configuration matrix changed')
    if smoke:
        # Accept any declared pilot seed without mixing it with paper evidence.
        expected=api.units(suite,seeds=range(1000,1005),smoke=True)
    expected_by_id={u['id']:u for u in expected};mode='smoke' if smoke else 'paper'
    folder=Path(results)/mode/suite;rows=[];failures=[];seen=set()
    for path in sorted(folder.iterdir()) if folder.exists() else []:
        if not path.is_dir():continue
        if path.name not in expected_by_id:
            failures.append(dict(unit_id=path.name,error='Unit outside declared matrix'));continue
        seen.add(path.name)
        try:rows.append(parse_unit(path,expected_by_id[path.name]))
        except (ValueError,KeyError,TypeError,OSError,OverflowError,StopIteration,IndexError) as e:
            failures.append(dict(unit_id=path.name,error=str(e)))
    if len({r['code_identity'] for r in rows})>1:failures.append(dict(unit_id='suite',error='Mixed source implementations'))
    # Verify pairing in the actual data, not just matching seed labels.
    tables={};obs={}
    for r in rows:
        if suite=='dynamic':continue
        key=(r.get('scm',r.get('env_name')),r['seed'])
        if suite=='static':
            if key in obs and obs[key]!=r['observational_sha256']:failures.append(dict(unit_id=r['unit_id'],error='Unpaired observational data'))
            obs[key]=r['observational_sha256'];initial={}
            for event in r['_initial']:initial.setdefault(tuple(event['arm']),[]).append(event)
        else:initial=r['_initial']
        for arm,values in initial.items():
            k=key+(arm,)
            if k in tables and tables[k]!=values:failures.append(dict(unit_id=r['unit_id'],error='Unpaired shared-arm initialization'))
            tables[k]=values
    protected=compare_protected(suite,rows)
    failures.extend(dict(unit_id=p['unit_id'],error='Protected executed trace differs') for p in protected if p['required'] and not p['exact_equal'])
    missing=sorted(set(expected_by_id)-seen)
    complete=not smoke and not missing and not failures and len(rows)==len(expected)
    output=dict(schema=1,suite=suite,mode=mode,expected_units=len(expected),valid_units=len(rows),complete=complete,
                inference_allowed=complete,status='complete' if complete else 'descriptive_only',
                missing_units=missing,failures=failures,protected_checks=protected,
                analysis_source_sha256=digest(__file__),units=[{k:v for k,v in r.items() if not k.startswith('_')} for r in rows],paired_effects=[])
    output['failure_count']=len(failures)
    output['missing_count']=len(missing)
    group_fields={'static':('scm','cond','method'),
                  'mcbo':('env_name','algo','menu','misspec'),
                  'dynamic':('setup','algo','action_menu')}[suite]
    metric_fields={'static':('final_recommendation_regret','cost_integrated_recommendation_regret'),
                   'mcbo':('endpoint','equal_round_sum'),
                   'dynamic':('slice0','slice1','slice2','committed_slice_sum')}[suite]
    output['descriptive_summaries']=[]
    for key in sorted({tuple(r[k] for k in group_fields) for r in rows}):
        group=[r for r in rows if tuple(r[k] for k in group_fields)==key]
        description=dict(zip(group_fields,key));description.update(n=len(group),metrics={})
        for metric in metric_fields:
            values=[r['slice_values'][int(metric[-1])] if metric.startswith('slice') else r[metric] for r in group]
            stats={'mean':statistics.mean(values)}
            if complete:
                critical=2.045229642132703 if len(values)==30 else 2.093024054408263
                half=critical*statistics.stdev(values)/math.sqrt(len(values))
                stats['ci95']=[stats['mean']-half,stats['mean']+half]
            description['metrics'][metric]=stats
        output['descriptive_summaries'].append(description)
    if not complete:return output
    if suite=='static':
        for scm,cond in sorted({(r['scm'],r['cond']) for r in rows if r['method']=='QCBO'}):
            group=[r for r in rows if (r['scm'],r['cond'])==(scm,cond)]
            output['paired_effects'].append(dict(scm=scm,cond=cond,contrast='QCBO minus CBO; minimization',metrics=[paired(group,lambda r:r['method']=='QCBO',lambda r:r['method']=='CBO',metric,list(range(2000,2030))) for metric in ('final_recommendation_regret','cost_integrated_recommendation_regret')]))
    elif suite=='mcbo':
        for env in ('ToyGraph','PSAGraph'):
            group=[r for r in rows if r['env_name']==env and not r['misspec']]
            for menu in ('coarse','full'):
                output['paired_effects'].append(dict(env=env,role='primary matched menu' if menu=='coarse' else 'secondary full menu',contrast='QMCBO minus MCBO; maximization',metrics=[paired(group,lambda r:r['algo']=='QMCBO',lambda r:r['algo']=='MCBO' and r['menu']==menu,metric,list(range(2000,2020))) for metric in ('endpoint','equal_round_sum')]))
        output['cost_interpretation']='Equal sequential rounds, not equal total costs.'
    else:
        for setup in ('stat','ind','nonstat_regularized'):
            group=[dict(r,**{'slice'+str(i):v for i,v in enumerate(r['slice_values'])}) for r in rows if r['setup']==setup]
            output['paired_effects'].append(dict(setup=setup,contrast='quotient/coarse minus fine/coarse; minimization',metrics=[paired(group,lambda r:r['algo']=='QDCBO',lambda r:r['algo']=='DCBO' and r['action_menu']=='coarse',metric,list(range(2000,2020))) for metric in ('slice0','slice1','slice2','committed_slice_sum')]))
        output['interpretation']='Committed-prefix population outcomes; no global-optimum regret or equal-cost claim. Sum scoring SE is an upper bound; slice errors are correlated.'
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',choices=['static','mcbo','dynamic'],required=True)
    p.add_argument('--results',type=Path,default=Path('results/paper'));p.add_argument('--smoke',action='store_true')
    p.add_argument('--out',type=Path,required=True,help='Summary JSON; sibling CSV contains seed-level outcomes')
    a=p.parse_args();summary=summarize(a.results,a.suite,a.smoke)
    if a.out.suffix!='.json':p.error('--out must name a .json file (the companion .csv is generated automatically)')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    rows=summary['units']
    if rows:
        fields=sorted({k for r in rows for k in r})
        with a.out.with_suffix('.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    print(json.dumps({k:summary[k] for k in ('status','expected_units','valid_units','inference_allowed')}))
    if summary['failures']:raise SystemExit(2)

if __name__=='__main__':main()
