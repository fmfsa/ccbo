#!/usr/bin/env python3
"""Frozen MCBO analysis: complete matrices, measured recommendations, paired seeds.

Primary contrast is QMCBO/coarse minus MCBO/coarse. Full-menu comparison is
separate. Endpoints and sums are indexed by equal sequential rounds, never
claimed to compare equal intervention costs. Engineering preflights yield no
performance inference. Protected edits compare full decision traces to main.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ENVS=("ToyGraph","PSAGraph")
SEEDS=list(range(2000,2020))
METHODS=(("MCBO","full"),("MCBO","coarse"),("QMCBO","coarse"))
EDITS={"ToyGraph":"del:0:1","PSAGraph":"add:2:3"}
T975_DF19=2.093024054408263
CORE_PROTOCOL="mcbo-corrected-v2"
IMPORT_PATHS={'mcbo.mcbo_trial':'third_party/mcbo/mcbo/mcbo_trial.py',
 'mcbo.models.gp_network':'third_party/mcbo/mcbo/models/gp_network.py',
 'mcbo.acquisition_function_optimization.optimize_acqf':'third_party/mcbo/mcbo/acquisition_function_optimization/optimize_acqf.py',
 'functions':'third_party/mcbo/scripts/functions.py'}


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_id(m):
    return hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load(path):
    def reject(x): raise ValueError('Nonfinite JSON constant: '+x)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def close(a,b): return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)


def stem(unit):
    label=unit['algo']+('-coarse-menu' if unit['algo']=='MCBO' and unit['menu']=='coarse' else '')
    return f"result/trial_results_{label}_{unit['env_name']}_{unit['seed']}"


def targets(env,menu):
    if env=='ToyGraph':
        return [[1,0,0],[0,1,0],[1,1,0],[0,0,0]] if menu=='full' else [[1,1,0],[0,0,0]]
    return [[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,1,1,0,0]] if menu=='full' else [[0,0,1,1,0,0]]


def validate_manifest(m):
    require(m.get('schema')==1 and m.get('source_sha256'),'Missing frozen schema/source hashes')
    require(m['source_sha256'].get('scripts/summarize_mcbo_campaign.py')==digest(__file__),'Analysis source differs from frozen manifest')
    for rel,sha in m['source_sha256'].items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Unsafe source path')
        require(isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha),'Invalid source SHA256')
    require(all(path in m['source_sha256'] for path in IMPORT_PATHS.values()),'Missing upstream source hashes')
    p=m['protocol']; stage=p['stage']
    require(p.get('family')=='mcbo-corrected' and p.get('adapter_protocol')==CORE_PROTOCOL,'Unsupported protocol')
    require(stage in ('engineering-preflight','replication','protected-extension'),'Unknown stage')
    require(p['score_samples']==100000 and p['beta']==10. and p['noise_scale']==0.,'Settings differ from frozen analysis')
    require(p['replication_seeds']==SEEDS,'Wrong replication seed matrix')
    require(p['initial_obs_samples']==5 and p['initial_int_samples']==2,'Wrong initialization protocol')
    expected_rounds=20 if stage=='engineering-preflight' else 100
    require(p['iterations']==expected_rounds,'Wrong round horizon')
    seeds=[1001] if stage=='engineering-preflight' else SEEDS
    methods=METHODS if stage!='protected-extension' else (METHODS[0],METHODS[2])
    expected={(e,a,menu,s) for e in ENVS for a,menu in methods for s in seeds}
    units=m['units']; actual=[(u['env_name'],u['algo'],u['menu'],u['seed']) for u in units]
    require(len(set(actual))==len(actual) and set(actual)==expected,'Incomplete or wrong expected unit matrix')
    require(len({u['id'] for u in units})==len(units),'Duplicate unit IDs')
    for u in units:
        require(u['id'] and all(c.isalnum() or c in '_-' for c in u['id']),'Unsafe unit ID')
        require(u['num_trials']==expected_rounds and u['score_samples']==100000,'Wrong unit settings')
        require(u['misspec']==(EDITS[u['env_name']] if stage=='protected-extension' else ''),'Wrong perturbation')
        argv=u['argv']; require(argv[:2]==['{python}','scripts/run_mcbo_campaign_unit.py'],'Unexpected command')
        expected=['--env',u['env_name'],'--algo',u['algo'],'--menu',u['menu'],'--seed',str(u['seed']),
                  '--num-trials',str(expected_rounds),'--score-samples','100000','--outdir','{unit_dir}/result']
        if u['misspec']: expected+=['--misspec',u['misspec']]
        require(argv[2:]==expected,'Command/config mismatch')
        require(set(u['outputs'])=={stem(u)+suffix for suffix in ('.csv','.decisions.json','_info.json')}|{'result/runtime.json'},'Wrong output schema')
    if stage=='protected-extension': require(p.get('reference_manifest_id'),'Missing protected reference')


def validate_events(info,events,unit):
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


def validate_unit(m,u,folder):
    status=load(folder/'execution.json')
    require(status.get('status')=='complete' and status.get('returncode')==0,'Execution incomplete/failed')
    require(status.get('manifest_id')==manifest_id(m) and status.get('unit')==u and status.get('protocol')==m['protocol'],'Execution identity mismatch')
    require(load(folder/'frozen-manifest.json')==m,'Frozen manifest mismatch')
    require(set(status.get('output_sha256',{}))==set(u['outputs']),'Output hash set mismatch')
    for rel in u['outputs']: require(digest(folder/rel)==status['output_sha256'][rel],'Output hash mismatch: '+rel)
    prefix=folder/stem(u); info=load(str(prefix)+'_info.json'); data=load(str(prefix)+'.decisions.json')
    runtime=load(folder/'result/runtime.json')
    require({k:runtime[k] for k in ('python','packages')}==m['protocol']['expected_runtime'],'Runtime version mismatch')
    require(runtime['vendor_git_revision']==m['protocol']['vendor_revision'],'Vendor revision mismatch')
    require(set(runtime.get('imports',{}))=={'mcbo.mcbo_trial','mcbo.models.gp_network',
        'mcbo.acquisition_function_optimization.optimize_acqf','functions'},'Missing actual imported-module provenance')
    for name,record in runtime['imports'].items():
        require(record['relative_source']==IMPORT_PATHS[name],'Imported module/source mismatch')
        require(m['source_sha256'].get(record['relative_source'])==record['sha256'],'Imported source hash mismatch')
    require(data.get('schema')==2 and data['unit']==info,'Decision/info identity mismatch')
    for field,key in (('env','env_name'),('algo','algo'),('menu','menu'),('seed','seed'),('num_trials','num_trials'),('misspec','misspec'),('score_samples','score_samples')):
        require(info[field]==u[key],'Settings mismatch: '+field)
    require(info['protocol']==CORE_PROTOCOL and info['beta']==10 and info['noise_scale']==0,'Adapter/settings mismatch')
    require(Path(info['csv']).name==Path(str(prefix)+'.csv').name,'CSV identity mismatch')
    require(math.isfinite(info['secs']) and info['secs']>=0,'Invalid runtime')
    row=validate_events(info,data['events'],u)
    with open(str(prefix)+'.csv',newline='') as f:
        reader=csv.DictReader(f); records=list(reader)
        require(reader.fieldnames==['trial_number','recommendation_population_estimate','oracle_best_visited'],'CSV schema mismatch')
    sequential=[e for e in data['events'] if e['phase']=='sequential']
    require(len(records)==len(sequential),'CSV round count mismatch')
    for i,(record,event) in enumerate(zip(records,sequential)):
        require(int(record['trial_number'])==i and close(float(record['recommendation_population_estimate']),event['recommendation_population_estimate'])
                and close(float(record['oracle_best_visited']),event['oracle_best_visited']),'CSV/scorer identity mismatch')
    return dict(unit_id=u['id'],env=u['env_name'],algo=u['algo'],menu=u['menu'],seed=u['seed'],misspec=u['misspec'],
                wall_seconds=info['secs'],**row)


def collect(m,results):
    validate_manifest(m); campaign=Path(results)/manifest_id(m); rows=[]; failures=[]
    if campaign.exists():
        for path in campaign.iterdir():
            if path.is_dir() and path.name not in {u['id'] for u in m['units']}:
                failures.append(dict(unit_id=path.name,status='unexpected',error='Unit outside matrix'))
    for unit in m['units']:
        folder=campaign/unit['id']
        try: rows.append(validate_unit(m,unit,folder))
        except (ValueError,KeyError,TypeError,OSError,OverflowError) as exc:
            state='missing' if not (folder/'execution.json').exists() else 'invalid'
            try:
                actual=load(folder/'execution.json').get('status')
                if actual in ('failed','running'): state=actual
            except (ValueError,OSError): pass
            failures.append(dict(unit_id=unit['id'],status=state,error=str(exc)))
    # Target-keyed initialization is identical across menus and algorithms.
    reference={}
    for row in rows:
        for target,initial in row['_initial'].items():
            key=(row['env'],row['seed'],target)
            if key in reference and reference[key]!=initial:
                failures.append(dict(unit_id=row['unit_id'],status='invalid',error='Common-target initial data mismatch'))
            else: reference[key]=initial
    return rows,failures


def interval(values,metric):
    require(len(values)==20 and [v['seed'] for v in values]==SEEDS,'Paired inference requires all20 seeds')
    diffs=[v['quotient']-v['fine'] for v in values]; mean=statistics.mean(diffs)
    half=T975_DF19*statistics.stdev(diffs)/math.sqrt(20)
    return dict(metric=metric,n=20,contrast='QMCBO minus MCBO',mean_paired_difference=mean,ci95=[mean-half,mean+half],
                interval='pointwise two-sided95% Student t, df19; no multiplicity adjustment',
                seed_values=[dict(v,difference=d) for v,d in zip(values,diffs)])


def public(row): return {k:v for k,v in row.items() if not k.startswith('_')}


def summarize(m,results,reference_manifest=None,reference_results=None):
    rows,failures=collect(m,results); stage=m['protocol']['stage']
    output=dict(schema=1,manifest_id=manifest_id(m),stage=stage,analysis_source_sha256=digest(__file__),
                frozen_source_sha256=m['source_sha256'],settings=m['protocol'],expected_units=len(m['units']),valid_units=len(rows),
                primary='population estimate of eligible measured-best recommendation at final sequential round',
                secondary='sum of recommendation population estimates over equal sequential rounds',
                cost_interpretation='Native initialization/menu costs differ; no equal-cost claim',comparisons=[],failures=failures)
    output['engineering_audit']=[{k:v for k,v in public(row).items() if k not in ('endpoint','equal_round_sum')} for row in rows]
    if stage=='protected-extension' and not failures:
        if reference_manifest is None or reference_results is None:
            failures.append(dict(unit_id='reference',status='missing',error='Main campaign reference required for protected comparisons'))
        else:
            require(manifest_id(reference_manifest)==m['protocol']['reference_manifest_id'],'Wrong reference manifest')
            require(reference_manifest['protocol']['stage']=='replication' and reference_manifest['source_sha256']==m['source_sha256'],'Reference source/stage mismatch')
            reference,errors=collect(reference_manifest,reference_results)
            failures.extend(dict(f,unit_id='reference:'+f['unit_id']) for f in errors)
            if not failures:
                checks=[]
                for row in rows:
                    base=next(r for r in reference if (r['env'],r['algo'],r['menu'],r['seed'])==(row['env'],row['algo'],row['menu'],row['seed']))
                    # Ignore fit diagnostics (models may differ under an edit),
                    # compare full executed actions, data, scores, costs and choices.
                    def trace(r): return [{k:v for k,v in e.items() if k!='acquisition_diagnostics'} for e in r['_events']]
                    equal=trace(row)==trace(base)
                    checks.append(dict(unit_id=row['unit_id'],expected_invariant=row['algo']=='QMCBO',full_trace_equal=equal))
                    if row['algo']=='QMCBO' and not equal:
                        failures.append(dict(unit_id=row['unit_id'],status='invalid',error='Protected quotient trajectory changed'))
                output['protected_trace_checks']=checks
    output['failure_count']=len(failures)
    output['failure_counts_by_status']={s:sum(f['status']==s for f in failures) for s in sorted({f['status'] for f in failures})}
    output['inference_status']='blocked' if failures else 'engineering_only' if stage=='engineering-preflight' else 'validated_protected_traces' if stage=='protected-extension' else 'complete'
    if failures or stage!='replication': return output
    output['seed_level_metrics']=[public(r) for r in rows]
    for env in ENVS:
        for menu,label in (('coarse','primary matched-action-menu comparison'),('full','secondary full-menu comparison')):
            metrics=[]
            for metric in ('endpoint','equal_round_sum'):
                values=[]
                for seed in SEEDS:
                    find=lambda algo,which:next(r[metric] for r in rows if (r['env'],r['algo'],r['menu'],r['seed'])==(env,algo,which,seed))
                    values.append(dict(seed=seed,fine=find('MCBO',menu),quotient=find('QMCBO','coarse')))
                metrics.append(interval(values,metric))
            output['comparisons'].append(dict(env=env,fine_menu=menu,quotient_menu='coarse',role=label,metrics=metrics))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-manifest',type=Path);p.add_argument('--reference-results',type=Path)
    args=p.parse_args(); result=summarize(load(args.manifest),args.results,
        load(args.reference_manifest) if args.reference_manifest else None,args.reference_results)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    if result['inference_status']=='blocked':raise SystemExit(2)

if __name__=='__main__': main()
