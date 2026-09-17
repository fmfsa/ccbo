"""Strict frozen dynamic ledger audit; inference only for complete 20-seed finals."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SETUPS = ('stat', 'ind', 'nonstat_regularized')
METHODS = (('DCBO','native'), ('DCBO','coarse'), ('QDCBO','native'))
SEEDS = list(range(2000, 2020))
T975_DF19 = 2.093024054408263


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(m):
    return hashlib.sha256(json.dumps(m, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(a, b):
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b), rel_tol=1e-10, abs_tol=1e-9)
    except (ValueError, TypeError):
        return False


def finite(value, nonnegative=False):
    return isinstance(value, (int,float)) and math.isfinite(value) and (not nonnegative or value >= 0)


def relative(folder, path):
    require(not Path(path).is_absolute() and '..' not in Path(path).parts, 'Unsafe relative path')
    return folder/path


def options(unit):
    args=unit['argv']; result={}
    for i,value in enumerate(args):
        if value.startswith('--'):
            require(value not in result and i+1<len(args), 'Duplicate/missing CLI argument')
            result[value]=args[i+1]
    return result


def flatten(value):
    if isinstance(value,list):
        return [x for item in value for x in flatten(item)]
    return [value]


def validate_payload(payload, rows, info, settings, unit, campaign):
    meta=payload['unit']; T=settings['T']; trials=settings['trials']
    for key in ('setup','algo','seed','action_menu'):
        require(meta[key]==unit[key] and info[key]==unit[key], 'Unit identity mismatch: '+key)
    for key in ('T','trials','predictive_samples','feedback_samples'):
        require(meta[key]==settings[key] and info[key]==settings[key], 'Protocol mismatch: '+key)
    require(info['n_obs']==settings['n_obs'] and info['score_samples']==settings['score_samples'], 'Observation/scoring settings mismatch')
    require(meta['engine_sha']==info['engine_sha']=='manifest-sha256:'+campaign, 'Engine manifest identity mismatch')
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


def validate_unit(m, unit, folder):
    campaign=identity(m); status=load(folder/'execution.json')
    require(status.get('status')=='complete' and status.get('returncode')==0, 'Unit incomplete or failed')
    require(status['manifest_id']==campaign and status['unit']==unit and status['protocol']==m['protocol'], 'Execution manifest/unit identity mismatch')
    require(load(folder/'frozen-manifest.json')==m, 'Saved manifest differs')
    actual_argv=status['argv']; out_index=actual_argv.index('--outdir')+1
    actual_outdir=actual_argv[out_index]
    require(actual_outdir.endswith('/'+campaign+'/'+unit['id']+'/result'), 'Executor output destination mismatch')
    remote_folder=actual_outdir.removesuffix('/result')
    executable=m['protocol']['recorded_runtime']['executable']
    expected_argv=[x.replace('{python}',executable).replace('{unit_dir}',remote_folder) for x in unit['argv']]
    require(actual_argv==expected_argv, 'Actual command differs from frozen command')
    require(finite(status['started_unix']) and finite(status['finished_unix']) and status['finished_unix']>=status['started_unix'], 'Invalid executor timing')
    require(set(status['output_sha256'])==set(unit['outputs']) and len(unit['outputs'])==(4 if 'runtime_expected' in m['protocol'] else 3), 'Output set mismatch')
    for rel,sha in status['output_sha256'].items(): require(digest(relative(folder,rel))==sha, 'Output hash mismatch: '+rel)
    args=options(unit); settings=m['protocol']
    for key in ('setup','algo','seed','action_menu'):
        require(args['--'+key.replace('_','-')]==str(unit[key]), 'CLI identity mismatch')
    for key in ('T','trials','n_obs','num_anchor_points','predictive_samples','feedback_samples','score_samples'):
        require(int(args['--'+key.replace('_','-')])==settings[key], 'CLI protocol mismatch: '+key)
    require(args['--mechanism-protocol']=='coherent' and args['--objective-protocol']=='population-mc', 'Wrong CLI protocol')
    require(args['--outdir']=='{unit_dir}/result', 'Output location mismatch')
    paths={suffix:next((relative(folder,p) for p in unit['outputs'] if p.endswith(suffix)),None) for suffix in ('.csv','.decisions.json','_info.json')}
    require(all(paths.values()), 'Missing typed output')
    payload=load(paths['.decisions.json']); info=load(paths['_info.json'])
    require(info['csv']==actual_outdir+'/'+paths['.csv'].name, 'Recorded CSV destination differs')
    with paths['.csv'].open() as handle: rows=list(csv.DictReader(handle))
    result=validate_payload(payload,rows,info,settings,unit,campaign)
    result.update(unit_id=unit['id'],setup=unit['setup'],algo=unit['algo'],action_menu=unit['action_menu'],seed=unit['seed'],
        executor_seconds=status['finished_unix']-status['started_unix'],
        provenance={'source_identity':'frozen manifest + engine_sha + output hashes',
        'runtime_identity':'manifest records environment; no per-execution package attestation in pilot schema',
        'scoring_order':'source-attested model.run then score_events; no serialized phase timestamps in pilot schema'})
    if 'runtime_expected' in m['protocol']:
        runtime=load(folder/'result/runtime.json')
        expected=m['protocol']['runtime_expected']
        require(runtime['python']==expected['python'] and runtime['packages']==expected['packages'], 'Actual runtime version mismatch')
        require(runtime['imports'] and set(runtime['imports'])==set(m['protocol']['runtime_imports']), 'Actual import set mismatch')
        for name,relative_source in m['protocol']['runtime_imports'].items():
            record=runtime['imports'][name]
            require(record['relative_source']==relative_source and record['sha256']==m['source_sha256'][relative_source], 'Actual imported source identity mismatch')
            require(record['path'].endswith(relative_source.removeprefix('third_party/DCBO/')) if relative_source.startswith('third_party/DCBO/') else record['path'].endswith(relative_source), 'Actual import path mismatch')
        result['provenance']['runtime_identity']='Per-execution pinned package versions and imported source identities verified by wrapper'
    # Preserve learner-only information for optional exact identity comparisons.
    result['_learner']={'per_t':payload['per_t'],'events':[{k:v for k,v in e.items() if k not in ('population_mean','population_mc_se','scoring_samples','recommendation_population_mean','recommendation_population_mc_se')} for e in payload['population_events']],
                        'policy':[{k:v for k,v in c.items() if k not in ('population_mean','population_mc_se')} for c in payload['policy_commits']]}
    return result


def learner_trace(payload):
    return {'per_t':payload['per_t'],
        'events':[{k:v for k,v in e.items() if k not in ('population_mean','population_mc_se','scoring_samples','recommendation_population_mean','recommendation_population_mc_se')} for e in payload['population_events']],
        'policy':[{k:v for k,v in c.items() if k not in ('population_mean','population_mc_se')} for c in payload['policy_commits']]}


def validate_identity_pair(kind, left, right):
    require(kind in ('singleton','protected'), 'Unknown paired identity type')
    for key in ('setup','seed','T','trials','predictive_samples','feedback_samples','objective_protocol'):
        require(left['unit'][key]==right['unit'][key], 'Identity pair has unmatched settings')
    require(left['exploration_sets']==right['exploration_sets'], 'Identity pair menus differ')
    require(learner_trace(left)==learner_trace(right), 'Identity pair learner trace differs')
    return dict(kind=kind,status='passed',setup=left['unit']['setup'],seed=left['unit']['seed'])


def paired_summary(rows, setup):
    selected=[r for r in rows if r['setup']==setup]
    index={(r['algo'],r['action_menu'],r['seed']):r for r in selected}
    require(len(selected)==60 and {r['seed'] for r in selected}==set(SEEDS), 'Incomplete final paired matrix')
    pairs=[]
    for seed in SEEDS:
        q=index['QDCBO','native',seed]; f=index['DCBO','coarse',seed]
        differences=[a-b for a,b in zip(q['slice_values'],f['slice_values'])]
        pairs.append(dict(seed=seed,slice_difference=differences,sum_difference=q['committed_slice_sum']-f['committed_slice_sum'],
            quotient_score_se=q['score_se'],fine_coarse_score_se=f['score_se'],
            sum_difference_score_se_upper_bound=q['sum_score_se_upper_bound']+f['sum_score_se_upper_bound']))
    metrics=[]
    for i,name in enumerate(('slice0','slice1','slice2','committed_slice_sum')):
        values=[p['slice_difference'][i] if i<3 else p['sum_difference'] for p in pairs]
        mean=statistics.mean(values); se=statistics.stdev(values)/math.sqrt(20)
        metrics.append(dict(metric=name,mean=mean,between_seed_se=se,ci95=[mean-T975_DF19*se,mean+T975_DF19*se]))
    return dict(setup=setup,contrast='quotient/coarse minus fine/coarse; minimization',n=20,
        interval='pointwise two-sided Student t, df19; no multiplicity correction',
        metrics=metrics,paired_seed_values=pairs,
        limitation='Monte Carlo scoring error is reported separately; intervals describe finite-MC scored responses. No known global optimum and no regret is asserted.')


def validate_precision_gate(protocol, source_root, source_hashes):
    gate=protocol.get('precision_gate',{})
    require(gate.get('accepted') is True and gate.get('chosen_particles')==protocol['predictive_samples'], 'Precision acceptance/count mismatch')
    limits={'mean_normalized_rms':.05,'mean_normalized_max':.15,'variance_relative_rms':.10,'variance_relative_max':.30}
    expected={(setup,resolution) for setup in SETUPS for resolution in ('fine','quotient')}
    seen=set()
    for entry in gate.get('reports',[]):
        key=(entry['setup'],entry['resolution']);require(key in expected and key not in seen,'Precision report matrix mismatch');seen.add(key)
        path=relative(source_root,entry['path'])
        require(source_hashes.get(entry['path'])==entry['sha256']==digest(path),'Precision evidence hash mismatch')
        report=load(path)
        require((report['setup'],report['resolution'])==key and report['low']==gate['chosen_particles'] and report['high']==gate['reference_particles']>report['low'],'Precision report identity mismatch')
        require(report['observation_seed']==1001 and report['n_obs']==100 and report['integration_seeds']==[31337,27183,16180,14142], 'Precision fixed seeds/data changed')
        require(report['thresholds']==limits and report['passed'] is True,'Precision tolerance failed or changed')
        mean=[];variance=[];queries=[]
        for row in report['rows']:
            require(all(finite(row[k]) for k in ('low_mean','high_mean','low_variance','high_variance')) and row['low_variance']>=0 and row['high_variance']>=0,'Invalid precision moments')
            mean.append((row['low_mean']-row['high_mean'])/max(1.,math.sqrt(row['high_variance'])))
            variance.append((row['low_variance']-row['high_variance'])/max(1.,row['high_variance']))
            require(close(mean[-1],row['mean_error']) and close(variance[-1],row['variance_error']),'Precision normalized error mismatch')
            queries.append((row['history'],None if row['candidate'] is None else tuple(row['candidate']),row['seed']))
        grid={(history,candidate,seed) for history in ('natural','joint-fixed') for candidate in (None,(-2.,-1.),(0.,0.),(1.,2.)) for seed in (31337,27183,16180,14142)}
        require(len(queries)==32 and set(queries)==grid,'Precision fixed query grid changed')
        computed=dict(mean_normalized_rms=math.sqrt(statistics.mean(x*x for x in mean)),mean_normalized_max=max(abs(x) for x in mean),variance_relative_rms=math.sqrt(statistics.mean(x*x for x in variance)),variance_relative_max=max(abs(x) for x in variance))
        require(all(close(report['metrics'][k],v) and v<=limits[k] for k,v in computed.items()),'Precision recomputation failed tolerance')
    require(seen==expected,'All six precision reports required')


def summarize(m, results, source_root, vendor_root=None):
    require(m.get('schema')==1 and m.get('source_sha256'), 'Need frozen source manifest')
    require(m['protocol']['family']=='dynamic-coherent-population-v1', 'Unknown family')
    stage=m['protocol']['stage']; require(stage in ('resource-pilot','replication'), 'Unknown stage')
    final=stage=='replication'; seeds=SEEDS if final else [1001]
    expected={(s,a,menu,seed) for s in SETUPS for a,menu in METHODS for seed in seeds}
    keys=[(u['setup'],u['algo'],u['action_menu'],u['seed']) for u in m['units']]
    require(len(keys)==len(set(keys)) and set(keys)==expected and len({u['id'] for u in m['units']})==len(keys), 'Incomplete or unexpected declared matrix')
    require(m['protocol']['recorded_runtime'], 'Missing recorded runtime identity')
    for rel,sha in m['source_sha256'].items():
        path=relative(vendor_root,rel.removeprefix('third_party/DCBO/')) if vendor_root is not None and rel.startswith('third_party/DCBO/') else relative(source_root,rel)
        require(digest(path)==sha, 'Frozen source mismatch: '+rel)
    required={'ccbo/qdcbo/runner.py','ccbo/qdcbo/population.py','ccbo/qdcbo/coherent.py','scripts/run_frozen_unit.py'}
    require(required<=set(m['source_sha256']) and any(k.startswith('third_party/DCBO/dcbo/') for k in m['source_sha256']), 'Missing scientific/vendor provenance')
    p=m['protocol']
    for key,value in dict(T=3,trials=10,n_obs=100,num_anchor_points=100,feedback_samples=2048,score_samples=100000).items():
        require(p[key]==value,'Unexpected final/pilot fixed setting: '+key)
    require(isinstance(p['predictive_samples'],int) and p['predictive_samples']>=2, 'Invalid integration precision')
    if final:
        require('runtime_expected' in p and 'runtime_imports' in p, 'Final lacks per-execution runtime gate')
        require(p.get('replication_seeds')==SEEDS and p.get('expected_units')==180,'Final seed/unit declaration mismatch')
    if final or p.get('precision_gate',{}).get('accepted'):
        validate_precision_gate(p,source_root,m['source_sha256'])
    rows=[]; unavailable=[]; campaign=identity(m)
    for unit in m['units']:
        folder=results/campaign/unit['id']
        if not (folder/'execution.json').exists():
            unavailable.append(dict(unit=unit['id'],reason='missing')); continue
        status=load(folder/'execution.json')
        if status.get('status')!='complete':
            unavailable.append(dict(unit=unit['id'],reason=status.get('status','unknown'))); continue
        rows.append(validate_unit(m,unit,folder))
    complete=not unavailable
    # Optional identity companions are compared only when explicitly present.
    # Core180 does not claim such companions exist; they passed engineering gates.
    identity_checks=[]
    companions=m.get('identity_companions',[])
    row_index={r['unit_id']:r for r in rows}
    for pair in companions:
        require(pair['kind'] in ('singleton','protected'), 'Unknown identity control')
        if all(uid in row_index for uid in (pair['left'],pair['right'])):
            require(row_index[pair['left']]['_learner']==row_index[pair['right']]['_learner'], 'Paired learner identity failed')
            identity_checks.append(dict(pair,status='passed'))
    inference=final and complete
    effects=[paired_summary(rows,setup) for setup in SETUPS] if inference else []
    descriptions=[]
    if inference:
        for setup in SETUPS:
            for algo,menu in METHODS:
                group=[r for r in rows if (r['setup'],r['algo'],r['action_menu'])==(setup,algo,menu)]
                descriptions.append(dict(setup=setup,algo=algo,action_menu=menu,n=20,
                    mean_slice_values=[statistics.mean(r['slice_values'][t] for r in group) for t in range(3)],
                    mean_committed_slice_sum=statistics.mean(r['committed_slice_sum'] for r in group),
                    mean_coordinate_cost=statistics.mean(r['intervention_coordinate_cost'] for r in group),
                    mean_runtime_seconds=statistics.mean(r['total_runtime_seconds'] for r in group)))
    for r in rows: r.pop('_learner',None)
    return dict(schema=1,manifest_id=campaign,stage=stage,complete=complete,
        expected_units=len(expected),validated_units=len(rows),unavailable=unavailable,
        inference_allowed=inference,paired_effects=effects,method_descriptive_summaries=descriptions,units=rows,identity_checks=identity_checks,
        fine_full='Reported separately in per-seed units; not mixed with the matched-menu contrast.',
        interpretation=('Complete prespecified paired comparison; no invented global regret.' if inference else 'Pilot/incomplete campaigns provide validation and resources only; no performance inference.'),
        scoring_phase_evidence='Frozen-source postrun order and independent scoring-precision firewall tests; no event phase timestamps.',
        runtime_limitation=('Per-unit runtime/import attestation verified.' if 'runtime_expected' in p else 'Pilot environment was recorded globally; no per-execution package attestation.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vendor-root',type=Path,help='Optional independently fetched DCBO root; all declared vendor hashes still verified')
    p.add_argument('--identity-pair',nargs=3,action='append',default=[],metavar=('KIND','LEFT','RIGHT'))
    a=p.parse_args(); result=summarize(load(a.manifest),a.results,a.source_root,a.vendor_root)
    for kind,left,right in a.identity_pair:
        checked=validate_identity_pair(kind,load(left),load(right))
        checked.update(left_sha256=digest(left),right_sha256=digest(right))
        result['identity_checks'].append(checked)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','validated_units','expected_units','inference_allowed')}))
