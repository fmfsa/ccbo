import csv
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[2]/'scripts/summarize_dynamic_campaign.py'
spec=importlib.util.spec_from_file_location('dynamic_summary',PATH)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture():
    settings=dict(T=3,trials=3,n_obs=100,predictive_samples=256,feedback_samples=2048,score_samples=100000)
    unit=dict(setup='stat',algo='QDCBO',action_menu='native',seed=1001)
    meta=dict(unit,T=3,trials=3,predictive_samples=256,feedback_samples=2048,
        mechanism_protocol='coherent-union-parent-v1',objective_protocol='population-mc',
        stock_quirks=False,misspec='',engine_sha='manifest-sha256:test')
    info=dict(meta,n_obs=100,score_samples=100000,mechanism_protocol='coherent',secs=1.,final_best_so_far_per_t=[])
    p=dict(unit=meta,exploration_sets=[['X','Z']],per_t=[],population_events=[],policy_commits=[])
    history={v:[None]*3 for v in ('X','Z','Y')};rows=[]
    for t in range(3):
        levels=[None,[[0.,1.]],[[1.,2.]]]+([None] if t==0 else [])
        block=dict(t=t,trial_types=['o','i','i'],eligible_recommendation=[False,True,True],
            best_so_far=[1e7,2.,1.],recommended_values=[None,2.,1.],outcome_values=[1e7,1e7,2.,1.],
            chosen_sets=[['X','Z'],['X','Z']],per_trial_cost=[0,2,2],
            levels_by_set={'X+Z':levels},optimal_intervention_set=['X','Z'])
        p['per_t'].append(block)
        rows.append(dict(method='QDCBO-coherent-population',seed='1001',time_index=str(t),trial_index='0',eligible_recommendation='False',population_recommendation_mean='',population_mc_se='',training_incumbent='',cost='0',recommendation_event=''))
        for trial in (1,2):
            index=len(p['population_events']);x=[float(trial-1),float(trial)]; h=deepcopy(history)
            h['X'][t],h['Z'][t]=x
            e=dict(event_id=index,t=t,trial=trial,arm=['X','Z'],x=x,history=h,cost=2,
                training_mean=float(3-trial),training_mc_se=.02,training_samples=2048,scoring_samples=100000,
                population_mean=3-trial+.1,population_mc_se=.01,recommendation_event=index,
                recommendation_population_mean=3-trial+.1,recommendation_population_mc_se=.01)
            p['population_events'].append(e)
            rows.append(dict(method='QDCBO-coherent-population',seed='1001',time_index=str(t),trial_index=str(trial),eligible_recommendation='True',population_recommendation_mean=str(e['population_mean']),population_mc_se='.01',training_incumbent=str(e['training_mean']),cost='2',recommendation_event=str(index)))
        history=deepcopy(e['history']);p['policy_commits'].append({k:e[k] for k in ('t','arm','x','history','training_mean','population_mean','population_mc_se','recommendation_event')})
        info['final_best_so_far_per_t'].append(e['population_mean'])
    p['assigned_blanket']=history
    return p,rows,info,settings,unit,'test'


def test_complete_ledger():
    r=a.validate_payload(*fixture())
    assert r['intervention_coordinate_cost']==12
    assert r['slice_values']==[1.1]*3
    assert r['sum_score_se_upper_bound']==pytest.approx(.03)


@pytest.mark.parametrize('mutation',[
    lambda p,r,i:p['population_events'][1].update(recommendation_event=0),
    lambda p,r,i:p['population_events'][0].update(cost=1),
    lambda p,r,i:p['population_events'][0]['history']['Y'].__setitem__(0,0.),
    lambda p,r,i:p['population_events'][2]['history']['X'].__setitem__(0,-2.),
    lambda p,r,i:p['population_events'][0].update(scoring_samples=99),
    lambda p,r,i:p['population_events'][0].update(recommendation_population_mean=99.),
    lambda p,r,i:p['per_t'][0]['levels_by_set']['X+Z'].__setitem__(-1,[[0.,0.]]),
    lambda p,r,i:p['per_t'][1]['outcome_values'].__setitem__(2,99.),
    lambda p,r,i:p['policy_commits'][0].update(arm=['X']),
    lambda p,r,i:r[0].update(population_recommendation_mean='10000000'),
    lambda p,r,i:r[1].update(eligible_recommendation='False'),
    lambda p,r,i:i.update(engine_sha='wrong'),
    lambda p,r,i:p['unit'].update(action_menu='coarse'),
    lambda p,r,i:p['population_events'][0].update(training_mc_se=-1),
])
def test_corrupted_ledger_rejected(mutation):
    args=fixture();mutation(*args[:3])
    with pytest.raises(ValueError):a.validate_payload(*args)


def test_twenty_seed_paired_contrast_has_fixed_df_and_no_regret():
    rows=[]
    for seed in a.SEEDS:
        for algo,menu in a.METHODS:
            offset=1. if algo=='QDCBO' else 0.
            rows.append(dict(setup='stat',algo=algo,action_menu=menu,seed=seed,
                slice_values=[offset]*3,score_se=[.01]*3,committed_slice_sum=3*offset,sum_score_se_upper_bound=.03))
    r=a.paired_summary(rows,'stat')
    assert r['n']==20 and r['metrics'][0]['ci95']==[1.,1.]
    assert r['metrics'][3]['mean']==3.
    assert not any('regret' in x for x in r['metrics'][0])
    with pytest.raises(ValueError):a.paired_summary(rows[:-1],'stat')


def test_missing_declared_configuration_rejected_before_inference(tmp_path):
    manifest=dict(schema=1,source_sha256={'x':'x'},protocol=dict(family='dynamic-coherent-population-v1',stage='replication'),units=[])
    with pytest.raises(ValueError,match='matrix'):a.summarize(manifest,tmp_path,tmp_path)


def test_identity_companion_uses_complete_learner_trace():
    left=fixture()[0];right=deepcopy(left)
    assert a.validate_identity_pair('protected',left,right)['status']=='passed'
    right['population_events'][0]['training_mean']+=.1
    with pytest.raises(ValueError,match='trace'):a.validate_identity_pair('protected',left,right)


def materialized_unit(tmp_path):
    payload,rows,info,settings,unit,_=fixture()
    settings.update(num_anchor_points=9,stage='resource-pilot',recorded_runtime={'executable':'/test/python'})
    unit.update(id='synthetic',argv=['{python}','-m','ccbo.qdcbo.runner'],outputs=['result/run.csv','result/run.decisions.json','result/run_info.json'])
    for k in ('setup','algo','seed','action_menu'):
        unit['argv'] += ['--'+k.replace('_','-'),str(unit[k])]
    for k in ('T','trials','n_obs','num_anchor_points','predictive_samples','feedback_samples','score_samples'):
        unit['argv'] += ['--'+k.replace('_','-'),str(settings[k])]
    unit['argv']+=['--mechanism-protocol','coherent','--objective-protocol','population-mc','--outdir','{unit_dir}/result']
    manifest=dict(protocol=settings,units=[unit],schema=1,source_sha256={'example':'hash'})
    campaign=a.identity(manifest);folder=tmp_path/campaign/unit['id'];(folder/'result').mkdir(parents=True)
    payload['unit']['engine_sha']=info['engine_sha']='manifest-sha256:'+campaign
    info['csv']=str(folder/'result/run.csv')
    (folder/'result/run.decisions.json').write_text(json.dumps(payload))
    (folder/'result/run_info.json').write_text(json.dumps(info))
    with (folder/'result/run.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    status=dict(status='complete',returncode=0,manifest_id=campaign,unit=unit,protocol=settings,
        started_unix=1.,finished_unix=2.,argv=[x.replace('{python}','/test/python').replace('{unit_dir}',str(folder)) for x in unit['argv']],
        output_sha256={p:a.digest(folder/p) for p in unit['outputs']})
    (folder/'execution.json').write_text(json.dumps(status));(folder/'frozen-manifest.json').write_text(json.dumps(manifest))
    return manifest,unit,folder


def test_immutable_output_and_command_identity(tmp_path):
    m,u,folder=materialized_unit(tmp_path)
    assert a.validate_unit(m,u,folder)['intervention_coordinate_cost']==12
    p=folder/'execution.json';status=a.load(p);status['argv'][0]='/wrong/python';p.write_text(json.dumps(status))
    with pytest.raises(ValueError,match='command'):a.validate_unit(m,u,folder)
    status['argv'][0]='/test/python';p.write_text(json.dumps(status))
    with (folder/'result/run.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='hash'):a.validate_unit(m,u,folder)


@pytest.mark.parametrize('stage',['resource-pilot'])
def test_incomplete_campaign_emits_no_effects(tmp_path,stage):
    files=['ccbo/qdcbo/runner.py','ccbo/qdcbo/population.py','ccbo/qdcbo/coherent.py','scripts/run_frozen_unit.py','third_party/DCBO/dcbo/bases/root.py']
    sources={}
    for rel in files:
        p=tmp_path/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n');sources[rel]=a.digest(p)
    protocol=dict(family='dynamic-coherent-population-v1',stage=stage,T=3,trials=10,n_obs=100,
        num_anchor_points=100,feedback_samples=2048,score_samples=100000,predictive_samples=1024,
        recorded_runtime={'python':'test'},replication_seeds=a.SEEDS,expected_units=180,
        precision_gate={'accepted':False},runtime_expected={},runtime_imports={})
    seeds=a.SEEDS if stage=='replication' else [1001]
    units=[dict(id=f'{s}-{algo}-{menu}-{seed}',setup=s,algo=algo,action_menu=menu,seed=seed)
        for s in a.SETUPS for algo,menu in a.METHODS for seed in seeds]
    m=dict(schema=1,source_sha256=sources,protocol=protocol,units=units)
    result=a.summarize(m,tmp_path/'results',tmp_path/'source')
    assert not result['inference_allowed'] and not result['complete']
    assert result['paired_effects']==[] and result['validated_units']==0


def test_final_requires_all_precision_evidence(tmp_path):
    with pytest.raises(ValueError,match='six'):
        a.validate_precision_gate({'predictive_samples':1024,'precision_gate':{'accepted':True,'chosen_particles':1024,'reports':[]}},tmp_path,{})
