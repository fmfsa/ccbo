"""Pure strict-analysis tests; generated fixtures never count as experiments."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/summarize_mcbo_campaign.py'
spec=importlib.util.spec_from_file_location('mcbo_analysis',SCRIPT)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
IMPORTS={'mcbo.mcbo_trial':'third_party/mcbo/mcbo/mcbo_trial.py',
 'mcbo.models.gp_network':'third_party/mcbo/mcbo/models/gp_network.py',
 'mcbo.acquisition_function_optimization.optimize_acqf':'third_party/mcbo/mcbo/acquisition_function_optimization/optimize_acqf.py',
 'functions':'third_party/mcbo/scripts/functions.py'}


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))


def manifest(stage='engineering-preflight',reference=None):
    rounds=20 if stage=='engineering-preflight' else 100
    p=dict(family='mcbo-corrected',adapter_protocol=a.CORE_PROTOCOL,stage=stage,score_samples=100000,
           beta=10.,noise_scale=0.,initial_obs_samples=5,initial_int_samples=2,
           replication_seeds=list(a.SEEDS),iterations=rounds,vendor_revision='fixture',expected_runtime=dict(python='3.9.18',packages={'torch':'1.13.1'}))
    if reference:p['reference_manifest_id']=a.manifest_id(reference)
    units=[]; methods=a.METHODS if stage!='protected-extension' else (a.METHODS[0],a.METHODS[2])
    for env in a.ENVS:
        for algo,menu in methods:
            for seed in ([1001] if stage=='engineering-preflight' else a.SEEDS):
                edit=a.EDITS[env] if stage=='protected-extension' else ''
                u=dict(id=f'{env}-{algo}-{menu}-{seed}',env_name=env,algo=algo,menu=menu,seed=seed,num_trials=rounds,score_samples=100000,misspec=edit)
                u['argv']=['{python}','scripts/run_mcbo_campaign_unit.py','--env',env,'--algo',algo,'--menu',menu,'--seed',str(seed),
                           '--num-trials',str(rounds),'--score-samples','100000','--outdir','{unit_dir}/result']
                if edit:u['argv']+=['--misspec',edit]
                u['outputs']=[a.stem(u)+suffix for suffix in ('.csv','.decisions.json','_info.json')]+['result/runtime.json']
                units.append(u)
    sources={path:'a'*64 for path in IMPORTS.values()}
    sources['scripts/summarize_mcbo_campaign.py']=a.digest(SCRIPT)
    return dict(schema=1,protocol=p,source_sha256=sources,units=units)


def fixture_data(unit):
    env=unit['env_name'];n=3 if env=='ToyGraph' else 6;allowed=a.targets(env,unit['menu']);events=[]
    def event(target,phase,index):
        normalized=[.5]*n; physical=normalized.copy()
        if n==3:
            physical[:2]=[0.,7.5]
            observed_x=0.;z=physical[1] if target[1] else 1.
            reward=math.exp(-z/20)-math.cos(z); y=[observed_x,z,reward];se=0.;reps=1
        else:
            reward=-5. if any(target) else -1.
            y=[65.,26.,physical[2] if target[2] else .2,physical[3] if target[3] else .3,.3,reward];se=.002;reps=100000
        e=dict(phase=phase,index=index,target=target,X=target+normalized,physical_values=physical,network_observation=y,
               measured_y=reward,population_estimate=reward,score_se=se,score_samples=reps,cost=sum(target),recommendation_eligible=target in allowed)
        events.append(e)
        if phase=='sequential':
            eligible=[i for i,r in enumerate(events) if r['recommendation_eligible']]
            winner=max(eligible,key=lambda i:(events[i]['measured_y'],-i))
            e.update(recommendation_event=winner,recommendation_population_estimate=events[winner]['population_estimate'],
                     oracle_best_visited=max(events[i]['population_estimate'] for i in eligible),
                     usable_node_rows=[sum(not r['target'][k] for r in events) for k in range(n)],acquisition_diagnostics=[{'acq_value':0.}])
    for i in range(5):event([0]*n,'init',i)
    for target in allowed:
        if any(target):
            for i in range(2):event(target,'init',i)
    for i in range(unit['num_trials']):event(allowed[0],'sequential',i)
    initial=[e for e in events if e['phase']=='init']
    info=dict(protocol=a.CORE_PROTOCOL,env=env,algo=unit['algo'],menu=unit['menu'],seed=unit['seed'],num_trials=unit['num_trials'],
              misspec=unit['misspec'],score_samples=100000,beta=10.,noise_scale=0.,csv=a.stem(unit)+'.csv',secs=1.,targets=allowed,n_targets=len(allowed),
              initial_rows=len(initial),initial_cost=sum(e['cost'] for e in initial),
              initial_best_population_estimate=max(e['population_estimate'] for e in initial if e['recommendation_eligible']))
    return info,events


def create(root,m):
    for u in m['units']:
        folder=root/a.manifest_id(m)/u['id'];prefix=folder/a.stem(u)
        info,events=fixture_data(u)
        write(str(prefix)+'_info.json',info);write(str(prefix)+'.decisions.json',dict(schema=2,unit=info,events=events))
        with open(str(prefix)+'.csv','w') as f:
            w=csv.writer(f);w.writerow(['trial_number','recommendation_population_estimate','oracle_best_visited'])
            w.writerows((e['index'],e['recommendation_population_estimate'],e['oracle_best_visited']) for e in events if e['phase']=='sequential')
        write(folder/'result/runtime.json',dict(m['protocol']['expected_runtime'],vendor_git_revision='fixture',executable='/fixture/python',
            imports={name:dict(relative_source=path,sha256='a'*64,path='/fixture/'+path) for name,path in IMPORTS.items()}))
        write(folder/'frozen-manifest.json',m)
        write(folder/'execution.json',dict(status='complete',returncode=0,manifest_id=a.manifest_id(m),protocol=m['protocol'],unit=u,
             output_sha256={rel:a.digest(folder/rel) for rel in u['outputs']}))


def resign(folder):
    e=a.load(folder/'execution.json');e['output_sha256']={rel:a.digest(folder/rel) for rel in e['unit']['outputs']};write(folder/'execution.json',e)


class CampaignTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_preflight_has_no_performance_inference(self):
        m=manifest();create(self.root,m);r=a.summarize(m,self.root)
        self.assertEqual(r['inference_status'],'engineering_only');self.assertEqual(r['comparisons'],[])
        self.assertNotIn('seed_level_metrics',r)
        self.assertNotIn('endpoint',r['engineering_audit'][0])

    def test_missing_outputs_and_failed_execution_block(self):
        for failed in (False,True):
            root=self.root/str(failed);m=manifest();create(root,m);folder=root/a.manifest_id(m)/m['units'][0]['id']
            if failed:
                status=a.load(folder/'execution.json');status.update(status='failed',returncode=1);write(folder/'execution.json',status)
            else:(folder/m['units'][0]['outputs'][0]).unlink()
            r=a.summarize(m,root);self.assertEqual(r['inference_status'],'blocked');self.assertEqual(r['failure_count'],1)
            self.assertEqual(r['comparisons'],[])

    def test_wrong_seed_matrix_refused_before_analysis(self):
        m=manifest('replication');m['units'].pop()
        with self.assertRaisesRegex(ValueError,'matrix'):a.summarize(m,self.root)
        m=manifest('replication');m['protocol']['replication_seeds'][-1]=2029
        with self.assertRaisesRegex(ValueError,'seed matrix'):a.summarize(m,self.root)

    def test_psa_observations_cannot_be_recommended_despite_better_measured_y(self):
        m=manifest();create(self.root,m);u=next(u for u in m['units'] if u['env_name']=='PSAGraph')
        folder=self.root/a.manifest_id(m)/u['id'];p=Path(str(folder/a.stem(u))+'.decisions.json')
        data=a.load(p);data['events'][-1]['recommendation_event']=0;write(p,data);resign(folder)
        r=a.summarize(m,self.root);self.assertEqual(r['inference_status'],'blocked')
        self.assertIn('eligible measured argmax',r['failures'][0]['error'])

    def test_rehashed_eligibility_mask_and_runtime_tampering_refused(self):
        for mode in ('mask','runtime'):
            root=self.root/mode;m=manifest();create(root,m);u=m['units'][0];folder=root/a.manifest_id(m)/u['id']
            if mode=='mask':
                p=Path(str(folder/a.stem(u))+'.decisions.json');d=a.load(p);d['events'][0]['recommendation_eligible']=False;write(p,d)
            else:
                p=folder/'result/runtime.json';d=a.load(p);d['packages']['torch']='9.9';write(p,d)
            resign(folder);r=a.summarize(m,root);self.assertEqual(r['inference_status'],'blocked')

    def test_complete_main_and_student_t_direction(self):
        m=manifest('replication');create(self.root,m);r=a.summarize(m,self.root)
        self.assertEqual(r['inference_status'],'complete');self.assertEqual(len(r['comparisons']),4)
        self.assertEqual(r['comparisons'][0]['role'],'primary matched-action-menu comparison')
        vals=[dict(seed=s,fine=0.,quotient=float(s-2000)) for s in a.SEEDS]
        estimate=a.interval(vals,'endpoint');half=2.093024054408263*math.sqrt(35./20.)
        self.assertEqual(estimate['mean_paired_difference'],9.5)
        self.assertAlmostEqual(estimate['ci95'][0],9.5-half)
        self.assertAlmostEqual(estimate['ci95'][1],9.5+half)

    def test_protected_extension_needs_complete_reference_and_compares_full_traces(self):
        main=manifest('replication');extension=manifest('protected-extension',main)
        create(self.root,main);create(self.root,extension)
        self.assertEqual(a.summarize(extension,self.root)['inference_status'],'blocked')
        result=a.summarize(extension,self.root,main,self.root)
        self.assertEqual(result['inference_status'],'validated_protected_traces')
        self.assertEqual(len(result['protected_trace_checks']),80)
        self.assertTrue(all(r['full_trace_equal'] for r in result['protected_trace_checks']))
        self.assertEqual(result['comparisons'],[])

if __name__=='__main__':unittest.main()
