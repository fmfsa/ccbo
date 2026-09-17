"""Scientific ledger validation, independent of scheduler and release metadata."""
from copy import deepcopy
import pytest
from experiments.analyze import dynamic_events


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
    return p,rows,info,settings,unit


def test_complete_ledger():
    r=dynamic_events(*fixture())
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
    lambda p,r,i:p['unit'].update(action_menu='coarse'),
    lambda p,r,i:p['population_events'][0].update(training_mc_se=-1),
])
def test_corrupted_ledger_rejected(mutation):
    args=fixture();mutation(*args[:3])
    with pytest.raises(ValueError):dynamic_events(*args)
