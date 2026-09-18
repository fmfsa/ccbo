"""Reproduce the paper's empirical figures/tables from audited seed-level data.

Run: python experiments/plot.py --results OUT --outdir FIGURES --suite all
Dependencies: numpy, scipy, matplotlib. Inputs are audited portable experiment outputs; no bundled result data.
Intervals use seeds as replication units; no population score selects an action.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA={}
FIG=TAB=None
def read(name):return DATA.get(name,[])
def est(values):
    x=np.asarray(values,dtype=float);n=len(x)
    assert n in (20,30) and np.isfinite(x).all(), (n,x)
    return float(x.mean()),float(t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n))
def fmt(values,digits=3):
    m,h=est(values);return f'${m:.{digits}f} \\pm {h:.{digits}f}$'
def line(ax,rows,xkey,ykey,label,color,style='-',width=1.5):
    groups=defaultdict(list)
    for r in rows:groups[float(r[xkey])].append(float(r[ykey]))
    x=sorted(groups); mh=[est(groups[v]) for v in x]
    m=np.array([a for a,b in mh]);h=np.array([b for a,b in mh])
    ax.plot(x,m,color=color,linestyle=style,lw=width,label=label)
    ax.fill_between(x,m-h,m+h,color=color,alpha=.13,lw=0)
def axis(ax,title,xlabel,ylabel=None):
    ax.set_title(title,loc='left',fontsize=9)
    ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18,lw=.5)
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
plt.rcParams.update({'font.family':'sans-serif','font.size':8,'axes.labelsize':8,
    'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.formatter.useoffset':False})
BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; YELLOW='#B38B00'; PURPLE='#CC79A7'; GREY='#555555'
LABELS=['Fine / full','Fine / coarse','Quotient / coarse']
COLORS=[BLUE,GREEN,ORANGE]; STYLES=['-','-.','--']
def smallnum(x):
    if x==0:return '0'
    if abs(x)<.0005:
        mant,exp=f'{x:.2e}'.split('e');return mant+r'\times10^{'+str(int(exp))+'}'
    return f'{x:.3f}'

def family():
    dy=read('dynamic_curves.csv');mc=read('mcbo_curves.csv')
    assert (not dy or len(dy)==4860) and (not mc or len(mc)==12000)
    assert dy or mc
    both=bool(dy and mc)
    fig=plt.figure(figsize=(7.0,4.6 if both else 2.65)); gs=fig.add_gridspec(2 if both else 1,6,hspace=.66,wspace=.85)
    for i,(setup,title) in enumerate(zip(['stat','ind','nonstat_regularized'] if dy else [],['Stationary','Independent','Nonstationary (regularized)'])):
        ax=fig.add_subplot(gs[0,2*i:2*i+2])
        for k,lab in enumerate(LABELS):
            for sl in range(3):
                rs=[r for r in dy if r['setup']==setup and r['configuration']==lab and int(r['slice'])==sl]
                line(ax,rs,'intervention_round','population_recommendation_mean',lab if sl==0 else None,COLORS[k],STYLES[k])
        for x in [9.5,18.5]:ax.axvline(x,color='.65',lw=.7)
        ax.set_xticks([1,9,18,27]);axis(ax,f'({chr(97+i)}) {title}','Intervention round', 'Population outcome' if i==0 else None)
    handles=fig.axes[0].get_legend_handles_labels() if dy else None
    for i,env in enumerate(['ToyGraph','PSAGraph'] if mc else []):
        ax=fig.add_subplot(gs[1 if both else 0,3*i:3*i+3])
        for k,(algo,menu) in enumerate([('MCBO','full'),('MCBO','coarse'),('QMCBO','coarse')]):
            rs=[r for r in mc if (r['env'],r['algo'],r['menu'])==(env,algo,menu)]
            line(ax,rs,'trial','recommendation_population_estimate',LABELS[k],COLORS[k],STYLES[k])
        axis(ax,f'({chr((100 if both else 97)+i)}) {env}','Sequential round','Population reward' if i==0 else None)
    handles=handles or fig.axes[0].get_legend_handles_labels()
    fig.legend(*handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.015))
    fig.subplots_adjust(bottom=.14 if both else .30,top=.95,left=.085,right=.985)
    save(fig,'family_suite' if both else 'dynamic_population' if dy else 'mcbo_population')
    lines=[r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lccc}',r'\toprule',r'SCM & \shortstack{Fine/\\full} & \shortstack{Fine/\\coarse} & \shortstack{Quotient/\\coarse} \\',r'\midrule',r'\multicolumn{4}{l}{Dynamic: sum of committed outcomes $\downarrow$} \\']
    for setup,title in zip(['stat','ind','nonstat_regularized'] if dy else [],['Stationary','Independent','Regularized']):
        cells=[]
        for lab in LABELS:
            vals=defaultdict(float)
            for r in dy:
                if r['setup']==setup and r['configuration']==lab and int(r['trial_in_slice'])==9:vals[r['seed']]+=float(r['population_recommendation_mean'])
            m,h=est(list(vals.values()));cells.append(f'\\shortstack{{${m:.3f}$\\\\$\\pm {h:.3f}$}}')
        lines.append(title+' & '+' & '.join(cells)+r' \\[2pt]')
    lines.extend([r'\midrule',r'\multicolumn{4}{l}{Model-based: final recommendation reward $\uparrow$} \\'])
    for env in ['ToyGraph','PSAGraph'] if mc else []:
        cells=[]
        for algo,menu in [('MCBO','full'),('MCBO','coarse'),('QMCBO','coarse')]:
            vals=[float(r['recommendation_population_estimate']) for r in mc if (r['env'],r['algo'],r['menu'],int(r['trial']))==(env,algo,menu,100)]
            m,h=est(vals);meantext=f'{m:.9f}' if 0<h<.0005 else f'{m:.3f}'
            cells.append(f'\\shortstack{{${meantext}$\\\\$\\pm {smallnum(h)}$}}')
        lines.append(env+' & '+' & '.join(cells)+r' \\[2pt]')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\endgroup'])
    lines=[v for v in lines if not (('Dynamic: sum' in v and not dy) or ('Model-based: final' in v and not mc))]
    (TAB/('family_effects.tex' if both else 'dynamic_effects.tex' if dy else 'mcbo_effects.tex')).write_text('\n'.join(lines)+'\n')
    rows=read('mcbo_invariance.csv')
    if not rows:return
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'SCM & Method & Exact traces & Max. endpoint $|\Delta|$ & Max. round $|\Delta|$ \\',r'\midrule']
    for env in ['ToyGraph','PSAGraph'] if mc else []:
        for algo in ['MCBO','QMCBO']:
            rs=[r for r in rows if r['env']==env and r['algo']==algo];assert len(rs)==20
            equal=sum(r['full_trace_equal']=='True' for r in rs)
            end=max(abs(float(r['endpoint_reward_difference'])) for r in rs)
            rnd=max(float(r['max_recommendation_reward_difference']) for r in rs)
            num=smallnum
            lines.append(f'{env} & {algo} & {equal}/20 & ${num(end)}$ & ${num(rnd)}$'+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (TAB/'qmcbo_e2.tex').write_text('\n'.join(lines)+'\n')

def static():
    cs=read('static_curves.csv');outs=read('static_outcomes.csv')
    assert len(outs)==720 and len({r['unit_id'] for r in outs})==720
    def sel(scm,cond,method):return [r for r in cs if (r['scm'],r['cond'],r['method'])==(scm,cond,method)]
    methods={'CBO':(BLUE,'-'),'QCBO':(ORANGE,'--'),'BO':(YELLOW,':'),'BO-S':(GREY,'-.'),'CBO-NP':(PURPLE,'-.'),'CEO':('#56B4E9',':'),'HQCBO':(GREEN,'-'),'CBO-FALLBACK':('#7F4F24','--')}
    fig,axs=plt.subplots(2,1,figsize=(3.35,3.4))
    for ax,scm,c0,c1,title in zip(axs,['ParallelParent','FrontDoor'],['A0','B0'],['A1','B1'],['(a) ParallelParent','(b) FrontDoor']):
        for method in ['CBO','QCBO','BO']:
            col,sty=methods[method];line(ax,sel(scm,c0,method),'cum_cost','recommendation_population',method,col,sty)
        line(ax,sel(scm,c1,'CBO'),'cum_cost','recommendation_population','CBO, misspecified',BLUE,'--')
        axis(ax,title,'Total intervention cost','Population outcome')
        ax.set_xlim(12,100)
    axs[0].axhline(4.25,color='.45',ls=':',lw=.7)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(left=.18,right=.99,top=.95,bottom=.24,hspace=.72);save(fig,'minimal_misspec')
    fig,axs=plt.subplots(2,1,figsize=(3.35,3.5))
    for method in ['BO','CBO','QCBO','HQCBO','CEO']:
        rs=sel('MediatedChain','C0',method);col,sty=methods[method]
        line(axs[0],rs,'cum_cost','recommendation_population',method,col,sty)
        byseed=defaultdict(list)
        for r in rs:byseed[r['seed']].append(r)
        cumulative=[]
        for ss in byseed.values():
            total=0
            for r in sorted(ss,key=lambda a:float(a['cum_cost'])):
                total+=float(r['recommendation_regret']);cumulative.append(dict(r,integrated_gap=total))
        line(axs[1],cumulative,'cum_cost','integrated_gap',method,col,sty)
    axs[0].axhline(4,color='.4',ls=':',lw=.7);axs[0].axhline(.04,color='.4',ls=':',lw=.7)
    x=np.arange(12,121);axs[1].plot(x,3.96*(x-11),color='.4',ls=':',lw=.8)
    for ax,title,y in zip(axs,['(a) MediatedChain','(b) Cost-indexed gap sum'],['Population outcome','Sum of\nrecommendation gaps']):
        axis(ax,title,'Total intervention cost',y);ax.set_xlim(12,120)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(left=.2,right=.99,top=.95,bottom=.24,hspace=.72);save(fig,'minimal_refine')
    fig,axs=plt.subplots(1,3,figsize=(7,2.7))
    for i,(ax,scm,cond) in enumerate(zip(axs,['ParallelParent','FrontDoor','MediatedChain'],['A0','B0','C0'])):
        for method in methods:
            rs=sel(scm,cond,method)
            if rs:line(ax,rs,'cum_cost','recommendation_population',method,*methods[method])
        axis(ax,f'({chr(97+i)}) {scm}','Total intervention cost','Population outcome' if i==0 else None)
        ax.set_xlim(12,120 if i==2 else 100)
    handles={}
    for ax in axs:
        h,l=ax.get_legend_handles_labels();handles.update(zip(l,h))
    fig.legend(handles.values(),handles.keys(),loc='lower center',ncol=4,frameon=False)
    fig.subplots_adjust(left=.07,right=.99,top=.86,bottom=.31,wspace=.38);save(fig,'cost_indexed')
    groups=defaultdict(list)
    for r in outs:groups[r['scm'],r['cond'],r['method']].append(r)
    order={'ParallelParent':0,'FrontDoor':1,'MediatedChain':2}
    keys=sorted(groups,key=lambda k:(order[k[0]],k[1],k[2]))
    lines=[r'\begin{tabular}{llrr}',r'\toprule',r'Condition & Method & Final recommendation gap & Cost-indexed gap sum \\',r'\midrule']
    costs=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Condition & Method & Budget & Init. & Sequential & Refinement & Spent & Unspent \\',r'\midrule']
    short={'ParallelParent':'PP','FrontDoor':'FD','MediatedChain':'MC'}
    for k in keys:
        rs=groups[k];scm,cond,method=k
        name={'CBO-NP':r'CBO$-$np','BO-S':'BO-S'}.get(method,method)
        label=short[scm]+' '+cond
        lines.append(label+' & '+name+' & '+fmt([r['final_recommendation_regret'] for r in rs])+' & '+fmt([r['cost_integrated_recommendation_regret'] for r in rs],2)+r' \\')
        cells=[f'{np.mean([float(r[f]) for r in rs]):.2f}' for f in ['budget','init_cost','sequential_cost','refinement_cost','actual_cost','unspent_budget']]
        costs.append(label+' & '+name+' & '+' & '.join(cells)+r' \\')
    for lines_,name in [(lines,'minimal_ablations'),(costs,'cost_accounting')]:
        lines_.extend([r'\bottomrule',r'\end{tabular}']);(TAB/(name+'.tex')).write_text('\n'.join(lines_)+'\n')

def extract(results,suite,audit):
    """Extract only after the complete declared matrix passes analyze.summarize."""
    import analyze
    rows=[];outcomes=[];paired={}
    for record in audit['units']:
        folder=results/'paper'/suite/record['unit_id'];base=dict(seed=record['seed'],unit_id=record['unit_id'])
        if suite=='static':
            scores=analyze.load(folder/'result/scores.json');summary=analyze.load(folder/'result/summary.json');events=analyze.load(folder/'result/events.json')
            key={k:record[k] for k in ('scm','cond','method')}
            for c in scores['checkpoints']:rows.append(dict(base,**key,**{('cum_cost' if k=='cost' else k):v for k,v in c.items()}))
            outcomes.append(dict(base,**key,final_recommendation_regret=record['final_recommendation_regret'],cost_integrated_recommendation_regret=record['cost_integrated_recommendation_regret'],budget=summary['budget'],init_cost=summary['init_cost'],sequential_cost=sum(e['cost'] for e in events if e['phase']=='sequential'),refinement_cost=summary.get('split_init_cost',0),actual_cost=summary['actual_cost'],unspent_budget=summary['budget']-summary['actual_cost']))
        else:
            data=analyze.load(analyze.typed_files(folder)['decisions'])
            if suite=='dynamic':
                label={('DCBO','native'):LABELS[0],('DCBO','coarse'):LABELS[1],('QDCBO','native'):LABELS[2]}[record['algo'],record['action_menu']]
                for e in data['population_events']:
                    rows.append(dict(base,setup=record['setup'],configuration=label,slice=e['t'],trial_in_slice=e['trial'],intervention_round=e['t']*9+e['trial'],population_recommendation_mean=e['recommendation_population_mean']))
            else:
                events=[e for e in data['events'] if e['phase']=='sequential']
                if not record['misspec']:
                    for i,e in enumerate(events,1):rows.append(dict(base,env=record['env_name'],algo=record['algo'],menu=record['menu'],trial=i,recommendation_population_estimate=e['recommendation_population_estimate']))
                paired[record['unit_id']]=(record,events)
    if suite=='static':DATA.update({'static_curves.csv':rows,'static_outcomes.csv':outcomes})
    elif suite=='dynamic':DATA['dynamic_curves.csv']=rows
    else:DATA['mcbo_curves.csv']=rows
    return paired


def protected_table(results,full):
    """Failures stay failures; a table is emitted only for 80 decoded valid pairs."""
    import analyze
    if len(full['protected_checks'])!=80 or full['missing_units'] or any(f['error']!='Protected executed trace differs' for f in full['failures']):return
    rows={r['unit_id']:r for r in full['units']};records=[]
    def rewards(row):
        folder=results/'paper/mcbo'/row['unit_id']
        payload=analyze.load(analyze.typed_files(folder)['decisions'])
        return [e['recommendation_population_estimate'] for e in payload['events'] if e['phase']=='sequential']
    for check in full['protected_checks']:
        edited=rows[check['unit_id']]
        base=next(r for r in rows.values() if not r['misspec'] and (r['env_name'],r['algo'],r['menu'],r['seed'])==(edited['env_name'],edited['algo'],edited['menu'],edited['seed']))
        delta=[a-b for a,b in zip(rewards(edited),rewards(base))]
        records.append(dict(env=edited['env_name'],algo=edited['algo'],full_trace_equal=str(check['exact_equal']),endpoint_reward_difference=delta[-1],max_recommendation_reward_difference=max(map(abs,delta))))
    DATA['mcbo_invariance.csv']=records


def main():
    import argparse
    import analyze
    global FIG,TAB
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True,help='Same OUT passed to experiments/run.py; contains paper/SUITE.')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--suite',choices=['static','dynamic','mcbo','family','all'],default='all')
    a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    FIG=a.outdir/'figures';TAB=a.outdir/'tables';FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
    suites=['static','dynamic','mcbo'] if a.suite=='all' else ['dynamic','mcbo'] if a.suite=='family' else [a.suite]
    audits={}
    # Gate every requested suite before producing any figure or inferential table.
    for suite in suites:
        full=analyze.summarize(a.results,suite)
        (a.outdir/(suite+'-audit.json')).write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
        audit=analyze.summarize(a.results,suite,main_only=True) if suite=='mcbo' else full
        if suite=='mcbo':
            (a.outdir/'mcbo-main-audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
            print('MCBO protected checks:',len(full['protected_checks']),'required failures:',sum(c['required'] and not c['exact_equal'] for c in full['protected_checks']))
        if not audit['inference_allowed']:
            raise SystemExit(f'{suite}: complete validated matrix required; inspect {a.outdir}/{suite}-audit.json')
        audits[suite]=audit
        if suite=='mcbo':
            (TAB/'qmcbo_e2.tex').unlink(missing_ok=True)
            protected_table(a.results,full)
    for suite,audit in audits.items():extract(a.results,suite,audit)
    if 'static' in suites:static()
    if 'dynamic' in suites or 'mcbo' in suites:family()
    print('Wrote audited figures and tables to',a.outdir)

if __name__=='__main__':main()
