"""Recompute frozen dynamic summaries from all180 unit records; no HPC access."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

SETUPS=('stat','ind','nonstat_regularized')
CONFIGS=(('DCBO','native'),('DCBO','coarse'),('QDCBO','native'))
LABELS=('Fine / full','Fine / coarse','Quotient / coarse')
SEEDS=list(range(2000,2020))
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def estimate(vals):
    require(len(vals)==20 and all(math.isfinite(v) for v in vals),'Invalid sample')
    mean=statistics.mean(vals);se=statistics.stdev(vals)/math.sqrt(20);half=float(t.ppf(.975,19))*se
    return dict(mean=mean,between_seed_se=se,ci95=[mean-half,mean+half],n=20)
def same(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11)
def check_est(actual,expected,where):
    require(same(actual['mean'],expected['mean']) and same(actual['between_seed_se'],expected['between_seed_se']) and all(same(x,y) for x,y in zip(actual['ci95'],expected['ci95'])),where)
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(__file__).parent.parent/'dynamic-final-strict-analysis.json');p.add_argument('--outdir',type=Path,default=Path(__file__).parent);a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    raw=json.loads(a.source.read_text());units=raw['units']
    require(raw['complete'] and raw['inference_allowed'] and raw['expected_units']==180 and raw['validated_units']==180 and not raw['unavailable'],'Audit not complete')
    require(len(units)==180,'Wrong unit count');keys=[(r['setup'],r['algo'],r['action_menu'],r['seed']) for r in units]
    require(len(set(keys))==180 and set(keys)=={(s,al,m,n) for s in SETUPS for al,m in CONFIGS for n in SEEDS},'Wrong180unit matrix')
    for r in units:
        require(len(r['slice_values'])==len(r['score_se'])==3,'Bad slice count')
        require(same(sum(r['slice_values']),r['committed_slice_sum']),'Unit sum mismatch')
        require(all(math.isfinite(x) and x>=0 for x in r['score_se']),'Invalid scoringSE')
        require(same(sum(r['score_se']),r['sum_score_se_upper_bound']),'Scoring sum upper bound mismatch')
    groups=[];contrasts=[];aux=[]
    for s in SETUPS:
        for (al,m),label in zip(CONFIGS,LABELS):
            rs=sorted((r for r in units if (r['setup'],r['algo'],r['action_menu'])==(s,al,m)),key=lambda r:r['seed'])
            slices=[estimate([r['slice_values'][j] for r in rs]) for j in range(3)];total=estimate([r['committed_slice_sum'] for r in rs])
            existing=next(r for r in raw['method_descriptive_summaries'] if (r['setup'],r['algo'],r['action_menu'])==(s,al,m))
            require(existing['n']==20 and same(total['mean'],existing['mean_committed_slice_sum']) and all(same(x['mean'],y) for x,y in zip(slices,existing['mean_slice_values'])),'Group means mismatch')
            require(same(statistics.mean(r['intervention_coordinate_cost'] for r in rs),existing['mean_coordinate_cost']),'Cost mismatch')
            require(same(statistics.mean(r['total_runtime_seconds'] for r in rs),existing['mean_runtime_seconds']),'Runtime mismatch')
            groups.append(dict(setup=s,algo=al,action_menu=m,label=label,slices=slices,total=total))
            row=dict(setup=s,configuration=label,n=20,sum_mean=total['mean'],sum_ci_low=total['ci95'][0],sum_ci_high=total['ci95'][1],mean_coordinate_cost=existing['mean_coordinate_cost'],min_coordinate_cost=min(r['intervention_coordinate_cost'] for r in rs),max_coordinate_cost=max(r['intervention_coordinate_cost'] for r in rs),mean_runtime_seconds=existing['mean_runtime_seconds'],mean_sum_scoring_se_upper_bound=statistics.mean(r['sum_score_se_upper_bound'] for r in rs),max_sum_scoring_se_upper_bound=max(r['sum_score_se_upper_bound'] for r in rs))
            for j in range(3):row.update({f'slice{j}_mean':slices[j]['mean'],f'slice{j}_ci_low':slices[j]['ci95'][0],f'slice{j}_ci_high':slices[j]['ci95'][1],f'slice{j}_mean_score_se':statistics.mean(r['score_se'][j] for r in rs),f'slice{j}_max_score_se':max(r['score_se'][j] for r in rs)})
            aux.append(row)
        pairs=[]
        for seed in SEEDS:
            find=lambda al,m:next(r for r in units if (r['setup'],r['algo'],r['action_menu'],r['seed'])==(s,al,m,seed))
            f,q=find('DCBO','coarse'),find('QDCBO','native')
            pairs.append(dict(seed=seed,slice_difference=[b-c for b,c in zip(q['slice_values'],f['slice_values'])],sum_difference=q['committed_slice_sum']-f['committed_slice_sum']))
        estimates=[dict(metric='slice'+str(j),**estimate([r['slice_difference'][j] for r in pairs])) for j in range(3)]+[dict(metric='committed_slice_sum',**estimate([r['sum_difference'] for r in pairs]))]
        old=next(r for r in raw['paired_effects'] if r['setup']==s)
        require(old['n']==20 and sorted(r['seed'] for r in old['paired_seed_values'])==SEEDS,'Invalid paired seeds')
        for new in estimates:check_est(new,next(v for v in old['metrics'] if v['metric']==new['metric']),s+':'+new['metric'])
        for new in pairs:
            prior=next(r for r in old['paired_seed_values'] if r['seed']==new['seed'])
            require(all(same(x,y) for x,y in zip(new['slice_difference'],prior['slice_difference'])) and same(new['sum_difference'],prior['sum_difference']),'Paired raw values mismatch')
        contrasts.append(dict(setup=s,contrast='quotient/coarse minus fine/coarse',metrics=estimates,seed_values=pairs))
    result=dict(schema=1,source_sha256=sha(a.source),generator_sha256=sha(__file__),manifest_id=raw['manifest_id'],validation='Independent recomputation passed; no discrepancies in180unit matrix, unit sums,9group means/costs/runtimes,12paired estimates or60pair records.',limitations='Recomputation starts from strict-audit unit records; does not revalidate raw event or executable source hashes. Intervals are pointwise Studentt df19 across finite-MC scores, without multiplicity adjustment. Sum scoringSE uses an upper bound, not independence.',groups=groups,paired_effects=contrasts,seed_records=units)
    (a.outdir/'dynamic-recomputed-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(a.outdir/'dynamic-auxiliary-summary.csv',aux)
    flat=[]
    for r in units:
        row={k:r[k] for k in ('unit_id','setup','algo','action_menu','seed','committed_slice_sum','intervention_coordinate_cost','total_runtime_seconds','sum_score_se_upper_bound')}
        for j in range(3):row[f'slice{j}']=r['slice_values'][j];row[f'slice{j}_score_se']=r['score_se'][j]
        flat.append(row)
    write_csv(a.outdir/'dynamic-seed-records.csv',flat)
    plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.45));colors=['#2463A5','#639B8F','#D47720'];styles=['-','--',':'];markers=['o','s','^']
    for ax,s,title in zip(axes,SETUPS,('Stationary','Independent','Nonstationary (regularized)')):
        for k,label in enumerate(LABELS):
            row=next(r for r in groups if r['setup']==s and r['label']==label);means=[v['mean'] for v in row['slices']];err=[v['ci95'][1]-v['mean'] for v in row['slices']]
            ax.errorbar(range(3),means,yerr=err,label=label,color=colors[k],linestyle=styles[k],marker=markers[k],markersize=4,capsize=2,linewidth=1.2,elinewidth=.8,zorder=3-k)
        ax.set_title(title);ax.set_xticks([0,1,2]);ax.set_xlabel('Committed slice');ax.grid(axis='y',color='.9',linewidth=.5);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Population outcome (lower is better)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.51,-.005))
    fig.subplots_adjust(left=.08,right=.99,top=.85,bottom=.29,wspace=.36)
    fig.savefig(a.outdir/'dynamic-population.pdf');fig.savefig(a.outdir/'dynamic-population.png',dpi=220);plt.close(fig)
    def cell(est):return f"${est['mean']:.3f} \\pm {est['ci95'][1]-est['mean']:.3f}$"
    lines=[r'\begin{table*}[t]',r'\centering',r'\small',r'\begin{tabular}{lrrrr}',r'\hline',r'SCM & Fine/full & Fine/coarse & Quotient/coarse & Paired Q$-$fine/coarse \\',r'\hline']
    for s,title in zip(SETUPS,('Stationary','Independent','Regularized nonstationary')):
        cells=[cell(next(r for r in groups if r['setup']==s and r['label']==label)['total']) for label in LABELS]
        cells.append(cell(next(r for r in contrasts if r['setup']==s)['metrics'][-1]));lines.append(title+' & '+' & '.join(cells)+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\caption{Sum of committed-slice population outcomes across three slices (lower is better), mean $\pm$ pointwise 95\% Student-$t$ half-width over 20 seeds. The last column is computed from paired seed-level quotient/coarse minus fine/coarse differences. Each method is scored under its own intervention prefix. These are finite-Monte-Carlo policy outcomes, not global-optimum regrets or equal-cost comparisons. The independent matched-menu methods coincide in all scored seed-level slice values. Scoring uncertainty and costs are reported separately.}',r'\label{tab:dynamic-population}',r'\end{table*}']
    (a.outdir/'dynamic-population-summary.tex').write_text('\n'.join(lines)+'\n')
    (a.outdir/'dynamic-population-figure.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{dynamic-population.pdf}
\caption{Committed-slice population outcomes under coherent dynamics, with means and pointwise 95\% Student-$t$ intervals across 20 seeds. Fine/full, fine/coarse and quotient/coarse methods are shown separately; the two matched-menu curves overlap exactly for the independent SCM. Lower is better. Each method's earlier manipulator interventions remain fixed and free past variables are re-simulated. Scoring uses an independent 100,000-particle stream; its uncertainty is reported separately. The nonstationary regularized model is a new integrable companion. Connecting lines guide the eye across discrete slices; no running minimum is carried across slices.}
\label{fig:dynamic-population}
\end{figure*}
''')
    print(result['validation']);print(json.dumps(contrasts,indent=2)[:500])
if __name__=='__main__':main()
