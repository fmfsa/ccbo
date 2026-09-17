"""Complete only the explicitly rejected Bad job name static submission."""
import importlib.util,json,subprocess,time
from pathlib import Path
root=Path('/zhome/15/b/215295/Repositories/ccbo-experiments-20260916')
script=root/'operations/parallelism-20260917/transfer.py'
spec=importlib.util.spec_from_file_location('transfer',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
record=root/'launch-records/parallelism-static-20260917T125712';path=record/'plan.json';p=json.loads(path.read_text());c=p['settings']
assert p['kind']=='static' and p['status']=='blocked' and not p.get('supplement_job')
assert (record/'supplement-submission.txt').read_text().strip()=='Bad job name. Job not submitted.'
assert not m.stop_path('static',c).exists()
rows=m.states(c['original'],record);m.fail_states(rows);indices=p['transferred']
assert indices and all(rows[i]=='PSUSP' and not m.manifest_unit('static',c,i)[3].exists() for i in indices)
chunks=[];lo=hi=indices[0]
for i in indices[1:]:
 if i==hi+1:hi=i
 else:chunks.append(str(lo) if lo==hi else f'{lo}-{hi}');lo=hi=i
chunks.append(str(lo) if lo==hi else f'{lo}-{hi}')
name='ccbo_static_complement['+','.join(chunks)+']%12'
controller=record/'supplement.lsf';assert m.sha(controller)==p['supplement_controller_sha256']
p['retry_reason']='Original submission explicitly rejected: Bad job name. Retry uses compressed equivalent index ranges.';p['retry_name']=name;p['status']='retrying-rejected-name';m.write(path,p)
with controller.open() as f:r=subprocess.run(['bsub','-J',name],stdin=f,text=True,capture_output=True)
(record/'supplement-submission-compressed.txt').write_text(r.stdout+'\n'+r.stderr)
assert r.returncode==0,(r.stdout,r.stderr)
p['supplement_job']=m.jobid(r.stdout);p['status']='supplement-submitted';m.write(path,p)
dependency=f"numended({p['supplement_job']}, == {len(indices)})"
response=m.command(['bsub','-J','ccbo_static_transfer_finalize','-q','hpc','-n','1','-W','00:30','-R','rusage[mem=2GB]','-o',str(record/'finalize-%J.out'),'-e',str(record/'finalize-%J.err'),'-w',dependency,c['python'],str(script),'finalize','--plan',str(path)],record)
p['finalizer_job']=m.jobid(response);p['status']='active';m.write(path,p)
print(json.dumps({k:p[k] for k in ['status','supplement_job','finalizer_job','retry_name']},indent=2))
