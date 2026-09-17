#!/usr/bin/env python3
"""Owner-only LSF pending-work transfer. Never changes frozen scientific code."""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
BASE=Path('/zhome/15/b/215295/Repositories/ccbo-experiments-20260916')
SETTINGS={
 'mcbo':dict(original='29424838',total=200,cap=14,release='mcbo-campaign-v3',results='mcbo-campaign-v3',controller='mcbo_campaign.lsf',python='/zhome/15/b/215295/venvs/mcbo/bin/python'),
 'static':dict(original='29424844',total=720,cap=12,release='matched-preflight-v3',results='matched-replication',controller='matched_replication.lsf',python='/zhome/15/b/215295/venvs/ccbo/bin/python')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)
def command(argv,record):
 r=subprocess.run(argv,text=True,capture_output=True)
 with (record/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),argv=argv,code=r.returncode,stdout=r.stdout,stderr=r.stderr))+'\n')
 if r.returncode:raise RuntimeError(f'Command failed: {argv}: {r.stderr}')
 return r.stdout

def states(job,record):
 text=command(['bjobs','-a','-noheader','-o','stat job_name',str(job)],record)
 result={}
 for line in text.splitlines():
  fields=line.split(); match=re.search(r'\[(\d+)\]$',fields[-1]) if fields else None
  if not match:raise ValueError('Unparseable scheduler state: '+line)
  result[int(match[1])]=fields[0]
 if not result:raise ValueError('Empty scheduler query')
 return result

def array_counts(job,record):
 text=command(['bjobs','-A','-w',str(job)],record)
 lines=[x.split() for x in text.splitlines() if x.strip()]
 if len(lines)!=2 or len(lines[0])!=len(lines[1]):raise ValueError('Malformed array summary')
 row=dict(zip(lines[0],lines[1]))
 return {k:int(row[k]) for k in ('NJOBS','PEND','DONE','RUN','EXIT','SSUSP','USUSP','PSUSP')}

def batches(items,n=64):
 for i in range(0,len(items),n):yield items[i:i+n]

def fail_states(rows):
 bad={i:s for i,s in rows.items() if s in ('EXIT','UNKWN','ZOMBI')}
 if bad:raise ValueError('Failed/unknown scheduler peers: '+repr(bad))

def manifest_unit(kind,c,index):
 name=('main-manifest.json' if index<=120 else 'protected-manifest.json') if kind=='mcbo' else 'replication-manifest.json'
 offset=(index-1 if index<=120 else index-121) if kind=='mcbo' else index-1
 path=BASE/'releases'/c['release']/name;m=json.loads(path.read_text())
 ident=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 u=m['units'][offset]; folder=BASE/'results'/c['results']/ident/u['id']
 return m,ident,u,folder

def stop_path(kind,c):
 p=BASE/'results'/c['results']
 if kind=='static':p=p/manifest_unit(kind,c,1)[1]
 return p/'STOP.json'
def validate_complete(kind,c,indices):
 checked=set()
 for i in indices:
  m,ident,u,folder=manifest_unit(kind,c,i)
  if ident not in checked:
   for rel,digest in m['source_sha256'].items():
    if sha(BASE/'releases'/c['release']/rel)!=digest:raise ValueError('Frozen source drift: '+rel)
   checked.add(ident)
  st=json.loads((folder/'execution.json').read_text())
  if st.get('status')!='complete' or st.get('manifest_id')!=ident or st.get('unit')!=u:raise ValueError('Incomplete/mismatched unit '+str(i))
  actual={rel:sha(folder/rel) for rel in u['outputs']}
  if st.get('output_sha256')!=actual:raise ValueError('Output hash mismatch '+str(i))

def jobid(text):
 m=re.search(r'Job <(\d+)>',text)
 if not m:raise ValueError('Missing accepted job ID: '+text)
 return m[1]

def settle_suspensions(c,pending,record,timeout=180):
 deadline=time.monotonic()+timeout
 while True:
  snapshot=states(c['original'],record);fail_states(snapshot)
  unsettled={i:snapshot.get(i,'MISSING') for i in pending if snapshot.get(i) not in ('PSUSP','USUSP','DONE')}
  if not unsettled:return snapshot
  if any(v not in ('PEND','RUN') for v in unsettled.values()):raise ValueError('Unexpected suspension states: '+repr(unsettled))
  remaining=deadline-time.monotonic()
  print(f"Waiting for {len(unsettled)} suspension acknowledgements; {max(0,int(remaining))}s remaining",flush=True)
  if remaining<=0:raise ValueError('Suspension settling timed out; original tasks remain owned by original array')
  time.sleep(min(10,remaining))

def start(kind,path=None):
 continuing=path is not None
 if continuing:
  plan=json.loads(path.read_text());kind=plan['kind'];c=plan['settings'];record=path.parent
  if plan.get('status')!='blocked' or plan.get('supplement_job') or (record/'supplement-submission.txt').exists():raise ValueError('Continuation requires blocked pre-submission record')
  if c!=SETTINGS[kind]:raise ValueError('Unexpected continuation settings')
  expected={int(i) for i,state in plan.get('initial_states',{}).items() if state=='PEND'}
  if not expected or set(plan.get('suspended',[]))!=expected:raise ValueError('Not every original PEND suspension request was acknowledged; manual audit required')
  if 'Unexpected transfer state' not in plan.get('error','') and 'Suspension settling timed out' not in plan.get('error',''):raise ValueError('Continuation only supports unsettled suspension failure')
  pending=sorted(expected);plan['transferred']=[];plan['continuation_unix']=time.time();plan['status']='settling';write(path,plan)
 else:
  c=SETTINGS[kind];record=BASE/'launch-records'/('parallelism-'+kind+'-'+time.strftime('%Y%m%dT%H%M%S'))
  record.mkdir(exist_ok=False);plan=dict(kind=kind,settings=c,status='preparing',created=time.time(),transferred=[],suspended=[])
  path=record/'plan.json';write(path,plan)
 try:
  if not continuing:
   # Persistent guard: never repeat a transfer after a partial attempt without operator audit.
   (BASE/'launch-records'/('parallelism-'+kind+'-attempted')).mkdir(exist_ok=False)
   if stop_path(kind,c).exists():raise ValueError('Campaign STOP exists')
   initial=states(c['original'],record);fail_states(initial)
   if set(initial)!=set(range(1,c['total']+1)):raise ValueError('Original array index set incomplete')
   pending=[i for i,s in sorted(initial.items()) if s=='PEND'];plan['initial_states']=initial;write(path,plan)
   if not pending:raise ValueError('No pending work to transfer')
   for chunk in batches(pending):
    plan['suspension_requested']=chunk;write(path,plan)
    command(['bstop']+[f"{c['original']}[{i}]" for i in chunk],record)
    plan['suspended'].extend(chunk);write(path,plan)
  if stop_path(kind,c).exists():raise ValueError('Campaign STOP exists')
  after=settle_suspensions(c,pending,record)
  for i in pending:
   state=after[i]
   if state=='USUSP':command(['bresume',f"{c['original']}[{i}]"],record)
   elif state=='PSUSP':
    if manifest_unit(kind,c,i)[3].exists():raise ValueError('Suspended unit has existing outputs: '+str(i))
    plan['transferred'].append(i)
   elif state not in ('RUN','DONE'):raise ValueError('Unexpected transfer state: '+str((i,state)))
  indices=plan['transferred']
  if not indices:raise ValueError('No safely suspended untouched units')
  script=(BASE/'server-launch'/c['controller']).read_text()
  plan['original_controller_sha256']=hashlib.sha256(script.encode()).hexdigest()
  script=re.sub(r'^#BSUB -J .*\n','',script,flags=re.M)
  old='scheduler_states=$(bjobs -a -noheader -o stat "$LSB_JOBID")'
  assert script.count(old)==1
  new='''scheduler_states=$(bjobs -a -noheader -o stat "$LSB_JOBID")
[[ -n "$scheduler_states" ]] || { echo 'Empty supplementary scheduler query' >&2; exit 2; }
original_states=$(bjobs -a -noheader -o stat ORIGINAL)
[[ -n "$original_states" ]] || { echo 'Empty original scheduler query' >&2; exit 2; }
scheduler_states="$scheduler_states
$original_states"'''.replace('ORIGINAL',c['original'])
  script=script.replace(old,new)
  controller=record/'supplement.lsf';controller.write_text(script)
  plan['supplement_controller_sha256']=sha(controller);plan['status']='ready-to-submit';write(path,plan)
  command(['bash','-n',str(controller)],record)
  # Revalidate suspension immediately before submission; no resumptions yet.
  current=states(c['original'],record);fail_states(current)
  if any(current[i]!='PSUSP' or manifest_unit(kind,c,i)[3].exists() for i in indices):raise ValueError('Transfer precondition changed')
  name='ccbo_'+kind+'_complement['+','.join(map(str,indices))+']%'+str(c['cap'])
  with controller.open() as f:
   r=subprocess.run(['bsub','-J',name],stdin=f,capture_output=True,text=True)
  (record/'supplement-submission.txt').write_text(r.stdout+'\n'+r.stderr)
  if r.returncode:raise RuntimeError('Supplement submission failed')
  plan['supplement_job']=jobid(r.stdout);plan['status']='supplement-submitted';write(path,plan)
  dependency=f"numended({plan['supplement_job']}, == {len(indices)})"
  response=command(['bsub','-J','ccbo_'+kind+'_transfer_finalize','-q','hpc','-n','1','-W','00:30','-R','rusage[mem=2GB]','-o',str(record/'finalize-%J.out'),'-e',str(record/'finalize-%J.err'),'-w',dependency,c['python'],str(Path(__file__).resolve()),'finalize','--plan',str(path)],record)
  plan['finalizer_job']=jobid(response);plan['status']='active';write(path,plan)
  print(json.dumps(plan,indent=2))
 except BaseException as e:
  plan['status']='blocked';plan['error']=repr(e);write(path,plan)
  # Never blindly resume: a successfully submitted supplement may already run.
  raise

def finalize(path):
 plan=json.loads(path.read_text());record=path.parent;c=plan['settings'];kind=plan['kind']
 try:
  if stop_path(kind,c).exists():raise ValueError('Campaign STOP exists')
  counts=array_counts(plan['supplement_job'],record)
  if counts['NJOBS']!=len(plan['transferred']) or counts['DONE']!=len(plan['transferred']) or counts['EXIT']!=0:raise ValueError('Supplement not wholly DONE')
  old=states(c['original'],record);fail_states(old)
  if any(old[i]!='PSUSP' for i in plan['transferred']):raise ValueError('Original transfer ownership changed')
  validate_complete(kind,c,plan['transferred'])
  plan['status']='validated-resuming';plan['validated_unix']=time.time();plan['resumed']=[];write(path,plan)
  for chunk in batches(plan['transferred']):
   if stop_path(kind,c).exists():raise ValueError('STOP appeared during resumption')
   plan['resumption_requested']=chunk;write(path,plan)
   command(['bresume']+[f"{c['original']}[{i}]" for i in chunk],record)
   plan['resumed'].extend(chunk);write(path,plan)
  plan['status']='resumed-for-validated-skips';write(path,plan)
 except BaseException as e:
  plan['status']='blocked-finalization';plan['error']=repr(e);write(path,plan);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['start','continue-suspended','finalize']);p.add_argument('--campaign',choices=SETTINGS);p.add_argument('--plan',type=Path);a=p.parse_args()
 if a.action=='start':
  if not a.campaign:p.error('--campaign required')
  start(a.campaign)
 else:
  if not a.plan:p.error('--plan required')
  if a.action=='continue-suspended':start(None,a.plan)
  else:finalize(a.plan)
