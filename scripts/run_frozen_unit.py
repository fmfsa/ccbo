"""Run a declared experiment unit with code/output hashes and no stale resumes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def manifest_id(m):
    return hashlib.sha256(json.dumps(m, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def relative(root, rel):
    if Path(rel).is_absolute() or '..' in Path(rel).parts:
        raise ValueError('Expected relative path: '+rel)
    return root/rel

def output_hashes(unit, folder):
    if not unit.get('outputs'):
        raise ValueError('Expected outputs must be declared')
    hashes = {}
    for rel in unit['outputs']:
        p = relative(folder, rel)
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError('Missing/empty output: '+rel)
        hashes[rel] = digest(p)
    return hashes

def write(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)

def run(m, index, results, root=ROOT):
    if m.get('schema') != 1 or not m.get('source_sha256'):
        raise ValueError('Need schema 1 and frozen source hashes')
    for rel, sha in m['source_sha256'].items():
        p = relative(root, rel)
        if not p.is_file() or digest(p) != sha:
            raise ValueError('Frozen source mismatch: '+rel)
    ids = [u['id'] for u in m['units']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate unit IDs')
    if index < 0 or index >= len(ids):
        raise ValueError('Unit index out of bounds')
    unit = m['units'][index]; uid = unit['id']; campaign = manifest_id(m)
    if not uid or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in uid):
        raise ValueError('Invalid unit ID')
    folder = results.resolve()/campaign/uid
    status_path = folder/'execution.json'
    if folder.exists():
        if not status_path.exists():
            raise ValueError('Existing incomplete directory; preserve and investigate')
        status = json.loads(status_path.read_text())
        if status.get('status') != 'complete' or status.get('manifest_id') != campaign:
            raise ValueError('Unit incomplete/failed; preserve and investigate')
        if status.get('output_sha256') != output_hashes(unit, folder):
            raise ValueError('Completed output changed; refusing stale resume')
        return status
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    env.update({'PYTHONHASHSEED':'0','PYTHONPATH':str(root),'CCBO_GIT_SHA':'manifest-sha256:'+campaign})
    env.update({str(k):str(v) for k,v in unit.get('env',{}).items()})
    argv = [a.replace('{python}',sys.executable).replace('{unit_dir}',str(folder)) for a in unit['argv']]
    status = {'schema':1,'manifest_id':campaign,'protocol':m['protocol'],'unit':unit,'argv':argv,'status':'running','started_unix':time.time()}
    write(status_path,status);write(folder/'frozen-manifest.json',m)
    try:
        result = subprocess.run(argv,cwd=root,env=env,check=False)
        status['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Experiment exited '+str(result.returncode))
        status['output_sha256'] = output_hashes(unit,folder)
        status['status'] = 'complete'
    except BaseException as exc:
        status['status']='failed';status['error']=str(exc)
        raise
    finally:
        status['finished_unix']=time.time();write(status_path,status)
    return status

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--index',type=int,required=True,help='Zero-based unit index')
    p.add_argument('--results',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(json.loads(a.manifest.read_text()),a.index,a.results),sort_keys=True))
