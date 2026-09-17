"""Strict six-preflight gate and declared scaling projection; no scheduler calls."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

IDS={'preflight':'7c7558c600d6fc9b2041bc4a28bbcd225a1091aca483ac9cc9d12c0d1886b002','main':'83ab2e3e1b2ffe47b8dc600ec3bb74ceaf59f33c63ff0e5d51f84159b18f1b39','protected':'7e2d1009e60237113eba968c51b9866b08ac21cf7a2022621d43e1358ba2a288'}
def need(ok,message):
    if not ok:raise ValueError(message)
def identity(m):return hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def projection(rows):
    need(len(rows)==6,'Need six timing rows')
    times={(r['env'],r['algo'],r['menu']):r['wall_seconds'] for r in rows}
    expected={(e,a,m) for e in ('ToyGraph','PSAGraph') for a,m in (('MCBO','full'),('MCBO','coarse'),('QMCBO','coarse'))}
    need(set(times)==expected,'Timing configuration mismatch')
    need(all(math.isfinite(t) and t>0 for t in times.values()),'Invalid timing')
    main=sum(times.values())*5*20/3600
    protected=sum(t for (e,a,m),t in times.items() if (a,m) in (('MCBO','full'),('QMCBO','coarse')))*5*20/3600
    need(main+protected<=500,'Scaling projection exceeds fixed500CPUh cap')
    return dict(main_cpu_hours=main,protected_cpu_hours=protected,total_cpu_hours=main+protected,cap_cpu_hours=500,
        assumption='Per-configuration20-round walltime times5 times20seeds; one CPU per unit. Scaling projection only; GP fit growth and edited graph runtimes may be nonlinear.')
def check(release,results):
    release=Path(release); manifests={k:load(release/(k+'-manifest.json')) for k in IDS}
    for k,m in manifests.items():need(identity(m)==IDS[k],'Wrong frozen '+k+' manifest')
    sources=manifests['main']['source_sha256']
    need(all(m['source_sha256']==sources for m in manifests.values()),'Stage source mismatch')
    for rel,sha in sources.items():
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Unsafe source path')
        need(hashlib.sha256((release/rel).read_bytes()).hexdigest()==sha,'Frozen source mismatch: '+rel)
    spec=importlib.util.spec_from_file_location('mcbo_frozen_analysis',release/'scripts/summarize_mcbo_campaign.py')
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    for m in manifests.values():analyzer.validate_manifest(m)
    audit=analyzer.summarize(manifests['preflight'],Path(results))
    need(audit['inference_status']=='engineering_only' and audit['valid_units']==6 and audit['failure_count']==0,'All six strict preflights including actual runtime/import checks must pass')
    return dict(status='launch_gate_passed',manifest_ids=IDS,preflight_units=6,main_units=120,protected_units=80,
                projection=projection(audit['engineering_audit']),science_launched=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--preflight-results',type=Path,required=True);a=p.parse_args()
    try:r=check(a.release,a.preflight_results)
    except (ValueError,KeyError,TypeError,OSError) as e:
        print(json.dumps(dict(status='launch_blocked',error=str(e))),file=sys.stderr);raise SystemExit(2)
    print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()
