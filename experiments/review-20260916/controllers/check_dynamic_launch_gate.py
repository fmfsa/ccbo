"""Strict chosen-settings nine-pilot gate; conservative100 CPU-hour projection."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

IDS={'pilot':'7099d1ec0e3d5215da0c431636f438b5b8842b80f115308998870bf6599a80c5',
     'final':'d4dd9d45227cd0f36a3061af210fdcf1c9f2765a0e1bb1d363c8e2f9d8579275'}

def check(release,results):
    # Validate identities and every source byte before importing executable
    # analysis code from the release. Use only stdlib until this gate passes.
    manifests={k:json.loads((release/(k+'-manifest.json')).read_text()) for k in IDS}
    for key,manifest in manifests.items():
        actual=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if actual!=IDS[key]:raise ValueError('Unexpected frozen manifest: '+key)
    sources=manifests['pilot']['source_sha256']
    if sources!=manifests['final']['source_sha256']:raise ValueError('Pilot/final source mismatch')
    for relative,sha in sources.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe source path')
        if hashlib.sha256((release/relative).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen source mismatch: '+relative)
    spec=importlib.util.spec_from_file_location('dynamic_analysis',release/'scripts/summarize_dynamic_campaign.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    for key in ('T','trials','n_obs','num_anchor_points','predictive_samples','feedback_samples','score_samples','runtime_expected','runtime_imports','precision_gate'):
        a.require(manifests['pilot']['protocol'][key]==manifests['final']['protocol'][key],'Pilot/final settings differ: '+key)
    # Validate the complete180-unit declaration even before any final jobs exist.
    declaration=a.summarize(manifests['final'],release/'__no_results__',release)
    a.require(declaration['expected_units']==180,'Wrong final matrix')
    audit=a.summarize(manifests['pilot'],results,release)
    a.require(audit['complete'] and audit['validated_units']==9 and not audit['inference_allowed'],'All9 strict pilots must pass')
    times=[max(r['executor_seconds'],r['total_runtime_seconds']) for r in audit['units']]
    a.require(all(math.isfinite(t) and t>0 for t in times),'Invalid timing')
    hours=max(times)*180/3600
    a.require(hours<=100,'Maximum-pilot projection exceeds100CPUh cap')
    return dict(status='launch_gate_passed',manifest_ids=IDS,validated_pilots=9,final_units=180,
        projection=dict(maximum_pilot_seconds=max(times),projected_cpu_hours=hours,cap_cpu_hours=100,
        assumption='180 times maximum chosen-settings pilot walltime,1 CPU; planning estimate only.'),
        science_launched=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--pilot-results',type=Path,required=True);args=p.parse_args()
    print(json.dumps(check(args.release,args.pilot_results),indent=2,allow_nan=False))
