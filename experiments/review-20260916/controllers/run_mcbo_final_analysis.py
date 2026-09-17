"""Analyze both frozen matrices after all tasks end; STOP vetoes inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

EXPECTED_MANIFEST_IDS = {'main': '83ab2e3e1b2ffe47b8dc600ec3bb74ceaf59f33c63ff0e5d51f84159b18f1b39', 'protected': '7e2d1009e60237113eba968c51b9866b08ac21cf7a2022621d43e1358ba2a288'}


def verified_analyzer(release):
    """Verify canonical manifests and every source byte using only stdlib."""
    release=Path(release)
    manifests={key:json.loads((release/(key+'-manifest.json')).read_text()) for key in EXPECTED_MANIFEST_IDS}
    sources=None
    for key,manifest in manifests.items():
        actual=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if actual!=EXPECTED_MANIFEST_IDS[key]:raise ValueError('Unexpected frozen manifest: '+key)
        if sources is None:sources=manifest['source_sha256']
        elif sources!=manifest['source_sha256']:raise ValueError('Analysis manifest source mismatch')
    if not sources or 'scripts/summarize_mcbo_campaign.py' not in sources:raise ValueError('Analyzer source is not frozen')
    for relative,sha in sources.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe source path')
        if hashlib.sha256((release/relative).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen source mismatch: '+relative)
    spec=importlib.util.spec_from_file_location('verified_frozen_analysis',release/'scripts/summarize_mcbo_campaign.py')
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    return analyzer,manifests


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    mod,manifests=verified_analyzer(a.release)
    main=manifests['main'];protected=manifests['protected']
    reports={'main':mod.summarize(main,a.results),'protected':mod.summarize(protected,a.results,main,a.results)}
    stopped=(a.results/'STOP.json').exists()
    for name,report in reports.items():
        if stopped:
            report['inference_status']='blocked';report['comparisons']=[];report.pop('seed_level_metrics',None)
            report['failures'].append(dict(unit_id='campaign',status='stopped',error='Persistent STOP marker requires investigation'))
            report['failure_count']+=1;report['failure_counts_by_status']['stopped']=1
        # Exclusive creation prevents overwritten analyses/implicit retries.
        with (a.results/(name+'-strict-analysis.json')).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False)
    if any(r['inference_status']=='blocked' for r in reports.values()):raise SystemExit(2)
if __name__=='__main__':main()
