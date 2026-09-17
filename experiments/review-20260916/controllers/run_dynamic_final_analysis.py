"""Strict dynamic final analysis; persistent STOP vetoes all inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

EXPECTED_MANIFEST_IDS = {'final': 'd4dd9d45227cd0f36a3061af210fdcf1c9f2765a0e1bb1d363c8e2f9d8579275'}


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
    if not sources or 'scripts/summarize_dynamic_campaign.py' not in sources:raise ValueError('Analyzer source is not frozen')
    for relative,sha in sources.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe source path')
        if hashlib.sha256((release/relative).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen source mismatch: '+relative)
    spec=importlib.util.spec_from_file_location('verified_frozen_analysis',release/'scripts/summarize_dynamic_campaign.py')
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    return analyzer,manifests


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--results',type=Path,required=True);args=p.parse_args()
    a,manifests=verified_analyzer(args.release)
    try:
        report=a.summarize(manifests['final'],args.results,args.release)
    except Exception as error:
        report=dict(inference_allowed=False,complete=False,status='validation_failed',error=str(error),paired_effects=[])
    if (args.results/'STOP.json').exists():
        report.update(inference_allowed=False,status='campaign_stopped',paired_effects=[],method_descriptive_summaries=[])
        report['stop_record']=a.load(args.results/'STOP.json')
    args.results.mkdir(parents=True,exist_ok=True)
    with (args.results/'final-strict-analysis.json').open('x') as out:json.dump(report,out,indent=2,sort_keys=True,allow_nan=False)
    if not report['inference_allowed']:raise SystemExit(2)
