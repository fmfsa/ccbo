"""Frozen strict final analysis, with a persistent campaign-STOP veto."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

EXPECTED_MANIFEST_IDS = {'replication': '989629b3691bc2b0a015eff4beee55e5dbb24a287692401acf66f87605bc23d4'}


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
    if not sources or 'scripts/summarize_matched_campaign.py' not in sources:raise ValueError('Analyzer source is not frozen')
    for relative,sha in sources.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe source path')
        if hashlib.sha256((release/relative).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen source mismatch: '+relative)
    spec=importlib.util.spec_from_file_location('verified_frozen_analysis',release/'scripts/summarize_matched_campaign.py')
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    return analyzer,manifests



def apply_stop_veto(audit,stop_path):
    if not Path(stop_path).exists():return audit
    result=dict(audit)
    result["failures"]=list(audit["failures"])+[dict(unit_id="campaign",status="stopped",error="persistent STOP marker requires investigation",stop_marker=str(stop_path))]
    result["failure_count"]=len(result["failures"])
    counts=dict(result["failure_counts_by_status"]);counts["stopped"]=counts.get("stopped",0)+1
    result["failure_counts_by_status"]=counts
    result["inference_status"]="blocked"
    result["comparisons"]=[]
    result.pop("seed_level_metrics",None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("release","results","out"):p.add_argument("--"+n,type=Path,required=True)
    args=p.parse_args()
    analyzer,manifests=verified_analyzer(args.release)
    manifest=manifests['replication']
    result=analyzer.summarize(manifest,args.results)
    stop=args.results/analyzer.manifest_id(manifest)/"STOP.json"
    result=apply_stop_veto(result,stop)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as out:json.dump(result,out,sort_keys=True,indent=2,allow_nan=False);out.write("\n")
    raise SystemExit(2 if result["inference_status"]!="complete" else 0)
if __name__=="__main__":main()
