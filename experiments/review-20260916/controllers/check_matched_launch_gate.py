"""Fresh strict five-unit preflight gate; no scheduler or experiment calls."""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
import importlib.metadata
from pathlib import Path
import sys

PREFLIGHT_ID="5278540b2e92e3fc8ba9558f67580598a6991570fd8809950cc7d5cbfb5f9ef4"
REPLICATION_ID="989629b3691bc2b0a015eff4beee55e5dbb24a287692401acf66f87605bc23d4"
DIAGNOSTIC_ID="fb11da6900ee15d6e6fc9ec266849b9d059451d116f57ab67342a2a626dfc606"


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(m):return hashlib.sha256(json.dumps(m,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)
def load(path):return json.loads(Path(path).read_text())


PRINCIPAL_VERSIONS={"GPy":"1.13.2","numpy":"1.26.4","scipy":"1.12.0","pandas":"1.5.3","emukit":"0.5.1","networkx":"3.4.2","scikit-learn":"1.7.2","statsmodels":"0.13.5"}


class ResourceCapExceeded(ValueError):
    def __init__(self,projection):
        self.projection=projection
        super().__init__("resource phase cap exceeded; stop for resourcing without changing matrix")


def runtime_provenance(versions=None,python_version=None):
    if versions is None:
        try:versions={name:importlib.metadata.version(name) for name in PRINCIPAL_VERSIONS}
        except importlib.metadata.PackageNotFoundError as exc:raise ValueError("principal runtime package missing: "+str(exc)) from exc
    python_version=platform.python_version() if python_version is None else python_version
    need(python_version.split(".")[:2]==["3","10"],"validated runtime requires Python3.10; patch differences allowed")
    need(versions==PRINCIPAL_VERSIONS,"principal runtime package version mismatch: "+json.dumps(versions,sort_keys=True))
    return dict(python=python_version,principal_packages=versions,python_patch_policy="3.10.x accepted; validated HPC3.10.18/local3.10.20",platform=platform.platform())


def resource_projection(rows):
    scalar=[r["wall_seconds"] for r in rows if r["method"] in ("CBO","QCBO","HQCBO","CBO-FALLBACK")]
    ceo=[r["wall_seconds"] for r in rows if r["method"]=="CEO"]
    need(len(scalar)==4 and len(ceo)==1,"resource projection requires all five nominal timings")
    need(all(isinstance(t,(int,float)) and math.isfinite(t) and t>0 for t in scalar+ceo),"all nominal timing values must be positive and finite")
    hours=(630*max(scalar)+90*ceo[0])/3600
    projection=dict(formula="630*max(four scalar nominal seconds)+90*CEO nominal seconds",
                    scalar_units=630,scalar_seconds_used=max(scalar),ceo_units=90,ceo_seconds_used=ceo[0],
                    projected_cpu_hours=hours,phase_cap_cpu_hours=500,
                    queue_limits=dict(array_concurrency=4,cpu_per_unit=1,memory_gb_per_unit=6,wall_hours_per_unit=8,reserved_other_slots=4,global_slot_ceiling=8),
                    interpretation="engineering projection and phase gate only; not a guarantee of actual cost; no methods or seeds may be dropped")
    if hours>500:raise ResourceCapExceeded(projection)
    return projection


def check(release,results,diagnostic_results):
    release=Path(release)
    preflight=load(release/"preflight-manifest.json")
    final=load(release/"replication-manifest.json")
    need(identity(preflight)==PREFLIGHT_ID,"unexpected preflight manifest")
    need(identity(final)==REPLICATION_ID,"unexpected replication manifest")
    need(len(preflight["units"])==5 and len(final["units"])==720,"unexpected unit count")
    need(preflight["source_sha256"]==final["source_sha256"],"preflight/final source mismatch")
    # Verify source before importing/executing the frozen analyzer.
    for name,expected in final["source_sha256"].items():
        rel=Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts,"unsafe source path")
        need((release/rel).is_file() and digest(release/rel)==expected,"frozen source mismatch: "+name)
    spec=importlib.util.spec_from_file_location("frozen_matched_analyzer",release/"scripts/summarize_matched_campaign.py")
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    audit=analyzer.summarize(preflight,Path(results))
    need(audit["expected_units"]==5 and audit["valid_units"]==5 and audit["failure_count"]==0 and
         audit["inference_status"]=="not_applicable_engineering_pilot",
         "all five complete strict preflight units are required; "+json.dumps({k:audit[k] for k in ("valid_units","failure_count","failure_counts_by_status")}))
    # The full final declaration itself must be the frozen24x30 matrix.
    configurations={(u["scm"],u["cond"],u["method"]) for u in final["units"]}
    need(configurations==set(analyzer.STATIC_CONFIGURATIONS),"wrong final configuration matrix")
    need(all(sorted(u["seed"] for u in final["units"] if (u["scm"],u["cond"],u["method"])==g)==analyzer.SEEDS for g in configurations),"wrong final seed matrix")
    hq=next(r for r in audit["engineering_audit"] if r["method"]=="HQCBO")
    split=hq["backend_audit"].get("refinement")
    diagnostic="not_needed_natural_split_verified"
    if not split or not split.get("accepted"):
        m=load(release/"hq-split-diagnostic-manifest.json")
        need(identity(m)==DIAGNOSTIC_ID,"unexpected split diagnostic manifest")
        unit=m["units"][0];folder=Path(diagnostic_results)/DIAGNOSTIC_ID/unit["id"]
        record=load(folder/"execution.json")
        need(record.get("status")=="complete" and record.get("returncode")==0 and record.get("manifest_id")==DIAGNOSTIC_ID and record.get("unit")==unit,"split diagnostic incomplete")
        need(load(folder/"frozen-manifest.json")==m,"split diagnostic manifest mismatch")
        need(set(record.get("output_sha256",{}))==set(unit["outputs"]),"split diagnostic output schema")
        for name in unit["outputs"]:need(digest(folder/name)==record["output_sha256"][name],"split diagnostic output hash mismatch")
        gate=load(folder/"result/gate.json")
        need(gate.get("status")=="passed" and gate.get("tests_run")==1 and gate.get("skipped")==0 and gate.get("failures")==0 and gate.get("errors")==0,"split diagnostic did not pass")
        diagnostic="conditional_forced_fixture_passed"
    runtime=runtime_provenance()
    try:projection=resource_projection(audit["engineering_audit"])
    except ResourceCapExceeded as exc:
        exc.runtime=runtime
        raise
    return dict(status="launch_gate_passed",resource_projection=projection,runtime=runtime,preflight_manifest_id=PREFLIGHT_ID,
                replication_manifest_id=REPLICATION_ID,preflight_valid_units=5,
                final_units=720,split_diagnostic=diagnostic,science_launched=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release",required=True,type=Path)
    p.add_argument("--preflight-results",required=True,type=Path)
    p.add_argument("--diagnostic-results",required=True,type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    def save(record):
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            with args.out.open("x") as out:json.dump(record,out,indent=2,sort_keys=True);out.write("\n")
    try:result=check(args.release,args.preflight_results,args.diagnostic_results)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        record=dict(status="launch_blocked",error=str(exc),science_launched=False)
        if hasattr(exc,"projection"):record["resource_projection"]=exc.projection
        if hasattr(exc,"runtime"):record["runtime"]=exc.runtime
        save(record)
        print(json.dumps(record),file=sys.stderr)
        raise SystemExit(2)
    save(result)
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
