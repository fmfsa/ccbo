"""Independent full-log engineering audit; emits no comparative objective values."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
RELEASE=HERE/"releases/matched-preflight-v3"
RESULTS=HERE/"server/matched-preflight"
REPORT=HERE/"reports/matched-preflight-independent-audit.json"


def key(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(["matched-controlled-noisy-v2",*parts],separators=(",",":")).encode()).digest()[:4],"little")


def row(scm,iv,z):
    if scm=="MediatedChain":
        a=iv.get("X1",.5*z[1]); b=iv.get("X2",2*a+.2*z[2])
        return dict(X1=a,X2=b,Y=(b-4)**2+.1*z[3])
    if scm=="FrontDoor":
        a=iv.get("X1",.4*z[0]+.3*z[1]); b=iv.get("M",2*a+.2*z[2])
        return dict(X1=a,M=b,Y=2*(1-math.exp(-(b-1)**2/.18))+.4*z[0]+.1*z[3])
    raise ValueError(scm)


def main():
    spec=importlib.util.spec_from_file_location("strict_analyzer",RELEASE/"scripts/summarize_matched_campaign.py")
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    manifest=json.loads((RELEASE/"preflight-manifest.json").read_text())
    root=RESULTS/analyzer.manifest_id(manifest)
    reports=[]; event_tables={}; obs_hashes={}; pending=[]
    for unit in manifest["units"]:
        folder=root/unit["id"]
        execution=json.loads((folder/"execution.json").read_text())
        if execution["status"]!="complete":
            pending.append(dict(unit_id=unit["id"],status=execution["status"]));continue
        validated=analyzer.validate_unit(manifest,unit,folder,replication=False)
        events=json.loads((folder/"result/events.json").read_text())
        summary=json.loads((folder/"result/summary.json").read_text())
        obs=json.loads((folder/"result/observations.json").read_text())
        scm=unit["scm"];seed=unit["seed"]
        rng=np.random.RandomState(key(scm,seed,"observational"))
        maxerror=0.
        for actual in obs["rows"]:
            expected=row(scm,{},rng.randn(4))
            maxerror=max(maxerror,max(abs(v-expected[k]) for k,v in zip(obs["columns"],actual)))
        per_phase_arm={};seq=0
        for event in events:
            phase=event["phase"];arm=tuple(event["arm"])
            if phase=="sequential":
                noise_seed=key(scm,seed,"sequential-noise",seq);seq+=1
            else:
                counter=per_phase_arm.get((phase,arm),0);per_phase_arm[phase,arm]=counter+1
                stream="init" if phase=="init" else "split"
                noise_seed=key(scm,seed,stream+"-noise",arm,counter)
                xrng=np.random.RandomState(key(scm,seed,stream+"-levels",arm))
                xs=np.column_stack([xrng.uniform(-2 if scm=="MediatedChain" and v=="X2" else -3,2 if scm=="MediatedChain" and v=="X2" else 3,3) for v in arm])
                assert event["x"]==xs[counter].tolist()
            assert event["noise_seed"]==noise_seed
            expected=row(scm,dict(zip(arm,event["x"])),np.random.RandomState(noise_seed).randn(4))
            maxerror=max(maxerror,max(abs(v-expected[k]) for k,v in event["measured"].items()))
        assert maxerror<1e-12,(unit["id"],maxerror)
        acquisitions=0
        if unit["method"]=="CEO":
            backend=summary["backend"]
            assert backend["feedback_audit"]==dict(measurement_calls=seq,bookkeeping_replays=seq)
            for name,actual in backend["final_gp_y"].items():
                expected=[e["measured"]["Y"] for e in events if "+".join(e["arm"])==name]
                assert actual==expected
            for audit in backend["gp_noise_audit"].values():
                assert not audit["fixed"] and math.isfinite(audit["variance"]) and audit["variance"]>0
        else:
            for audit in summary["backend"]["gp_noise_audit"]:
                assert not audit["fixed"] and not audit["fixed_after_fit"]
                assert 1e-6-1e-12<=audit["final_variance"]<=1e6+1e-6
                for snap in audit.get("acquisition_snapshots",[]):
                    assert audit["fitted_before_acquisition"]
                    assert snap["parameters"]==audit["fitted_parameters"]
                    assert snap["variance"]==audit["fitted_variance"]
                    assert all(math.isfinite(v) for v in snap["parameters"])
                    acquisitions+=1
            assert acquisitions>0
        obs_hashes.setdefault(scm,set()).add(summary["observational_sha256"])
        event_tables[unit["method"]]=events
        reports.append(dict(unit_id=unit["id"],strict_validation="passed",full_measurement_and_observational_replay="passed",
                            maximum_numeric_replay_error=maxerror,acquisition_parameter_snapshots_checked=acquisitions,
                            actual_cost=summary["actual_cost"],initial_cost=summary["init_cost"],split_cost=summary.get("split_init_cost",0),
                            sequential_purchases=summary["sequential_purchases"],wall_seconds=summary["wall_seconds"]))
    assert all(len(h)==1 for h in obs_hashes.values())
    hqfolder=root/"MediatedChain-C0-HQCBO-seed1001"
    hq=json.loads((hqfolder/"result/summary.json").read_text())["backend"]["refinement"]
    assert hq and hq["accepted"]
    trigger=hq["trigger_event_id"]
    assert event_tables["HQCBO"][:trigger+1]==event_tables["QCBO"][:trigger+1]
    initial=[e["measured"]["Y"] for e in event_tables["HQCBO"] if e["phase"]=="init"]
    history=[min(initial)]
    first=None
    for e in event_tables["HQCBO"]:
        if e["phase"]!="sequential":continue
        history.append(min(history[-1],e["measured"]["Y"]))
        if len(history)>=6 and history[-6]-history[-1]<.001:
            first=len(history)-1;break
    assert first==hq["trigger_sequential_index"]==7
    output=dict(scope="engineering consistency only; no comparative objective rankings",completed_units=len(reports),pending=pending,
                completed_unit_audits=reports,shared_observations="passed",natural_HQ_split=dict(trigger_purchase=7,first_plateau_trigger="passed",QCBO_prefix="exact",paid_new_arm_cost=6),
                launch_ready=len(reports)==5 and not pending)
    REPORT.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))
if __name__=="__main__":main()
