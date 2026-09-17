#!/usr/bin/env python3
"""Strict, prespecified analysis of frozen matched-control campaigns.

No inference is emitted for pilots or incomplete/invalid replication matrices.
Student t intervals are pointwise, two-sided 95%, paired across 30 seeds.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS = list(range(2000, 2030))
BUDGETS = {"ParallelParent":100, "FrontDoor":100, "MediatedChain":120}
REQUIRED = {"result/"+x+".json" for x in ("config","observations","events","scores","summary")}
# Frozen 24 configurations from the scientific plan; 30 seeds each = 720 units.
STATIC_CONFIGURATIONS = [
    (scm, cond, method)
    for scm, cond in (("ParallelParent","A0"),("ParallelParent","A1"),("FrontDoor","B0"),("FrontDoor","B1"),("MediatedChain","C0"))
    for method in ("CBO","QCBO")
] + [(s,c,"CEO") for s,c in (("ParallelParent","A0"),("FrontDoor","B0"),("MediatedChain","C0"))] + [
    ("MediatedChain","C0","HQCBO"),
    ("ParallelParent","A0","CBO-NP"),("ParallelParent","A0","BO-S"),("ParallelParent","A0","BO"),
    ("ParallelParent","A1","CBO-NP"),
    ("FrontDoor","B0","CBO-NP"),("FrontDoor","B0","BO-S"),("FrontDoor","B0","BO"),("FrontDoor","B0","CBO-FALLBACK"),
    ("MediatedChain","C0","BO-S"),("MediatedChain","C0","BO")
]

# scipy.stats.t.ppf(.975, 29); fixed df because exactly 30 seeds are required.
T975_DF29 = 2.045229642132703


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_id(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",",":"), allow_nan=False).encode()).hexdigest()


def load(path):
    def reject(x):
        raise ValueError("nonfinite JSON constant: "+x)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a,b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)


def population(scm, arm, x):
    """Independent closed forms for the frozen three SCMs, including natural variance."""
    iv = dict(zip(arm,x))
    if scm == "ParallelParent":
        return ((iv["X1"]-2)**2 if "X1" in iv else 4.25) + ((iv["X2"]+2)**2 if "X2" in iv else 4.25)
    if scm == "FrontDoor":
        mu = iv["M"] if "M" in iv else 2*iv.get("X1",0.)
        variance = 0. if "M" in iv else .04 if "X1" in iv else 1.04
        return 2*(1-.3/math.sqrt(.09+variance)*math.exp(-(mu-1)**2/(2*(.09+variance))))
    if scm == "MediatedChain":
        return (iv["X2"]-4)**2 if "X2" in iv else (2*iv.get("X1",0.)-4)**2 + (.04 if "X1" in iv else 1.04)
    raise ValueError("unsupported SCM "+scm)


def validate_events(unit, config, events, scores, summary, replication, nominal_budget=False):
    scm=unit["scm"]; budget=config["budget"]
    require(events and isinstance(events,list), "empty event log")
    cost=0; best=None; init_cost=0; split_cost=0; sequential=0; scored=[]; oracle_best=math.inf
    initial_counts={}; split_counts={}; split_ids=[]; sequential_started=False
    nodes = {"X1","M"} if scm == "FrontDoor" else {"X1","X2"}
    optimum=.04 if scm == "MediatedChain" else 0.
    for index,e in enumerate(events):
        arm=e["arm"]; x=e["x"]
        require(e["event_id"] == index, "event IDs must be consecutive")
        require(arm == sorted(set(arm)) and set(arm) <= nodes and 0 < len(arm)==len(x), "invalid canonical arm")
        require(e["phase"] in ("init","split_init","sequential"), "invalid event phase")
        for v,z in zip(arm,x):
            bound=2 if scm=="MediatedChain" and v=="X2" else 3
            require(math.isfinite(z) and -bound-1e-9 <= z <= bound+1e-9, "invalid intervention domain")
            require(close(e["measured"][v],z), "measured clamped coordinate mismatch")
        require(set(e["measured"]) == nodes|{"Y"} and all(math.isfinite(v) for v in e["measured"].values()), "invalid measured row")
        require(e["cost"] == len(arm), "purchase cost != arm size")
        cost += len(arm)
        require(e["cum_cost"] == cost and cost <= budget, "cumulative cost mismatch or overshoot")
        if e["phase"]=="init":
            require(not sequential_started, "initialization after sequential purchase")
            init_cost+=len(arm); initial_counts[tuple(arm)]=initial_counts.get(tuple(arm),0)+1
        elif e["phase"]=="split_init":
            require(unit["method"]=="HQCBO" and scm=="MediatedChain" and unit["cond"]=="C0",
                    "split initialization outside declared refinement method")
            require(sequential_started and tuple(arm) not in initial_counts,"split arm was already initialized")
            require(len(arm)==1,"refinement exposes singleton arms only")
            split_cost+=len(arm); split_counts[tuple(arm)]=split_counts.get(tuple(arm),0)+1; split_ids.append(index)
        else:
            sequential_started=True; sequential+=1
            require(initial_counts.get(tuple(arm),split_counts.get(tuple(arm),0))==config["n_init"],
                    "sequential arm lacks complete purchased initial design")
        key=lambda r:(r["measured"]["Y"],tuple(r["arm"]),r["event_id"])
        if best is None or key(e)<key(best): best=e
        require(e.get("recommendation_event_id") == best["event_id"], "recommendation is not measured-data winner")
        value=population(scm,best["arm"],best["x"])
        oracle_best=min(oracle_best,population(scm,arm,x))
        scored.append(dict(event_id=index,cum_cost=cost,recommendation_event_id=best["event_id"],
                           recommendation_population=value,recommendation_regret=value-optimum,
                           oracle_best_visited=oracle_best))
    require(all(n==config["n_init"] for n in initial_counts.values()), "initial count does not match protocol")
    require(summary.get("split_init_cost",0)==split_cost,"split initialization cost mismatch")
    if split_ids:
        require(split_ids==list(range(split_ids[0],split_ids[-1]+1)) and split_cost==6 and
                split_counts=={("X1",):3,("X2",):3},"split design is not the complete contiguous atomic design")
    refinement=summary.get("backend",{}).get("refinement")
    if unit["method"]=="HQCBO" and refinement is not None:
        trigger=refinement["trigger_event_id"]
        require(0<=trigger<len(events) and events[trigger]["phase"]=="sequential","invalid split trigger")
        require(refinement["trigger_sequential_index"]==sum(e["phase"]=="sequential" for e in events[:trigger+1]),"split trigger index mismatch")
        if refinement["accepted"]:
            require(bool(split_ids) and split_ids[0]==trigger+1 and refinement["split_event_ids"]==split_ids and
                    refinement["split_cost"]==6 and sorted(refinement["split_arms"])==[["X1"],["X2"]],"accepted split metadata mismatch")
            require(refinement["recommendation_after_split"]==events[split_ids[-1]]["recommendation_event_id"],"post-split recommendation mismatch")
        else:
            require(not split_ids and refinement["split_cost"]==0 and budget-events[trigger]["cum_cost"]<6,
                    "declined split must be unaffordable and uncharged")
    else:
        require(not split_ids,"split events missing refinement metadata")
    require(summary["actual_cost"]==cost and summary["init_cost"]==init_cost and summary["n_purchases"]==len(events)
            and summary["sequential_purchases"]==sequential, "summary ledger counts mismatch")
    if replication: require(cost==budget, "replication has not reached common budget B")
    elif nominal_budget:
        cheapest=min(len(arm) for arm in set(initial_counts)|set(split_counts))
        require(0 <= budget-cost < cheapest, "nominal-budget pilot still has affordable actions")
    else: require(sequential==config["max_purchases"], "pilot purchase count mismatch")
    def same_row(a,b):
        require(set(a)==set(b), "score row schema mismatch")
        for k,v in b.items(): require(close(a[k],v), "score/recommendation mismatch: "+k)
    require(len(scores["scored_events"])==len(scored), "score event count mismatch")
    for a,b in zip(scores["scored_events"],scored): same_row(a,b)
    same_row(scores["final"],scored[-1]); same_row(summary["final"],scored[-1])
    checkpoints=[]
    for c in range(12,cost+1):
        row=next(r for r in reversed(scored) if r["cum_cost"]<=c)
        checkpoints.append(dict(cost=c,**{k:v for k,v in row.items() if k!="cum_cost"}))
    require(len(scores["checkpoints"])==len(checkpoints), "cost checkpoints mismatch")
    for a,b in zip(scores["checkpoints"],checkpoints): same_row(a,b)
    area=sum(r["recommendation_regret"] for r in checkpoints)
    require(close(scores["cost_integrated_recommendation_regret"],area) and
            close(summary["cost_integrated_recommendation_regret"],area), "cost area mismatch")
    return dict(final_recommendation_regret=scored[-1]["recommendation_regret"],
                cost_integrated_recommendation_regret=area, actual_cost=cost,
                purchases=len(events), sequential_purchases=sequential)


def options(unit):
    argv=unit["argv"]
    require(len(argv)>=2 and argv[1]=="scripts/run_matched_controlled.py", "unexpected experiment command")
    require(len(argv[2:])%2==0, "malformed CLI options")
    result={}
    for flag,value in zip(argv[2::2],argv[3::2]):
        require(flag.startswith("--") and flag not in result,"duplicate/malformed option")
        result[flag]=value
    return result


def validate_unit(m, unit, folder, replication):
    nominal_budget=not replication and m["protocol"].get("pilot_mode")=="nominal_budget"
    execution=load(folder/"execution.json")
    require(execution.get("status")=="complete" and execution.get("returncode")==0,"execution not complete")
    require(execution.get("unit")==unit and execution.get("protocol")==m["protocol"] and
            execution.get("manifest_id")==manifest_id(m), "execution identity mismatch")
    require(load(folder/"frozen-manifest.json")==m,"unit frozen manifest mismatch")
    require(set(unit["outputs"])==REQUIRED,"unexpected declared output schema")
    require(set(execution.get("output_sha256",{}))==REQUIRED,"execution output hash set mismatch")
    for rel in REQUIRED:
        require(digest(folder/rel)==execution["output_sha256"][rel],"output hash mismatch: "+rel)
    config=load(folder/"result/config.json"); summary=load(folder/"result/summary.json")
    args=options(unit)
    for key in ("scm","cond","method","seed"):
        require(config.get(key)==unit[key] and summary.get(key)==unit[key], "unit/config identity mismatch: "+key)
        require(str(unit[key])==args["--"+key], "manifest CLI identity mismatch: "+key)
    require(config.get("protocol_id")==m["protocol"]["family"], "protocol family mismatch")
    require(config.get("stage")==("replication" if replication else "pilot"),"config stage mismatch")
    require(args.get("--stage")==config["stage"],"CLI stage mismatch")
    require(config.get("budget")==m["protocol"]["budget"][unit["scm"]]==BUDGETS[unit["scm"]],"budget mismatch")
    require(config.get("n_obs")==100 and config.get("n_init")==3,"initial information mismatch")
    require(config.get("feedback_mode")=="single_true_SCM_draw", "feedback mode mismatch")
    require(config.get("recommendation_rule")=="minimum_measured_Y;ties_canonical_arm_then_execution_index", "recommendation rule mismatch")
    if replication or nominal_budget:
        require(config.get("max_purchases") is None and "--max-purchases" not in args,"capped nominal-budget run")
    else:
        require(config.get("max_purchases")==m["protocol"]["sequential_purchases"]==int(args["--max-purchases"]), "pilot cap mismatch")
    require(summary.get("status")==("budget_complete" if replication or nominal_budget else "pilot_complete"),"run status mismatch")
    for k,v in config.items(): require(summary.get(k)==v,"summary/config mismatch: "+k)
    source=config.get("source_sha256",{})
    require(source and all(m["source_sha256"].get(k)==v for k,v in source.items()),"recorded source hash mismatch")
    if unit["method"]=="CEO":
        prefix="third_party/CEO/"
        expected_ceo={k[len(prefix):]:v for k,v in m["source_sha256"].items() if k.startswith(prefix+"src/") and k.endswith(".py")}
        require(expected_ceo and config.get("ceo_source_sha256")==expected_ceo,"pinned CEO source provenance mismatch")
    require(digest(folder/"result/observations.json")==config["observational_sha256"],"observation hash mismatch")
    observations=load(folder/"result/observations.json")
    columns=["X1","M","Y"] if unit["scm"]=="FrontDoor" else ["X1","X2","Y"]
    require(observations["columns"]==columns and len(observations["rows"])==100 and
            all(len(row)==3 and all(math.isfinite(x) for x in row) for row in observations["rows"]),"observation matrix schema mismatch")
    require(digest(folder/"result/events.json")==summary["events_sha256"],"event hash mismatch")
    events=load(folder/"result/events.json")
    ledger=validate_events(unit,config,events,load(folder/"result/scores.json"),summary,replication,nominal_budget=nominal_budget)
    require(math.isfinite(summary["wall_seconds"]) and summary["wall_seconds"]>=0,"invalid runtime")
    return dict(unit_id=unit["id"],scm=unit["scm"],cond=unit["cond"],method=unit["method"],seed=unit["seed"],
                wall_seconds=summary["wall_seconds"],observational_sha256=config["observational_sha256"],
                _initial_rows=[{k:e[k] for k in ("arm","x","measured","noise_seed")} for e in events if e["phase"]=="init"],
                backend_audit=summary["backend"],**ledger)


def paired_interval(rows, metric):
    require(len(rows)==30 and [r["seed"] for r in rows]==SEEDS,"paired sample must contain all 30 frozen seeds")
    differences=[r["QCBO"]-r["CBO"] for r in rows]
    mean=statistics.mean(differences); se=statistics.stdev(differences)/math.sqrt(30)
    return dict(metric=metric, contrast="QCBO minus CBO",n=30,mean_paired_difference=mean,
                confidence_level=.95,interval_kind="pointwise two-sided Student t; df=29; no multiplicity adjustment",
                ci95=[mean-T975_DF29*se,mean+T975_DF29*se],
                seed_values=[dict(r,difference=d) for r,d in zip(rows,differences)])


def summarize(m, results):
    require(m.get("schema")==1 and m.get("source_sha256"),"manifest must freeze schema and source")
    require(m["protocol"]["family"].startswith("matched-controlled-noisy-v"),"unexpected family")
    stage=m["protocol"]["stage"]
    require(stage in ("engineering-pilot","replication"),"unknown campaign stage")
    replication=stage=="replication"; units=m["units"]
    keys=[(u["scm"],u["cond"],u["method"],u["seed"]) for u in units]
    require(units and len(set(keys))==len(keys) and len({u["id"] for u in units})==len(units),"duplicate/empty matrix")
    groups={key[:3] for key in keys}
    if replication:
        require(m["protocol"].get("replication_seeds")==SEEDS,"replication seed declaration mismatch")
        for group in groups: require(sorted(k[3] for k in keys if k[:3]==group)==SEEDS,"incomplete declared seed matrix")
        expected=set(STATIC_CONFIGURATIONS)
        require(len(STATIC_CONFIGURATIONS)==len(expected),"internal duplicate expected configurations")
        require(groups==expected,"incomplete or unexpected static configuration matrix")
        require(m["protocol"].get("expected_configurations")==[list(x) for x in STATIC_CONFIGURATIONS],"expected configuration declaration mismatch")
        require(m["protocol"].get("expected_units")==len(expected)*len(SEEDS)==len(units),"expected unit count mismatch")
    else:
        require(all(k[3] in range(1000,1005) for k in keys),"pilot seed outside engineering set")
        require(m["protocol"].get("pilot_mode","capped") in ("capped","nominal_budget"),"unknown engineering pilot mode")
    campaign=Path(results)/manifest_id(m)
    extras=sorted(p.name for p in campaign.iterdir() if p.is_dir() and p.name not in {u["id"] for u in units}) if campaign.exists() else []
    failures=[dict(unit_id=x,status="unexpected",error="unit absent from frozen matrix") for x in extras]
    rows=[]
    for unit in units:
        try:
            require(unit["id"] and all(c.isalnum() or c in "_-" for c in unit["id"]),"invalid unit ID")
            rows.append(validate_unit(m,unit,campaign/unit["id"],replication))
        except (ValueError,KeyError,TypeError,FileNotFoundError,StopIteration) as exc:
            status="missing" if not (campaign/unit["id"]/"execution.json").exists() else "invalid"
            if status!="missing":
                try:
                    execution=load(campaign/unit["id"]/"execution.json")
                    if execution.get("status") in ("failed","running"): status=execution["status"]
                except (ValueError,OSError): pass
            failures.append(dict(unit_id=unit["id"],status=status,error=str(exc),
                                 **{k:unit[k] for k in ("scm","cond","method","seed")}))
    # Cross-method/condition observational pairing must actually hold.
    for scm,seed in {(r["scm"],r["seed"]) for r in rows}:
        hashes={r["observational_sha256"] for r in rows if (r["scm"],r["seed"])==(scm,seed)}
        if len(hashes)!=1: failures.append(dict(unit_id=f"{scm}-seed{seed}",status="invalid",error="unpaired observational datasets"))
        arm_tables={}
        for r in rows:
            if (r["scm"],r["seed"])!=(scm,seed): continue
            for arm in {tuple(e["arm"]) for e in r["_initial_rows"]}:
                table=[e for e in r["_initial_rows"] if tuple(e["arm"])==arm]
                if arm in arm_tables and arm_tables[arm]!=table:
                    failures.append(dict(unit_id=r["unit_id"],status="invalid",error="shared-arm initial data mismatch"))
                else: arm_tables[arm]=table
    for r in rows: r.pop("_initial_rows")
    result=dict(schema=1,manifest_id=manifest_id(m),stage=stage,
                analysis_source_sha256=digest(__file__),frozen_source_sha256=m["source_sha256"],
                expected_units=len(units),valid_units=len(rows),failure_count=len(failures),failures=failures,
                failure_counts_by_status={s:sum(f["status"]==s for f in failures) for s in sorted({f["status"] for f in failures})},
                inference_status="blocked" if failures else "not_applicable_engineering_pilot" if not replication else "complete",
                primary="final data-selected recommendation regret at B",
                secondary="sum of right-continuous recommendation regret at integer costs 12..B",
                comparisons=[])
    result["engineering_audit"]=[{k:v for k,v in r.items() if k not in ("final_recommendation_regret","cost_integrated_recommendation_regret")} for r in rows]
    if failures or not replication: return result
    result["seed_level_metrics"]=rows
    for scm,cond in sorted({(r["scm"],r["cond"]) for r in rows}):
        group=[r for r in rows if (r["scm"],r["cond"])==(scm,cond)]
        if not {"CBO","QCBO"}<={r["method"] for r in group}: continue
        metrics=[]
        for metric in ("final_recommendation_regret","cost_integrated_recommendation_regret"):
            values=[dict(seed=s,**{method:next(r[metric] for r in group if r["method"]==method and r["seed"]==s)
                                       for method in ("CBO","QCBO")}) for s in SEEDS]
            metrics.append(paired_interval(values,metric))
        result["comparisons"].append(dict(scm=scm,cond=cond,metrics=metrics))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True,type=Path)
    p.add_argument("--results",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    result=summarize(load(a.manifest),a.results)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if result["inference_status"]=="blocked": raise SystemExit(2)

if __name__=="__main__": main()
