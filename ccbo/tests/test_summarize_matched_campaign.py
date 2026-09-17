"""Standard-library tests of strict campaign analysis; no GP stack required."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[2]/"scripts/summarize_matched_campaign.py"
spec=importlib.util.spec_from_file_location("matched_summary",SCRIPT)
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def manifest(replication=True):
    seeds=a.SEEDS if replication else [1000]
    stage="replication" if replication else "pilot"
    units=[]
    for method in ("CBO","QCBO"):
        for seed in seeds:
            unit=dict(id=f"PP-A0-{method}-{seed}",scm="ParallelParent",cond="A0",method=method,seed=seed,
                      argv=["{python}","scripts/run_matched_controlled.py","--scm","ParallelParent","--cond","A0",
                            "--method",method,"--seed",str(seed),"--stage",stage], outputs=sorted(a.REQUIRED))
            if not replication: unit["argv"] += ["--max-purchases","3"]
            unit["argv"] += ["--outdir","{unit_dir}/result"]
            units.append(unit)
    return dict(schema=1,protocol=dict(family="matched-controlled-noisy-v2",stage="replication" if replication else "engineering-pilot",
                   replication_seeds=a.SEEDS,budget=a.BUDGETS,sequential_purchases=3,
                   expected_configurations=[list(x) for x in a.STATIC_CONFIGURATIONS],expected_units=30*len(a.STATIC_CONFIGURATIONS)),
                source_sha256={"ccbo/matched_protocol.py":"a"*64},units=units)


def create_results(root,m):
    replication=m["protocol"]["stage"]=="replication"
    full_budget=replication or m["protocol"].get("pilot_mode")=="nominal_budget"
    for unit in m["units"]:
        folder=root/a.manifest_id(m)/unit["id"]
        write(folder/"frozen-manifest.json",m)
        write(folder/"result/observations.json",{"columns":["X1","X2","Y"],"rows":[[0,0,8]]*100})
        config={k:unit[k] for k in ("scm","cond","method","seed")}
        config.update(protocol_id=m["protocol"]["family"],stage="replication" if replication else "pilot",
                      budget=100,n_obs=100,n_init=3,max_purchases=None if full_budget else 3,
                      feedback_mode="single_true_SCM_draw",recommendation_rule="minimum_measured_Y;ties_canonical_arm_then_execution_index",
                      source_sha256=m["source_sha256"],observational_sha256=a.digest(folder/"result/observations.json"))
        events=[]; scored=[]
        count=50 if full_budget else 6
        for i in range(count):
            x=[0.,0.] if i<3 or unit["method"]=="CBO" else [1.,-1.]
            e=dict(event_id=i,phase="init" if i<3 else "sequential",arm=["X1","X2"],x=x,
                   measured=dict(X1=x[0],X2=x[1],Y=10. if i<3 else -1.),cost=2,cum_cost=2*(i+1),noise_seed=i,
                   recommendation_event_id=0 if i<3 else 3)
            events.append(e)
            v=a.population("ParallelParent",e["arm"],x)
            scored.append(dict(event_id=i,cum_cost=e["cum_cost"],recommendation_event_id=e["recommendation_event_id"],
                               recommendation_population=v,recommendation_regret=v,oracle_best_visited=v))
        checkpoints=[]
        for c in range(12,2*count+1):
            s=scored[c//2-1]
            checkpoints.append(dict(cost=c,**{k:v for k,v in s.items() if k!="cum_cost"}))
        area=sum(r["recommendation_regret"] for r in checkpoints)
        scores=dict(final=scored[-1],scored_events=scored,checkpoints=checkpoints,cost_integrated_recommendation_regret=area)
        write(folder/"result/config.json",config); write(folder/"result/events.json",events); write(folder/"result/scores.json",scores)
        summary=dict(config,status="budget_complete" if full_budget else "pilot_complete",actual_cost=2*count,init_cost=6,n_purchases=count,
                     sequential_purchases=count-3,final=scored[-1],cost_integrated_recommendation_regret=area,wall_seconds=1.,
                     backend={"noise_policy":"fixture"},events_sha256=a.digest(folder/"result/events.json"))
        write(folder/"result/summary.json",summary)
        execution=dict(schema=1,manifest_id=a.manifest_id(m),protocol=m["protocol"],unit=unit,status="complete",returncode=0,
                       output_sha256={rel:a.digest(folder/rel) for rel in a.REQUIRED})
        write(folder/"execution.json",execution)


def resign(folder):
    execution=a.load(folder/"execution.json")
    execution["output_sha256"]={rel:a.digest(folder/rel) for rel in a.REQUIRED}
    write(folder/"execution.json",execution)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.matrix_patch=patch.object(a,"STATIC_CONFIGURATIONS",[("ParallelParent","A0","CBO"),("ParallelParent","A0","QCBO")])
        self.matrix_patch.start()
        self.addCleanup(self.matrix_patch.stop)
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_complete_replication_pairs_direction_and_cost_area(self):
        m=manifest(); create_results(self.root,m)
        result=a.summarize(m,self.root)
        self.assertEqual(result["inference_status"],"complete")
        metrics=result["comparisons"][0]["metrics"]
        self.assertEqual(metrics[0]["mean_paired_difference"],-6)
        self.assertEqual(metrics[0]["ci95"],[-6,-6])
        self.assertEqual(metrics[1]["mean_paired_difference"],-6*89)
        self.assertEqual(len(metrics[0]["seed_values"]),30)

    def test_pilot_emits_no_performance_metrics_or_inference(self):
        m=manifest(False); create_results(self.root,m)
        r=a.summarize(m,self.root)
        self.assertEqual(r["inference_status"],"not_applicable_engineering_pilot")
        self.assertEqual(r["comparisons"],[])
        self.assertNotIn("seed_level_metrics",r)
        self.assertNotIn("final_recommendation_regret",r["engineering_audit"][0])

    def test_missing_unit_blocks_inference_and_counts_failure(self):
        m=manifest(); create_results(self.root,m)
        folder=self.root/a.manifest_id(m)/m["units"][0]["id"]
        (folder/"execution.json").unlink()
        r=a.summarize(m,self.root)
        self.assertEqual(r["inference_status"],"blocked"); self.assertEqual(r["failure_count"],1)
        self.assertEqual(r["comparisons"],[])

    def test_output_tampering_is_rejected(self):
        m=manifest(False); create_results(self.root,m)
        f=self.root/a.manifest_id(m)/m["units"][0]["id"]/"result/scores.json"
        f.write_text(f.read_text()+" ")
        r=a.summarize(m,self.root)
        self.assertIn("hash mismatch",r["failures"][0]["error"])

    def test_rehashed_false_recommendation_and_cost_cannot_pass(self):
        for mode in ("recommendation","cost"):
            root=self.root/mode; m=manifest(False); create_results(root,m)
            folder=root/a.manifest_id(m)/m["units"][0]["id"]
            events=a.load(folder/"result/events.json")
            if mode=="recommendation": events[-1]["recommendation_event_id"]=0
            else: events[-1]["cum_cost"]+=1
            write(folder/"result/events.json",events)
            s=a.load(folder/"result/summary.json"); s["events_sha256"]=a.digest(folder/"result/events.json")
            write(folder/"result/summary.json",s); resign(folder)
            r=a.summarize(m,root)
            self.assertEqual(r["inference_status"],"blocked")

    def test_incomplete_declared_matrix_rejected(self):
        m=manifest(); m["units"].pop()
        with self.assertRaisesRegex(ValueError,"seed matrix"): a.summarize(m,self.root)

    def test_entire_missing_configuration_rejected_despite_all_remaining_seeds(self):
        m=manifest(); m["units"]=[u for u in m["units"] if u["method"]=="CBO"]
        with self.assertRaisesRegex(ValueError,"configuration matrix"): a.summarize(m,self.root)

    def test_false_expected_count_and_configuration_declaration_rejected(self):
        m=manifest(); m["protocol"]["expected_units"]-=1
        with self.assertRaisesRegex(ValueError,"unit count"): a.summarize(m,self.root)
        m=manifest(); m["protocol"]["expected_configurations"].pop()
        with self.assertRaisesRegex(ValueError,"configuration declaration"): a.summarize(m,self.root)

    def test_nominal_budget_pilot_requires_full_budget_but_never_infers(self):
        m=manifest(False); m["protocol"]["pilot_mode"]="nominal_budget"
        for unit in m["units"]:
            i=unit["argv"].index("--max-purchases"); del unit["argv"][i:i+2]
        create_results(self.root,m)
        result=a.summarize(m,self.root)
        self.assertEqual(result["inference_status"],"not_applicable_engineering_pilot")
        self.assertEqual(result["comparisons"],[])
        self.assertEqual(result["valid_units"],2)
        # A capped pilot cannot be disguised merely by changing the manifest mode.
        capped=manifest(False); capped["protocol"]["pilot_mode"]="nominal_budget"
        create_results(self.root,capped)
        rejected=a.summarize(capped,self.root)
        self.assertEqual(rejected["inference_status"],"blocked")
        self.assertTrue(any("capped" in x["error"] for x in rejected["failures"]))

    def test_nominal_budget_stops_only_without_affordable_action(self):
        fixture=list(self.refinement_fixture(accepted=False,stop_cost=118))
        # Coarse-only family costs2: remainder2 remains affordable and must fail.
        with self.assertRaisesRegex(ValueError,"affordable"):
            a.validate_events(*fixture,replication=False,nominal_budget=True)
        fixture[1]["budget"]=119  # Synthetic odd-budget ledger gate only.
        a.validate_events(*fixture,replication=False,nominal_budget=True)

    def test_student_t_not_normal_and_sign_retained(self):
        rows=[dict(seed=s,CBO=0.,QCBO=float(s-2000)) for s in a.SEEDS]
        r=a.paired_interval(rows,"metric")
        half=2.045229642132703*math.sqrt(77.5/30)
        self.assertAlmostEqual(r["mean_paired_difference"],14.5)
        self.assertAlmostEqual(r["ci95"][0],14.5-half)
        self.assertAlmostEqual(r["ci95"][1],14.5+half)

    def test_false_oracle_score_even_rehashed_is_rejected(self):
        m=manifest(False); create_results(self.root,m)
        folder=self.root/a.manifest_id(m)/m["units"][0]["id"]
        scores=a.load(folder/"result/scores.json"); scores["scored_events"][-1]["recommendation_regret"]=-100.
        write(folder/"result/scores.json",scores); resign(folder)
        r=a.summarize(m,self.root)
        self.assertEqual(r["inference_status"],"blocked")
        self.assertIn("score/recommendation mismatch",r["failures"][0]["error"])

    def test_data_selected_regret_may_worsen_and_positive_effect_is_reported(self):
        m=manifest(); create_results(self.root,m)
        for unit in m["units"]:
            if unit["method"]!="QCBO": continue
            folder=self.root/a.manifest_id(m)/unit["id"]
            events=a.load(folder/"result/events.json")
            for e in events[3:]:
                e["x"]=[-3.,3.]; e["measured"].update(X1=-3.,X2=3.)
            write(folder/"result/events.json",events)
            scores=a.load(folder/"result/scores.json")
            for row in scores["scored_events"][3:]+scores["checkpoints"]+[scores["final"]]:
                row.update(recommendation_population=50.,recommendation_regret=50.,oracle_best_visited=8.)
            scores["cost_integrated_recommendation_regret"]=50.*89
            write(folder/"result/scores.json",scores)
            summary=a.load(folder/"result/summary.json")
            summary.update(final=scores["final"],cost_integrated_recommendation_regret=50.*89,
                           events_sha256=a.digest(folder/"result/events.json"))
            write(folder/"result/summary.json",summary); resign(folder)
        r=a.summarize(m,self.root)
        self.assertEqual(r["inference_status"],"complete")
        self.assertEqual(r["comparisons"][0]["metrics"][0]["mean_paired_difference"],42.)

    def test_rehashed_config_source_mismatch_blocks_analysis(self):
        m=manifest(False); create_results(self.root,m)
        folder=self.root/a.manifest_id(m)/m["units"][0]["id"]
        for rel in ("result/config.json","result/summary.json"):
            obj=a.load(folder/rel); obj["source_sha256"]["ccbo/matched_protocol.py"]="b"*64
            write(folder/rel,obj)
        resign(folder)
        r=a.summarize(m,self.root)
        self.assertEqual(r["inference_status"],"blocked")
        self.assertIn("source hash mismatch",r["failures"][0]["error"])

    def test_failed_execution_blocks_even_with_complete_outputs(self):
        m=manifest(False); create_results(self.root,m)
        folder=self.root/a.manifest_id(m)/m["units"][0]["id"]
        status=a.load(folder/"execution.json"); status.update(status="failed",returncode=1)
        write(folder/"execution.json",status)
        r=a.summarize(m,self.root)
        self.assertEqual(r["inference_status"],"blocked")
        self.assertEqual(r["failure_counts_by_status"],{"failed":1})

    def refinement_fixture(self, accepted=True, stop_cost=120):
        events=[]; cost=0; best=None; rows=[]; oracle_best=math.inf
        def add(phase,arm,x,y):
            nonlocal cost,best,oracle_best
            cost+=len(arm)
            measured={"X1":0.,"X2":0.,"Y":y}; measured.update(zip(arm,x))
            e=dict(event_id=len(events),phase=phase,arm=arm,x=x,measured=measured,cost=len(arm),cum_cost=cost)
            events.append(e)
            best=min(events,key=lambda r:(r["measured"]["Y"],tuple(r["arm"]),r["event_id"]))
            e["recommendation_event_id"]=best["event_id"]
            value=a.population("MediatedChain",best["arm"],best["x"])
            oracle_best=min(oracle_best,a.population("MediatedChain",arm,x))
            rows.append(dict(event_id=e["event_id"],cum_cost=cost,recommendation_event_id=best["event_id"],
                             recommendation_population=value,recommendation_regret=value-.04,oracle_best_visited=oracle_best))
        for _ in range(3): add("init",["X1","X2"],[0.,0.],10.)
        if accepted:
            add("sequential",["X1","X2"],[0.,0.],10.)
            for _ in range(3): add("split_init",["X1"],[2.],-2.)
            for _ in range(3): add("split_init",["X2"],[0.],10.)
            refinement=dict(accepted=True,trigger_event_id=3,trigger_sequential_index=1,split_event_ids=list(range(4,10)),
                            split_cost=6,split_arms=[["X1"],["X2"]],recommendation_after_split=4)
            while cost<stop_cost: add("sequential",["X1"],[0.],1.)
        else:
            while cost<stop_cost: add("sequential",["X1","X2"],[0.,0.],10.)
            refinement=dict(accepted=False,trigger_event_id=57,trigger_sequential_index=55,split_cost=0)
        checkpoints=[]
        for c in range(12,cost+1):
            row=next(r for r in reversed(rows) if r["cum_cost"]<=c)
            checkpoints.append(dict(cost=c,**{k:v for k,v in row.items() if k!="cum_cost"}))
        area=sum(r["recommendation_regret"] for r in checkpoints)
        scores=dict(final=rows[-1],scored_events=rows,checkpoints=checkpoints,cost_integrated_recommendation_regret=area)
        summary=dict(actual_cost=cost,init_cost=6,split_init_cost=6 if accepted else 0,n_purchases=len(events),
                     sequential_purchases=sum(e["phase"]=="sequential" for e in events),final=rows[-1],
                     cost_integrated_recommendation_regret=area,backend=dict(refinement=refinement))
        return dict(scm="MediatedChain",cond="C0",method="HQCBO"),dict(budget=120,n_init=3),events,scores,summary

    def test_split_purchase_immediately_updates_recommendation_and_area(self):
        fixture=self.refinement_fixture()
        result=a.validate_events(*fixture,replication=True)
        self.assertEqual(fixture[3]["scored_events"][4]["cum_cost"],9)
        self.assertAlmostEqual(fixture[3]["scored_events"][4]["recommendation_regret"],0.)
        self.assertAlmostEqual(result["cost_integrated_recommendation_regret"],0.)
        self.assertEqual(result["actual_cost"],120)
        fixture[2][5]["recommendation_event_id"]=0
        with self.assertRaisesRegex(ValueError,"recommendation"): a.validate_events(*fixture,replication=True)

    def test_declined_atomic_split_does_not_excuse_incomplete_budget(self):
        a.validate_events(*self.refinement_fixture(accepted=False),replication=True)
        with self.assertRaisesRegex(ValueError,"common budget"):
            a.validate_events(*self.refinement_fixture(accepted=False,stop_cost=118),replication=True)

if __name__=="__main__": unittest.main()
