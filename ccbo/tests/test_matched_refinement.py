"""Graph-free refinement contracts and actual resumed-CBO integration."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from matched_refinement import RefinementExperiment, run_hqcbo
from ccbo.matched_protocol import run_scalar, score_events
from ccbo import minibench as mb


class RefinementContracts(unittest.TestCase):
    def test_unaffordable_split_is_atomic(self):
        e=RefinementExperiment(mb.MC_NAME,1000,[("X1","X2")],budget=10)
        before=copy.deepcopy(e.events)
        self.assertIsNone(e.expose([("X1",),("X2",)]))
        self.assertEqual(e.events,before)
        self.assertEqual(e.arms,[("X1","X2")])
        self.assertEqual(e.cost,6)

    def test_paid_split_rows_are_immediately_recommendable_and_keyed(self):
        e=RefinementExperiment(mb.MC_NAME,1000,[("X1","X2")],budget=12)
        other=RefinementExperiment(mb.MC_NAME,1000,[("X1","X2")],budget=12)
        before=len(e.events)
        rows=e.expose([("X1",),("X2",)])
        rows2=other.expose([("X2",),("X1",)])
        self.assertEqual(rows,rows2)
        self.assertEqual(e.cost,12)
        self.assertEqual(len(e.events)-before,6)
        self.assertTrue(all(x["phase"]=="split_init" for x in e.events[before:]))
        winner=min(e.events,key=lambda x:(x["measured"]["Y"],tuple(x["arm"]),x["event_id"]))
        self.assertEqual(e.events[-1]["recommendation_event_id"],winner["event_id"])
        self.assertEqual(score_events(e.scm,e.events,e.cost)["final"]["recommendation_event_id"],winner["event_id"])
        # Every new-arm dataset consists only of genuinely purchased rows of that arm.
        for arm,data in rows.items():
            self.assertEqual(data,[x["measured"] for x in e.events if tuple(x["arm"])==arm])


@unittest.skipUnless(os.environ.get("MATCHED_INTEGRATION")=="1","requires actual CBO runtime")
class RefinementBackendTests(unittest.TestCase):
    def test_resumed_coarse_prefix_equals_qcbo(self):
        a,_=run_scalar(mb.MC_NAME,"C0","QCBO",1000,max_purchases=5)
        b,meta=run_hqcbo(mb.MC_NAME,"C0",1000,max_purchases=5)
        self.assertEqual(a.events,b.events)
        self.assertIsNone(meta["refinement"])

    def test_actual_split_has_graph_guard_plain_new_arms_and_full_prefix(self):
        # Controlled trigger fixture for exercising post-split backend, never a run setting.
        a,_=run_scalar(mb.MC_NAME,"C0","QCBO",1000,max_purchases=5)
        with patch("ccbo.benchmark._plateau",side_effect=lambda h,k,d,t:len(h)>=6), patch.object(mb,"population_evaluator",side_effect=AssertionError("oracle used for refinement")):
            b,meta=run_hqcbo(mb.MC_NAME,"C0",1000,max_purchases=7)
        self.assertEqual(a.events,b.events[:len(a.events)])
        split=meta["refinement"]
        self.assertTrue(split["accepted"])
        self.assertEqual(split["trigger_sequential_index"],5)
        self.assertEqual(split["split_cost"],6)
        self.assertEqual(meta["prior_mask"],[True,False,False])
        self.assertEqual(b.sequential_index,7)
        self.assertEqual(b.cost,sum(x["cost"] for x in b.events))
        self.assertLessEqual(b.cost,b.budget)
        splitrows=[e for e in b.events if e["phase"]=="split_init"]
        self.assertEqual(len(splitrows),6)
        self.assertEqual(split["recommendation_after_split"],splitrows[-1]["recommendation_event_id"])
        self.assertTrue(any(a.get("acquisition_snapshots") and len(a["acquisition_snapshots"][0]["arm"])==1 for a in meta["gp_noise_audit"]))

    def test_budget_refuses_split_without_free_information(self):
        with patch("ccbo.benchmark._plateau",side_effect=lambda h,k,d,t:len(h)>=6):
            e,meta=run_hqcbo(mb.MC_NAME,"C0",1000,budget=20)
        self.assertFalse(meta["refinement"]["accepted"])
        self.assertEqual(e.cost,20)
        self.assertTrue(all(x["phase"]!="split_init" for x in e.events))
        self.assertEqual(e.arms,[("X1","X2")])

if __name__=="__main__":unittest.main()
