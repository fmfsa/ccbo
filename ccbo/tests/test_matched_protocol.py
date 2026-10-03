"""Pure protocol contracts; optional real-backend tests are explicitly opt-in.

Run: python -m unittest ccbo.tests.test_matched_protocol -v
Real pilot integration (requires GPy/Emukit): MATCHED_INTEGRATION=1 ...
"""
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from ccbo import minibench as mb
from ccbo.matched_protocol import (Experiment, BudgetExhausted, PilotComplete,
    all_arms, sample_true, score_events, observational_data, keyed_seed, run_scalar)


class MatchedProtocolTests(unittest.TestCase):
    def test_shared_init_survives_arm_reordering_removal(self):
        for scm in (mb.PP_NAME, mb.FD_NAME, mb.MC_NAME):
            arms = all_arms(scm)
            a = Experiment(scm, 1000, arms)
            b = Experiment(scm, 1000, list(reversed(arms)))
            c = Experiment(scm, 1000, [arms[1]])
            self.assertEqual(a.initial, b.initial)
            self.assertEqual(a.initial[arms[1]], c.initial[arms[1]])

    def test_fresh_observational_dataset_and_stream_isolation(self):
        a = observational_data(mb.FD_NAME, 1000)
        np.random.randn(1000)
        b = observational_data(mb.FD_NAME, 1000)
        self.assertTrue(a.equals(b))
        self.assertEqual(a.shape, (100,3))
        self.assertFalse(a.equals(observational_data(mb.FD_NAME,1001)))
        self.assertNotEqual(keyed_seed("a","observational"), keyed_seed("a","init-noise"))

    def test_measurement_does_not_call_population_oracle(self):
        with patch.object(mb, "population_evaluator", side_effect=AssertionError("oracle reached learner")):
            a = Experiment(mb.PP_NAME,1000,all_arms(mb.PP_NAME))
            a.purchase(("X1",),[.2])
        self.assertEqual(len(a.events),10)

    def test_budget_and_pilot_cap_are_distinct(self):
        e=Experiment(mb.PP_NAME,1000,all_arms(mb.PP_NAME),budget=3)
        self.assertEqual(e.cost,0)                      # initial points are given, not charged
        self.assertEqual(sum(x["phase"]=="init" for x in e.events),9)
        e.purchase(("X1","X2"),[0,0])
        self.assertEqual(e.cost,2)
        self.assertFalse(e.affordable(("X1","X2")))
        with self.assertRaises(BudgetExhausted): e.purchase(("X1","X2"),[0,0])
        self.assertEqual(e.cost,2)
        e.purchase(("X1",),[0])
        with self.assertRaises(BudgetExhausted): e.purchase(("X1",),[0])
        p=Experiment(mb.PP_NAME,1000,all_arms(mb.PP_NAME),max_purchases=1)
        p.purchase(("X1",),[0])
        with self.assertRaises(PilotComplete): p.purchase(("X1",),[0])
        self.assertGreater(p.remaining,0)

    def test_adversarial_measured_recommendation_not_oracle_cummin(self):
        events=[dict(event_id=0,phase="init",arm=["X1","X2"],x=[2,-2],
                     measured={"Y":1.},cost=2,cum_cost=2),
                dict(event_id=1,phase="sequential",arm=["X1","X2"],x=[0,0],
                     measured={"Y":-1.},cost=2,cum_cost=4)]
        original=copy.deepcopy(events)
        scores=score_events(mb.PP_NAME,events,4)
        self.assertEqual(scores["scored_events"][0]["recommendation_population"],0)
        self.assertEqual(scores["final"]["recommendation_population"],8)
        self.assertEqual(scores["final"]["oracle_best_visited"],0)
        self.assertEqual(scores["final"]["recommendation_event_id"],1)
        self.assertEqual(events,original)

    def test_exact_scoring_cannot_advance_measurements(self):
        a=Experiment(mb.MC_NAME,1000,all_arms(mb.MC_NAME))
        b=Experiment(mb.MC_NAME,1000,all_arms(mb.MC_NAME))
        score_events(a.scm,a.events,a.budget)
        with patch.object(mb,"population_evaluator",return_value=lambda arm,x:999.):
            score_events(a.scm,a.events,a.budget)
        self.assertEqual(a.purchase(("X1",),[2]),b.purchase(("X1",),[2]))

    def test_common_noise_and_clamps_all_arms(self):
        for scm in (mb.PP_NAME,mb.FD_NAME,mb.MC_NAME):
            for arm in all_arms(scm):
                iv={v:.2 for v in arm}
                row=sample_true(scm,iv,np.array([1.,2.,3.,4.]))
                for v in arm: self.assertEqual(row[v],.2)
                self.assertNotIn("U",row)
                self.assertEqual(row,sample_true(scm,iv,np.array([1.,2.,3.,4.])))

    def test_monte_carlo_means_all_arms(self):
        rng=np.random.RandomState(832)
        for scm in (mb.PP_NAME,mb.FD_NAME,mb.MC_NAME):
            oracle=mb.population_evaluator(scm)
            for arm in all_arms(scm):
                iv={v:.4 for v in arm}
                ys=np.array([sample_true(scm,iv,rng.randn(4))["Y"] for _ in range(10000)])
                self.assertLess(abs(ys.mean()-oracle(arm,list(iv.values()))),6*ys.std()/np.sqrt(len(ys))+1e-8)


@unittest.skipUnless(os.environ.get("MATCHED_INTEGRATION")=="1", "requires full runtime; opt in explicitly")
class ExistingBackendIntegrationTests(unittest.TestCase):
    def test_scalar_backends_are_exact_and_budgeted(self):
        for method in ("CBO","QCBO","BO-S","BO"):
            e,meta=run_scalar(mb.PP_NAME,"A0",method,1000,max_purchases=1)
            self.assertEqual(e.sequential_index,1)
            self.assertEqual(e.cost,sum(x["cost"] for x in e.events))
            self.assertTrue(all(x["cost"]==0 for x in e.events if x["phase"]=="init"))
            self.assertTrue(meta["gp_noise_audit"])
            self.assertTrue(all(a["fixed"] for a in meta["gp_noise_audit"]))
            for row in e.events:
                self.assertAlmostEqual(row["measured"]["Y"],mb.population_do(e.scm,row["arm"],row["x"]))
            if method == "CBO":
                with patch("ccbo.matched_protocol.score_events", side_effect=AssertionError("scorer reached backend")):
                    again,_=run_scalar(mb.PP_NAME,"A0",method,1000,max_purchases=1)
                self.assertEqual(e.events,again.events)

    def test_affordable_singleton_under_tight_budget(self):
        e,meta=run_scalar(mb.PP_NAME,"A0","BO-S",1000,budget=1)
        self.assertEqual(sum(x["phase"]=="init" for x in e.events),9)
        self.assertEqual(e.cost,1)
        self.assertEqual(e.sequential_index,1)
        self.assertEqual(len(e.events[-1]["arm"]),1)
        snapshots=[c for a in meta["gp_noise_audit"] for c in a["acquisition_snapshots"]]
        self.assertEqual({len(c["arm"]) for c in snapshots},{1})   # the joint arm is unaffordable
        self.assertTrue(all(c["n_rows"]==3 for c in snapshots))

    def test_protected_qcbo_pair_uses_identical_measurements(self):
        a,_=run_scalar(mb.PP_NAME,"A0","QCBO",1000,max_purchases=2)
        b,_=run_scalar(mb.PP_NAME,"A1","QCBO",1000,max_purchases=2)
        self.assertEqual(a.events,b.events)

if __name__=="__main__": unittest.main()
