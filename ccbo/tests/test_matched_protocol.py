"""Pure protocol contracts; optional real-backend tests are explicitly opt-in.

Run: python -m unittest ccbo.tests.test_matched_protocol -v
Real pilot integration (requires GPy/Emukit/CEO): MATCHED_INTEGRATION=1 ...
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
        e=Experiment(mb.PP_NAME,1000,all_arms(mb.PP_NAME),budget=13)
        self.assertEqual(e.cost,12)
        self.assertFalse(e.affordable(("X1","X2")))
        with self.assertRaises(BudgetExhausted): e.purchase(("X1","X2"),[0,0])
        self.assertEqual(e.cost,12)
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
    def test_scalar_backends_are_noisy_and_budgeted(self):
        for method in ("CBO","QCBO","BO-S","BO"):
            e,meta=run_scalar(mb.PP_NAME,"A0",method,1000,max_purchases=1)
            self.assertEqual(e.sequential_index,1)
            self.assertEqual(meta["new_observation_rows"],0)
            self.assertTrue(meta["gp_noise_audit"])
            self.assertTrue(all(not x["fixed"] for x in meta["gp_noise_audit"]))
            oracle=mb.population_evaluator(e.scm)
            self.assertTrue(any(abs(row["measured"]["Y"]-oracle(row["arm"],row["x"]))>1e-8 for row in e.events))
            if method == "CBO":
                with patch.object(mb, "population_evaluator", side_effect=AssertionError("oracle reached backend")), patch("ccbo.matched_protocol.score_events", side_effect=AssertionError("scorer reached backend")):
                    poisoned,_=run_scalar(mb.PP_NAME,"A0",method,1000,max_purchases=1)
                self.assertEqual(e.events,poisoned.events)

    def test_affordable_singleton_and_fit_before_acquisition(self):
        e,meta=run_scalar(mb.PP_NAME,"A0","BO-S",1000,budget=13)
        self.assertEqual(sum(x["cost"] for x in e.events if x["phase"]=="init"),12)
        self.assertEqual(e.cost,13)
        self.assertEqual(e.sequential_index,1)
        self.assertEqual(len(e.events[-1]["arm"]),1)
        audited=[a for a in meta["gp_noise_audit"] if a.get("acquisition_snapshots")]
        self.assertEqual(len(audited),2)  # both affordable singleton acquisitions
        for audit in audited:
            self.assertTrue(audit["fitted_before_acquisition"])
            for call in audit["acquisition_snapshots"]:
                self.assertEqual(call["parameters"],audit["fitted_parameters"])
                self.assertEqual(call["variance"],audit["fitted_variance"])
                self.assertEqual(call["n_rows"],3)
        self.assertTrue(any(a["initial_variance"] != a["fitted_variance"] for a in audited))

    def test_protected_qcbo_pair_uses_identical_measurements(self):
        a,_=run_scalar(mb.PP_NAME,"A0","QCBO",1000,max_purchases=2)
        b,_=run_scalar(mb.PP_NAME,"A1","QCBO",1000,max_purchases=2)
        self.assertEqual(a.events,b.events)

    def test_real_ceo_replays_purchased_rows(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
        from matched_ceo_runtime import run_ceo
        with patch.object(mb, "population_evaluator", side_effect=AssertionError("oracle reached CEO")), patch("ccbo.matched_protocol.score_events", side_effect=AssertionError("scorer reached CEO")):
            e,meta=run_ceo(mb.MC_NAME,"C0",1000,max_purchases=3,ceo_root=os.environ.get("CEO_ROOT"))
        self.assertEqual(e.sequential_index,3)
        self.assertEqual(meta["feedback_audit"],dict(measurement_calls=3,bookkeeping_replays=3))
        self.assertEqual(meta["new_observation_rows"],0)

if __name__=="__main__": unittest.main()


def test_ceo_graphs_keep_correct_chain_and_declared_graph_edits():
    from ccbo.baselines.ceo import observable_edges, variant_graph
    assert set(observable_edges(mb.MC_NAME, 'C0')) == {('X1', 'X2'), ('X2', 'Y')}
    for condition in mb.PERTURBATIONS:
        scm, cond = condition['scm'], condition['id']
        expected = set(mb.variant_edges(cond))
        graph = variant_graph(scm, cond)
        assert set(observable_edges(scm, cond)) == expected
        assert set(graph.edges()) == {(u + '_0', v + '_0') for u, v in expected}


def test_ceo_hidden_confounding_does_not_create_observed_parent_nodes():
    from ccbo.baselines.ceo import SCMS, variant_graph
    for scm, cond in [(mb.PP_NAME, 'A0'), (mb.FD_NAME, 'B0')]:
        graph = variant_graph(scm, cond)
        assert set(graph.nodes()) == {v + '_0' for v in SCMS[scm]['nodes']}
        assert all(not v.startswith('U') for v in graph.nodes())
