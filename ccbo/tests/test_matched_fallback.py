"""Native prior-fallback control must exactly reproduce B1 paired behavior."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from matched_fallback import run_fallback, force_observational_fallback
from ccbo import minibench as mb
from ccbo.matched_protocol import run_scalar, observational_data, score_events


@unittest.skipUnless(os.environ.get("MATCHED_INTEGRATION")=="1","requires actual CBO runtime")
class NativeFallbackIntegration(unittest.TestCase):
    def test_true_graph_forced_fallback_matches_actual_b1_three_decisions(self):
        a,control=run_fallback(mb.FD_NAME,"B0",1000,max_purchases=3)
        b,normal=run_scalar(mb.FD_NAME,"B1","CBO",1000,max_purchases=3)
        self.assertEqual(a.events,b.events)
        self.assertEqual(a.arms,b.arms)
        self.assertEqual(a.cost,b.cost)
        self.assertEqual(control["gp_noise_audit"],normal["gp_noise_audit"])
        self.assertEqual(score_events(a.scm,a.events,a.cost,a.null_estimate),score_events(b.scm,b.events,b.cost,b.null_estimate))
        self.assertFalse(control["graph_substitution"])
        self.assertTrue(control["native_causal_kernel_retained"])
        self.assertEqual({tuple(q) for q in control["forced_functional_queries"]}, {("M",),("X1",),("M","X1")})

    def test_native_fallback_functions_mean_and_variance_match_at_multiple_points(self):
        from ccbo.scm_graphs import get_original_graph
        from ccbo.coarsened_graph import CoarsenedGraph
        mb.register_variants()
        obs=observational_data(mb.FD_NAME,1000)
        graph=get_original_graph(mb.FD_NAME,obs)
        correct=CoarsenedGraph(graph,mb.fine_partition(mb.FD_NAME),mb.FD_NAME,obs,assumed_graph_name=mb.variant_name("B0"))
        wrong=CoarsenedGraph(graph,mb.fine_partition(mb.FD_NAME),mb.FD_NAME,obs,assumed_graph_name=mb.variant_name("B1"))
        # Arm membership remains that derived from the true B0 graph.
        arms_before=[list(a) for a in correct.get_sets()[0]]
        with force_observational_fallback():
            forced=correct.get_all_do()
        native=wrong.get_all_do()
        self.assertEqual(arms_before,correct.get_sets()[0])
        for arm in arms_before:
            name="compute_do_"+"".join(arm)
            for point in ([-2.]*len(arm),[0.]*len(arm),[2.]*len(arm)):
                self.assertEqual(forced[name](obs,{},point),native[name](obs,{},point))
                self.assertEqual(forced[name](obs,{},point),(float(obs.Y.mean()),float(obs.Y.var())))
        self.assertTrue(all(correct._arm_identifiable.values()))
        self.assertTrue(all(not ok for ok in wrong._arm_identifiable.values()))

    def test_scope_restores_and_rejects_unintended_conditions(self):
        import ccbo.coarsened_graph as graphs
        before=graphs.make_cdag_do_function
        with self.assertRaises(RuntimeError):
            with force_observational_fallback():
                raise RuntimeError("test")
        self.assertIs(before,graphs.make_cdag_do_function)
        with self.assertRaises(ValueError): run_fallback(mb.FD_NAME,"B1",1000)
        with self.assertRaises(ValueError): run_fallback(mb.PP_NAME,"A0",1000)

if __name__=="__main__":unittest.main()
