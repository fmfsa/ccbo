"""Prior-attribution control: true FD graph and native observational fallback.

This opt-in control intercepts only effect-functional construction, causing the
existing CoarsenedGraph.get_all_do fallback branch to supply its own observational
mean AND variance. Native fine arms, causal kernel, fitted-noise policy and data
streams are retained. No graph is substituted and no plain-RBF ablation is used.
"""
from contextlib import contextmanager
from ccbo import minibench as mb
from ccbo.matched_protocol import run_scalar, observational_data


@contextmanager
def force_observational_fallback():
    import ccbo.coarsened_graph as graphs
    original=graphs.make_cdag_do_function
    calls=[]
    def decline_functional(dag, arm, partition, target, observations, *args, **kwargs):
        calls.append(list(arm))
        return None, {"method":"none", "reason":"explicit prior-attribution control: force native observational fallback"}
    graphs.make_cdag_do_function=decline_functional
    try:
        yield calls
    finally:
        graphs.make_cdag_do_function=original


def run_fallback(scm, cond, seed, budget=None, n_init=3, max_purchases=None):
    if scm != mb.FD_NAME or cond != "B0":
        raise ValueError("CBO-FALLBACK is only defined for FrontDoor/B0")
    with force_observational_fallback() as calls:
        experiment,meta=run_scalar(scm,cond,"CBO",seed,budget,n_init,max_purchases)
    obs=observational_data(scm,seed)
    meta.update(prior_control="native CoarsenedGraph observational fallback forced on true B0 graph",
                graph_substitution=False, native_causal_kernel_retained=True,
                forced_functional_queries=calls,
                fallback_mean=float(obs.Y.mean()), fallback_variance=float(obs.Y.var()),
                comparison_target="FrontDoor/B1/CBO under the same paired seed and budget")
    return experiment,meta
