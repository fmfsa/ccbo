"""Structure-injection shim for the external baselines.

Hard invariant: every method optimises the SAME objective (the true SCM's
noise-free ``intervention_function``); only the *structure each method reasons
with* is perturbed. For the benchmark's BO_CBO baseline that structure is
``model_results['adjacency_matrix']`` (consumed by ``fit_all_models`` for the
causal prior), which is independent of the objective (built from
``sem_equations`` dependencies+coefficients). So we perturb the adjacency
*before* ``setup_optimization_from_discovery`` and leave the SEM untouched.

``seam_ok`` is the guard that this decoupling actually holds: it asserts the
objective is byte-identical under the perturbed vs. true adjacency.
"""

import numpy as np

from ccbo import benchmark, clusterbench10 as cb


def perturbed_adjacency_matrix(node_names, pid):
    """Binary adjacency[child, parent] for the perturbation ``pid``.

    fit_all_models only tests ``adjacency_matrix[i, j] != 0`` (edge presence),
    so a binary matrix from the variant's edge set is sufficient and makes P0
    via this path identical to the shipped dataset (same parent sets).
    """
    idx = {n: i for i, n in enumerate(node_names)}
    A = np.zeros((len(node_names), len(node_names)))
    for (u, v) in cb.variant_edges(pid):
        if u in idx and v in idx:
            A[idx[v], idx[u]] = 1.0
    return A


def build_cbo_graph(ds, seed, pid):
    """DiscoveredGraph for ``ds`` whose causal prior uses the perturbed
    structure ``pid`` but whose objective is the true SCM.

    Returns ``(graph, observational_samples, functions, config)``.
    """
    benchmark._ensure_benchmark_on_path()
    from baselines.BO_CBO.graph import setup_optimization_from_discovery

    mr, sem_eq, feat, obs, config = benchmark._load_raw(ds, seed=seed)
    mr = dict(mr)
    mr["adjacency_matrix"] = perturbed_adjacency_matrix(mr["node_names"], pid)
    graph, obs_samples, functions = setup_optimization_from_discovery(
        mr, sem_eq, obs, config["target"], config, feat)
    return graph, obs_samples, functions, config


def seam_ok(ds, pid, n=8, tol=1e-9, seed=0):
    """Assert the OBJECTIVE is unchanged by the structure perturbation.

    Evaluates intervention_function on every singleton arm at random points
    under the true (P0) vs perturbed (pid) adjacency; returns True iff the
    outcomes match to ``tol`` (i.e. the perturbation touched only the prior).
    """
    g0, _, _, config = build_cbo_graph(ds, seed, "P0")
    g1, _, _, _ = build_cbo_graph(ds, seed, pid)
    rng = np.random.RandomState(0)
    manip = list(config["intervention"])
    dom = config["interventional_domain"]
    for v in manip:
        idx_map = {v: 0}
        f0, _ = g0.intervention_function(idx_map)
        f1, _ = g1.intervention_function(idx_map)
        xs = rng.uniform(dom[v][0], dom[v][1], n)
        for x in xs:
            a = float(f0(np.array([[x]]))[0, 0])
            b = float(f1(np.array([[x]]))[0, 0])
            if abs(a - b) > tol:
                return False
    return True


if __name__ == "__main__":
    cb.register_variants()
    print(f"seam check on {cb.NAME} (objective must be invariant to structure):")
    for p in cb.PERTURBATIONS:
        if p["id"] == "P0":
            continue
        ok = seam_ok(cb.NAME, p["id"])
        print(f"  {p['id']:4s} {p['locus']:16s} objective-invariant={ok}")
        assert ok, f"SEAM LEAK: objective changed under {p['id']}"
    print("OK: structure injection leaves the objective byte-identical")
