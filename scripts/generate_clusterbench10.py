"""Generate the ClusterBench10 benchmark dataset (linear-Gaussian SCM).

Writes the CausalBO_Benchmark-format files for the misspecification stress test
into ``third_party/CausalBO_Benchmark/`` and runs oracle pre-checks. Structure
(edges/partition) comes from ``ccbo.clusterbench10``; numeric coefficients live
here because they are tuned to satisfy the oracle constraints:

  (C1) do(X1) is the strongest *singleton* arm  (so the bow deletes the best
       singleton at the finest partition -> Tier-2, unrecoverable);
  (C2) the global optimum (uncapped) is do(all manipulables) = do(C1 u C2),
       a *union of clusters* -> the coarse partition is LOSSLESS (Prop. 5);
  (C3) the observational data exhibits corr(X1,X2) ~ 0.7 via the latent U12,
       so the bow X1->X2 + X1<->X2 is a genuine confounded misspecification.

The objective every method optimises is DiscoveredGraph's noise-free SEM
(linear, deterministic). The latent U12 enters only the *observational data*
(via this script's own sampler) and the *structure* (declared as a confounder
for QCBO / parsed from latent_variables); it is zero in the noise-free objective
-- exactly as the benchmark treats its own latents.

Run:  PYTHONPATH=. python scripts/generate_clusterbench10.py
"""

import os
import sys
import json
import pickle
import itertools

import numpy as np
import pandas as pd
import yaml

from ccbo import clusterbench10 as cb
from ccbo import benchmark as bench

DATASET = cb.NAME
TASK = "min"
N_OBS = 500
SEED = 0

# ---------------------------------------------------------------------------
# SCM coefficients (linear; all positive so the optimum sets levers to the
# lower range endpoint). See module docstring for the oracle constraints.
# ---------------------------------------------------------------------------
# Directed-edge coefficients (parent -> child).
COEF = {
    ("X3", "X2"): 0.3,          # intra-C1 (Tier-1 target)
    ("X1", "M2"): 1.0, ("X2", "M2"): 1.0, ("X3", "M1"): 1.0,
    ("X4", "M3"): 1.0, ("X5", "M3"): 1.0, ("X5", "Z"): 1.0,
    ("X1", "Y"): 3.5, ("X2", "Y"): 1.5, ("X3", "Y"): 0.5,
    ("M1", "Y"): 1.0, ("M2", "Y"): 1.0, ("M3", "Y"): 1.0, ("Z", "Y"): 1.0,
}
# Exogenous Gaussian noise std per node (observational sampler only).
NOISE_STD = {n: 0.7 for n in cb.NODES}
ROOT_STD = {"X1": 0.7, "X3": 1.0, "X5": 1.0}  # roots' own spread (obs only)
# Latent-confounder spreads (U12 -> {X1,X2}; U15 -> {X1,X5}); see cb.CONFOUNDERS.
LATENT_STD = {"U12": 1.6, "U15": 1.0}


def _var_latents():
    """node -> list of latent confounders feeding it (from cb.CONFOUNDERS)."""
    m = {}
    for lat, vs in cb.CONFOUNDERS:
        for v in vs:
            m.setdefault(v, []).append(lat)
    return m


def _causal_order(edges):
    """Topological order over NODES given directed edges."""
    from collections import defaultdict, deque
    succ = defaultdict(list)
    indeg = {n: 0 for n in cb.NODES}
    for u, v in edges:
        succ[u].append(v)
        indeg[v] += 1
    q = deque([n for n in cb.NODES if indeg[n] == 0])
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    assert len(order) == len(cb.NODES), "graph not acyclic"
    return order


def _parents(edges):
    p = {n: [] for n in cb.NODES}
    for u, v in edges:
        p[v].append(u)
    return p


def sample_observational(n, seed):
    """Forward-sample the true SCM (linear + shared latent confounders) -> DataFrame."""
    rng = np.random.RandomState(seed)
    order = _causal_order(cb.TRUE_EDGES)
    parents = _parents(cb.TRUE_EDGES)
    latents = {lat: rng.normal(0, LATENT_STD.get(lat, 1.0), n)
               for lat, _ in cb.CONFOUNDERS}
    var_latents = _var_latents()
    vals = {}
    for node in order:
        mean = np.zeros(n)
        for p in parents[node]:
            mean = mean + COEF[(p, node)] * vals[p]
        for lat in var_latents.get(node, []):
            mean = mean + latents[lat]              # shared latent confounding
        std = ROOT_STD.get(node, NOISE_STD[node])
        vals[node] = mean + rng.normal(0, std, n)
    return pd.DataFrame({n_: vals[n_] for n_ in cb.NODES})


def build_sem_equations(order, parents):
    """sem_equations.json: linear nodes; roots are deps=[] (deterministic 0 when
    free in the objective); X1/X2 carry latent_variables=['U12'] so parse_dag
    registers the X1<->X2 confounder for QCBO."""
    var_latents = _var_latents()
    variables = {}
    for node in cb.NODES:
        rp = {"noise_std": float(NOISE_STD[node])}
        if node in var_latents:
            rp["latent_variables"] = list(var_latents[node])
        variables[node] = {
            "type": "endogenous",
            "dependencies": list(parents[node]),
            "intercept": 0.0,
            "coefficients": {p: float(COEF[(p, node)]) for p in parents[node]},
            "relationship_type": "linear",
            "relationship_params": rp,
        }
    return {"variables": variables, "causal_order": list(order)}


def build_model_results(order):
    """model_results.pkl: node_names, adjacency_matrix[child,parent], causal_order."""
    idx = {n: i for i, n in enumerate(cb.NODES)}
    A = np.zeros((len(cb.NODES), len(cb.NODES)))
    for (u, v), c in COEF.items():
        A[idx[v], idx[u]] = c
    return {"node_names": list(cb.NODES), "adjacency_matrix": A,
            "causal_order": list(order)}


def write_dataset(paths):
    order = _causal_order(cb.TRUE_EDGES)
    parents = _parents(cb.TRUE_EDGES)

    obs = sample_observational(N_OBS, SEED)
    feat = {n: {"min": float(obs[n].min()), "max": float(obs[n].max()),
                "mean": float(obs[n].mean()), "std": float(obs[n].std())}
            for n in cb.NODES}
    interventional_domain = {v: [feat[v]["min"], feat[v]["max"]]
                             for v in cb.MANIPULATIVE}
    config = {
        "cols_to_drop": None,
        "predetermined_exogenous_vars": [lat for lat, _ in cb.CONFOUNDERS],
        "target": "Y",
        "num_intervention": 1,
        "intervention": list(cb.MANIPULATIVE),
        "interventional_domain": interventional_domain,
        "num_observations": N_OBS,
        "task": TASK,
    }
    sem_eq = build_sem_equations(order, parents)
    model_results = build_model_results(order)

    os.makedirs(os.path.dirname(paths["config"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["sem_eq"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["obs"]), exist_ok=True)
    with open(paths["config"], "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(paths["sem_eq"], "w") as f:
        json.dump(sem_eq, f, indent=2)
    with open(paths["sem_model"], "wb") as f:
        pickle.dump(model_results, f)
    obs.to_csv(paths["obs"], index=False)
    with open(paths["feat"], "w") as f:
        json.dump(feat, f, indent=2)
    print(f"wrote dataset files for {DATASET}; corr(X1,X2)={obs['X1'].corr(obs['X2']):.3f}")
    return config, feat


def generate_interventional_pkls(graph, config, feat, n_bo=400, n_cbo=200):
    """BO/CBO initial-interventional pickles in the benchmark's format.

    BO:  [data_x (N, n_manip), data_y (N,)].
    CBO: list of [n_vars, intervention_set, data_x, data_y] per get_sets() arm.
    Outcomes use the graph's noise-free objective (the same one every method
    optimises), so the initial incumbents are consistent across methods.
    """
    rng = np.random.RandomState(SEED)
    manip = list(config["intervention"])

    def sample_arm(arm, n):
        idx_map = {v: i for i, v in enumerate(arm)}
        tfn, _ = graph.intervention_function(idx_map)
        X = np.column_stack([rng.uniform(feat[v]["min"], feat[v]["max"], n)
                             for v in arm])
        Y = np.array([float(tfn(X[i:i + 1])[0, 0]) for i in range(n)])
        return X, Y

    Xall, Yall = sample_arm(manip, n_bo)
    bo_data = [Xall, Yall]

    cbo_vars, _, _ = graph.get_sets()
    cbo_data = []
    for arm in cbo_vars:
        X, Y = sample_arm(list(arm), n_cbo)
        Xc = X[:, 0] if len(arm) == 1 else X
        cbo_data.append([len(arm), list(arm), Xc, Y[:, None]])

    idir = os.path.join(bench.BENCH_ROOT, "interventional_datasets")
    os.makedirs(idir, exist_ok=True)
    with open(os.path.join(idir, f"{DATASET}_interventional_BO.pkl"), "wb") as f:
        pickle.dump(bo_data, f)
    with open(os.path.join(idir, f"{DATASET}_interventional_CBO.pkl"), "wb") as f:
        pickle.dump(cbo_data, f)
    print(f"wrote interventional pkls: BO {Xall.shape}, CBO {len(cbo_data)} arms")
    return bo_data, cbo_data


def main():
    bench._ensure_benchmark_on_path()
    from baselines.BO_CBO.graph import setup_optimization_from_discovery

    paths = bench._paths(DATASET)
    config, feat = write_dataset(paths)

    # Build the benchmark graph from the files we just wrote.
    df = pd.read_csv(paths["obs"])
    with open(paths["sem_eq"]) as f:
        sem_eq = json.load(f)
    model_results = np.load(paths["sem_model"], allow_pickle=True)
    graph, obs_samples, _ = setup_optimization_from_discovery(
        model_results, sem_eq, df, "Y", config, feat)

    # ---- Oracle pre-checks: deterministic noise-free objective per arm ----
    ranges = {v: (feat[v]["min"], feat[v]["max"]) for v in cb.MANIPULATIVE}

    def eval_corner(arm, signs):
        idx_map = {v: i for i, v in enumerate(arm)}
        tfn, _ = graph.intervention_function(idx_map)
        x = np.array([[ranges[v][0] if s < 0 else ranges[v][1]
                       for v, s in zip(arm, signs)]])
        return float(tfn(x)[0, 0])

    def best_over_corners(arm):
        best = np.inf
        for signs in itertools.product((-1, 1), repeat=len(arm)):
            best = min(best, eval_corner(arm, signs))
        return best

    singletons = {(v,): best_over_corners([v]) for v in cb.MANIPULATIVE}
    print("\nsingleton arm optima (lower=better):")
    for s, val in sorted(singletons.items(), key=lambda kv: kv[1]):
        print(f"  do({s[0]:<3s}) = {val:8.3f}")
    best_singleton = min(singletons, key=singletons.get)

    all_arm = list(cb.MANIPULATIVE)
    v_all = best_over_corners(all_arm)
    v_c1 = best_over_corners(["X1", "X2", "X3"])
    print(f"\n  do(C1=X1,X2,X3) = {v_c1:8.3f}")
    print(f"  do(all=C1uC2)   = {v_all:8.3f}  (global optimum candidate)")

    # Check the global optimum over ALL subsets is a union of clusters.
    global_best, global_arm = np.inf, None
    for r in range(1, len(all_arm) + 1):
        for arm in itertools.combinations(all_arm, r):
            val = best_over_corners(list(arm))
            if val < global_best - 1e-9:
                global_best, global_arm = val, arm
    clusters = [set(c) for c in cb.COARSE_CLUSTERS]
    is_union = set(global_arm) == set().union(*[c for c in clusters
                                                if c <= set(global_arm)]) and all(
        (set(global_arm) & c in (set(), c)) for c in clusters)
    print(f"\n  global optimum: do{global_arm} = {global_best:.3f}; "
          f"union-of-clusters={is_union}")

    print("\n=== ORACLE CHECKS ===")
    ok1 = best_singleton == ("X1",)
    ok2 = is_union
    print(f"  (C1) best singleton is do(X1): {ok1}  [{best_singleton}]")
    print(f"  (C2) global optimum is union of clusters: {ok2}")
    if not (ok1 and ok2):
        print("  -> TUNE COEFFICIENTS and re-run.")
        sys.exit(1)

    # theoretical_best y*: best Y over random interventional samples (matches
    # the benchmark's own convention) + the oracle global optimum.
    ystar = float(global_best)
    with open(paths["ybest"], "w") as f:
        json.dump({"dataset": DATASET, "target": "Y", "task": TASK,
                   "global_best_value": ystar,
                   "note": "deterministic oracle over arm corners"}, f, indent=2)
    print(f"\nwrote theoretical_best y*={ystar:.4f}")

    generate_interventional_pkls(graph, config, feat)
    print("=== ClusterBench10 generation OK ===")


if __name__ == "__main__":
    main()
