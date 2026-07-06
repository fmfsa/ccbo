"""Generate ClusterBench10L: the LOSSY-partition companion to ClusterBench10.

Same 10-variable structure (edges, latents, partition) as ``ccbo.clusterbench10``;
only coefficients and the interventional domain differ. The design realizes the
lossy side of the price-of-coarsening proposition: the global optimum requires
intervening on a *strict subset* of cluster C1, so committing to the coarse
partition costs a bounded plateau above y*.

Mechanism.  With hard interventions, deterministic mechanisms, and boxes that
cover the natural responses, pinning a variable can always replicate leaving it
responsive, so a whole-cluster arm matches any sub-cluster arm and coarsening
is lossless. The price appears exactly when a variable's *optimal natural
response* lies outside its interventional (policy) box:

  * the X3->X2 mechanism is strengthened to 1.5 (was 0.3), and Y's bowl centers
    X2 at 2.7;
  * interventions are restricted to a policy box [-2, 2] per manipulable
    variable (narrower than the natural range -- you cannot *prescribe* extreme
    values, but the mechanism can *produce* them);
  * hence the optimum is do(X1, X3): X3 ~ 1.7 drives X2 = 1.5*X3 ~ 2.6 near its
    center, which no in-box pin of X2 can reach. Every cluster-union arm
    (do(C1), do(C1 u C2)) must pin X2 <= 2 and pays the gap.

Oracle checks:
  (L1) the global optimum arm is NOT a union of coarse clusters (it splits C1);
  (L2) the price V(coarse) - y* is bounded away from 0 (> 0.3);
  (L3) the global optimum is interior to the policy box (not a vertex artifact).

Writes the same benchmark-format files as generate_clusterbench10.py under the
dataset name ``ClusterBench10L`` (interventional pkls are skipped: only the
QCBO runners consume this dataset and they sample their own initial data).

Run:  PYTHONPATH=. python scripts/generate_clusterbench10L.py
"""

import os
import sys
import json
import pickle
import itertools

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from ccbo import clusterbench10 as cb
from ccbo import benchmark as bench

import generate_clusterbench10 as gen

DATASET = "ClusterBench10L"
TASK = "min"
N_OBS = 500
SEED = 0

# Policy box: hard interventions are restricted to [-2, 2] per manipulable
# variable (independent of the observational range).
POLICY_BOX = {v: (-2.0, 2.0) for v in cb.MANIPULATIVE}

# --- L coefficients (structure identical to ClusterBench10) -----------------
LIN_COEF = dict(gen.LIN_COEF)
LIN_COEF[("X3", "X2")] = 1.5          # strong mechanism: X2 responds to X3

YA = {"X1": 3.0, "X2": 2.0, "X3": 1.0, "M1": 0.4, "M2": 0.3, "M3": 0.3, "Z": 0.3}
# X2's center 2.7 is OUTSIDE the policy box but reachable as 1.5 * X3 with
# X3 ~ 1.8 inside it; M2's center sits at the natural optimum X1 + X2 ~ 4.
YC = {"X1": 1.5, "X2": 2.7, "X3": 1.5, "M1": 1.5, "M2": 4.0, "M3": 0.0, "Z": 0.0}
Y_INTERCEPT = 0.0


def _patch_gen():
    """Point the shared generator machinery at the L coefficients/dataset."""
    gen.DATASET = DATASET
    gen.LIN_COEF = LIN_COEF
    gen.YA = YA
    gen.YC = YC
    gen.Y_INTERCEPT = Y_INTERCEPT
    gen.N_OBS = N_OBS
    gen.SEED = SEED
    gen.TASK = TASK


def write_dataset(paths):
    """Same files as gen.write_dataset but with the policy-box domain."""
    order = gen._causal_order(cb.TRUE_EDGES)
    parents = gen._parents(cb.TRUE_EDGES)

    obs = gen.sample_observational(N_OBS, SEED)
    # The benchmark's DiscoveredGraph.get_interventional_ranges() reads the hard-
    # intervention bounds from feature_params[var]['min'/'max'] and IGNORES the
    # config's interventional_domain (baselines/BO_CBO/graph.py:788). If we leave
    # feat at the observational min/max, a cluster-union arm can pin X2 anywhere in
    # the wide data range and the policy-box constraint never binds -- the coarse
    # partition then reaches the split optimum and the price of coarsening collapses
    # to ~0 (the L1-L3 oracle assumes the [-2,2] box). So write the POLICY_BOX as the
    # min/max for manipulable variables; mean/std and non-manipulable bounds are
    # informational (get_interventional_ranges only touches intervention vars).
    feat = {n: {"min": float(obs[n].min()), "max": float(obs[n].max()),
                "mean": float(obs[n].mean()), "std": float(obs[n].std())}
            for n in cb.NODES}
    for v in cb.MANIPULATIVE:
        feat[v]["min"], feat[v]["max"] = float(POLICY_BOX[v][0]), float(POLICY_BOX[v][1])
    interventional_domain = {v: [POLICY_BOX[v][0], POLICY_BOX[v][1]]
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
    sem_eq = gen.build_sem_equations(order, parents)
    model_results = gen.build_model_results(order)

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
    print(f"wrote dataset files for {DATASET}; "
          f"corr(X1,X2)={obs['X1'].corr(obs['X2']):.3f}")
    return config, feat


def main():
    _patch_gen()
    bench._ensure_benchmark_on_path()
    from baselines.BO_CBO.graph import setup_optimization_from_discovery

    paths = bench._paths(DATASET)
    config, feat = write_dataset(paths)

    df = pd.read_csv(paths["obs"])
    with open(paths["sem_eq"]) as f:
        sem_eq = json.load(f)
    model_results = np.load(paths["sem_model"], allow_pickle=True)
    graph, obs_samples, _ = setup_optimization_from_discovery(
        model_results, sem_eq, df, "Y", config, feat)

    ranges = POLICY_BOX

    def best_over_box(arm, n_starts=12):
        idx_map = {v: i for i, v in enumerate(arm)}
        tfn, _ = graph.intervention_function(idx_map)
        lo = np.array([ranges[v][0] for v in arm])
        hi = np.array([ranges[v][1] for v in arm])

        def f(x):
            return float(tfn(x.reshape(1, -1))[0, 0])

        rng = np.random.RandomState(0)
        best, best_x = np.inf, None
        for _ in range(n_starts):
            x0 = lo + rng.rand(len(arm)) * (hi - lo)
            res = minimize(f, x0, bounds=list(zip(lo, hi)), method="L-BFGS-B")
            if res.fun < best:
                best, best_x = float(res.fun), res.x
        return best, best_x

    all_arm = list(cb.MANIPULATIVE)

    # Global optimum over ALL subsets; smallest-cardinality argmin.
    global_best, global_arm, global_x = np.inf, None, None
    for r in range(1, len(all_arm) + 1):
        for arm in itertools.combinations(all_arm, r):
            val, x = best_over_box(list(arm))
            if val < global_best - 1e-6:
                global_best, global_arm, global_x = val, arm, x

    # Coarse value: best over the cluster-union arms.
    union_arms = [["X1", "X2", "X3"], ["X4", "X5"],
                  ["X1", "X2", "X3", "X4", "X5"]]
    union_vals = {tuple(a): best_over_box(a)[0] for a in union_arms}
    v_coarse = min(union_vals.values())
    price = v_coarse - global_best

    clusters = [set(c) for c in cb.COARSE_CLUSTERS]
    is_union = all((set(global_arm) & c in (set(), c)) for c in clusters)
    interior = all(ranges[v][0] + 1e-3 < global_x[i] < ranges[v][1] - 1e-3
                   for i, v in enumerate(global_arm))

    print(f"\n  global optimum: do{global_arm} = {global_best:.3f} "
          f"at {np.round(global_x, 3)}")
    for a, v in union_vals.items():
        print(f"  union arm do({','.join(a)}) = {v:.3f}")
    print(f"  V(coarse) = {v_coarse:.3f}; price of coarsening = {price:.3f}")

    print("\n=== ORACLE CHECKS (lossy variant) ===")
    ok1 = not is_union
    ok2 = price > 0.3
    ok3 = interior
    print(f"  (L1) global optimum is NOT a union of clusters: {ok1}  [do{global_arm}]")
    print(f"  (L2) price of coarsening > 0.3: {ok2}  [{price:.3f}]")
    print(f"  (L3) global optimum is interior: {ok3}  [{np.round(global_x, 3)}]")
    if not (ok1 and ok2 and ok3):
        print("  -> TUNE COEFFICIENTS (YA / YC / LIN_COEF) and re-run.")
        sys.exit(1)

    with open(paths["ybest"], "w") as f:
        json.dump({"dataset": DATASET, "target": "Y", "task": TASK,
                   "global_best_value": float(global_best),
                   "global_best_arm": list(global_arm),
                   "coarse_best_value": float(v_coarse),
                   "price_of_coarsening": float(price),
                   "note": "continuous oracle over the policy box [-2,2]^5"},
                  f, indent=2)
    print(f"\nwrote theoretical_best y*={global_best:.4f} "
          f"(V(coarse)={v_coarse:.4f}, price={price:.4f})")
    print("=== ClusterBench10L generation OK ===")


if __name__ == "__main__":
    main()
