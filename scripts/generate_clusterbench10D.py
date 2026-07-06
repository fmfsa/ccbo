"""Generate ClusterBench10D: the DAMAGE companion to ClusterBench10.

Same 10-variable structure (edges, latents, partition) as ``ccbo.clusterbench10``;
only Y's bowl centers differ. The design realizes the negative sign of the
search-set-change channel: the strongest singleton do(X1) is not a decoy but the
*global optimum itself*, so the intra-cluster bow P1 (add X1->X2 on the
X1<->X2-confounded pair) deletes the cheap optimal arm from the finest
exploration set and forces full-DAG CBO onto the higher-dimensional arms
do(X1,X2) / do(C1) that reach the same value more slowly -- a genuine
convergence cost (dGAP < 0), not a prior bias the data can wash out.

Mechanism. Y's centers are placed at the values do(X1=2) naturally induces in
the noise-free SEM (X2=0.3*X3=0, X3=0, M1=0, M2=X1+X2=2, M3=0, Z=0), so:

  * do(X1=2) attains y* = 0 with a 1-dimensional search;
  * do(C1)=(2,0,0) attains the same 0, so the optimum is ALSO a union of
    clusters -- the coarse partition stays LOSSLESS (no price-of-coarsening
    confound in the robustness comparison);
  * every other singleton is far worse (>= a1*c1^2 = 12), so nothing else
    competes with the deleted arm.

Contrast with ClusterBench10, where do(X1) is a decoy and deleting it HELPS:
together the pair shows both signs of the same mechanism, with QCBO-coarse
byte-identical in both (the bow is intra-cluster either way).

Oracle checks:
  (D1) do(X1) attains the global optimum (gap to y* < 0.02) and beats every
       other singleton by >= 5;
  (D2) the coarse partition is lossless (V(coarse) - y* < 0.02);
  (D3) corr(X1,X2) ~ 0.7 (the bow is a genuine confounded misspecification).

Writes the same benchmark-format files as generate_clusterbench10.py under the
dataset name ``ClusterBench10D``.

Run:  PYTHONPATH=.:scripts conda run -n ccbo python scripts/generate_clusterbench10D.py
"""

import json
import sys
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ccbo import clusterbench10 as cb
from ccbo import benchmark as bench

import generate_clusterbench10 as gen

DATASET = "ClusterBench10D"
TASK = "min"
N_OBS = 500
SEED = 0

# --- D coefficients (structure and mechanisms identical to ClusterBench10; ----
# only Y's centers move so that do(X1=2) is the global optimum).
LIN_COEF = dict(gen.LIN_COEF)
YA = dict(gen.YA)
YC = {"X1": 2.0, "X2": 0.0, "X3": 0.0, "M1": 0.0, "M2": 2.0, "M3": 0.0, "Z": 0.0}
Y_INTERCEPT = 0.0


def _patch_gen():
    gen.DATASET = DATASET
    gen.LIN_COEF = LIN_COEF
    gen.YA = YA
    gen.YC = YC
    gen.Y_INTERCEPT = Y_INTERCEPT
    gen.N_OBS = N_OBS
    gen.SEED = SEED
    gen.TASK = TASK


def main():
    _patch_gen()
    bench._ensure_benchmark_on_path()
    from baselines.BO_CBO.graph import setup_optimization_from_discovery

    paths = bench._paths(DATASET)
    config, feat = gen.write_dataset(paths)

    df = pd.read_csv(paths["obs"])
    with open(paths["sem_eq"]) as f:
        sem_eq = json.load(f)
    model_results = np.load(paths["sem_model"], allow_pickle=True)
    graph, _, _ = setup_optimization_from_discovery(
        model_results, sem_eq, df, "Y", config, feat)

    ranges = {v: (feat[v]["min"], feat[v]["max"]) for v in cb.MANIPULATIVE}

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
    singles = {v: best_over_box([v])[0] for v in all_arm}

    global_best, global_arm, global_x = np.inf, None, None
    for r in range(1, len(all_arm) + 1):
        for arm in itertools.combinations(all_arm, r):
            val, x = best_over_box(list(arm))
            if val < global_best - 1e-6:
                global_best, global_arm, global_x = val, arm, x

    union_arms = [["X1", "X2", "X3"], ["X4", "X5"],
                  ["X1", "X2", "X3", "X4", "X5"]]
    v_coarse = min(best_over_box(a)[0] for a in union_arms)

    x1_best = singles["X1"]
    runner_up = min(v for k, v in singles.items() if k != "X1")

    print(f"\n  singletons: " + "  ".join(f"do({k})={v:.2f}"
                                          for k, v in sorted(singles.items())))
    print(f"  global optimum: do{global_arm} = {global_best:.4f} "
          f"at {np.round(global_x, 3)}")
    print(f"  V(coarse) = {v_coarse:.4f}")

    print("\n=== ORACLE CHECKS (damage variant) ===")
    ok1 = (x1_best - global_best < 0.02) and (runner_up - x1_best >= 5.0)
    ok2 = (v_coarse - global_best) < 0.02
    corr = float(df["X1"].corr(df["X2"]))
    ok3 = 0.55 <= corr <= 0.85
    print(f"  (D1) do(X1) attains the optimum and beats every other singleton "
          f"by >=5: {ok1}  [do(X1)={x1_best:.4f}, runner-up={runner_up:.2f}]")
    print(f"  (D2) coarse partition lossless: {ok2}  "
          f"[V(coarse)-y*={v_coarse - global_best:.4f}]")
    print(f"  (D3) corr(X1,X2) ~ 0.7: {ok3}  [{corr:.3f}]")
    if not (ok1 and ok2 and ok3):
        print("  -> TUNE COEFFICIENTS (YA / YC) and re-run.")
        sys.exit(1)

    with open(paths["ybest"], "w") as f:
        json.dump({"dataset": DATASET, "target": "Y", "task": TASK,
                   "global_best_value": float(global_best),
                   "global_best_arm": list(global_arm),
                   "coarse_best_value": float(v_coarse),
                   "singleton_X1_value": float(x1_best),
                   "note": "continuous oracle over the observational box"},
                  f, indent=2)
    print(f"\nwrote theoretical_best y*={global_best:.4f}")
    print("=== ClusterBench10D generation OK ===")


if __name__ == "__main__":
    main()
