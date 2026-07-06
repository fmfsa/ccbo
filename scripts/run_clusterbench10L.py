"""Price-of-coarsening experiment on ClusterBench10L (the lossy variant).

ClusterBench10L shares ClusterBench10's structure but its global optimum requires
splitting cluster C1 (arm do(X1,X3)), so the coarse partition is LOSSY: Prop. 5's
price of coarsening is realized. This runs the same four seam methods as Table 1
(BO, CBO=QCBO-finest, QCBO-coarse) on the correct DAG and quantifies the plateau
QCBO-coarse pays against the optimum CBO reaches.

Same fair-comparison seam as scripts/run_misspec_fullfield.py: one objective (the
true ClusterBench10L SEM), shared seeds/budget, only the assumed structure per
method. Writes results/clusterbench10L.json.

Run:
  PYTHONPATH=.:scripts conda run -n ccbo python scripts/run_clusterbench10L.py \
      --seeds 10 --trials 100 --jobs 10
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from ccbo import benchmark, clusterbench10 as cb
from ccbo.metrics import simple_regret, cumulative_regret
import misspec_inject

sys.path.insert(0, benchmark.BENCH_ROOT)
from metrics.GAP import GAP            # noqa: E402
from metrics.PA_GAP import PA_GAP      # noqa: E402

DS = "ClusterBench10L"
RUNDIR = os.path.join(benchmark.BENCH_ROOT, "results", "_lossy")
COARSE = [list(c) for c in cb.COARSE_CLUSTERS]


def score(traj, ystar, task, n):
    t = list(traj)[: n + 1]
    g = GAP(ystar); g.calculate_GAP(t, task)
    p = PA_GAP(ystar); p.calculate_PA_GAP(t, task)
    return g.GAP_value, p.PA_GAP_value, float(t[-1])


def run_qcbo(label, clusters, seed, trials, ninit):
    csv = os.path.join(RUNDIR, f"{label}_seed{seed}.csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv):
        tr = pd.read_csv(csv)["current_optimal"].tolist()
        if len(tr) >= trials:
            return tr
    benchmark.run_qcbo_benchmark(
        DS, coarse_clusters=clusters, seed=seed, num_trials=trials,
        num_interventions=ninit, max_intervention_size=5, out_csv=csv,
        method_label=label, assumed_graph_name=DS, gating=True)
    return pd.read_csv(csv)["current_optimal"].tolist()


def run_bo(seed, trials, ninit):
    csv = os.path.join(RUNDIR, f"BO_seed{seed}.csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv):
        tr = pd.read_csv(csv)["current_optimal"].tolist()
        if len(tr) >= trials:
            return tr
    from baselines.BO_CBO.BO import NonCausal_BO
    np.random.seed(seed)
    graph, obs, functions, config = misspec_inject.build_cbo_graph(DS, seed, "P0")
    manip = list(config["intervention"])
    task = config["task"]
    ranges = graph.get_interventional_ranges()
    dict_ranges = {v: (ranges[v][0], ranges[v][1]) for v in manip}
    costs = graph.get_cost_structure(1)
    dxl, dyl, bx, oy, _ = benchmark._initial_interventional_data(
        graph, [manip], ninit, task, seed)
    NonCausal_BO(trials, graph, dict_ranges, dxl[0], dyl[0], costs, obs,
                 functions, bx, oy, manip, Causal_prior=False, task=task,
                 csv_log_file=csv)
    return pd.read_csv(csv)["current_optimal"].tolist()


def run_unit(method, seed, trials, ninit):
    t0 = time.time()
    try:
        if method == "BO":
            tr = run_bo(seed, trials, ninit)
        elif method == "QCBO-coarse":
            tr = run_qcbo("QCBO-coarse", COARSE, seed, trials, ninit)
        else:  # CBO == QCBO-finest
            tr = run_qcbo("QCBO-finest", None, seed, trials, ninit)
        return dict(method=method, seed=seed, ok=True, traj=tr,
                    secs=time.time() - t0)
    except Exception as e:
        import traceback
        return dict(method=method, seed=seed, ok=False,
                    err=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()[-600:], secs=time.time() - t0)


def _sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--ninit", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=min(10, (os.cpu_count() or 8) - 2))
    args = ap.parse_args()

    ybest = json.load(open(benchmark._paths(DS)["ybest"]))
    ystar = float(ybest["global_best_value"])
    price_oracle = float(ybest.get("price_of_coarsening", float("nan")))
    task = benchmark.load_config(DS)["task"]
    methods = ["BO", "CBO", "QCBO-finest", "QCBO-coarse"]

    units = [(m, s) for m in methods for s in range(args.seeds)
             if not (m == "CBO")]  # CBO reuses QCBO-finest trajectory (Prop. 1)
    print(f"ClusterBench10L: {len(units)} units, y*={ystar:.4f}, "
          f"oracle price={price_oracle:.3f}", flush=True)

    traj = {m: {} for m in methods}
    fails = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_unit, m, s, args.trials, args.ninit)
                for (m, s) in units]
        for f in as_completed(futs):
            r = f.result()
            if r["ok"]:
                traj[r["method"]][r["seed"]] = r["traj"]
                print(f"OK   {r['method']:12s} seed{r['seed']} "
                      f"finalY={r['traj'][-1]:+.3f} ({r['secs']:.0f}s)", flush=True)
            else:
                fails.append(r)
                print(f"FAIL {r['method']:12s} seed{r['seed']} :: {r['err']}",
                      flush=True)
                print(r.get("tb", ""), flush=True)

    # CBO == QCBO-finest (Prop. 1): reuse its trajectories byte-for-byte.
    traj["CBO"] = dict(traj["QCBO-finest"])

    out = {"dataset": DS, "y_star": ystar, "task": task,
           "price_of_coarsening_oracle": price_oracle,
           "trials": args.trials, "seeds": args.seeds, "methods": {}}
    print(f"\n{'method':13s} {'finalY':>9s} {'GAP@20':>7s} {'GAP@50':>7s} "
          f"{'GAP@100':>7s} {'PA-GAP':>7s} {'rT':>7s} {'RT':>8s}")
    for m in methods:
        fin, g20, g50, g100, pag, rT, RT = ([] for _ in range(7))
        for s, tr in traj[m].items():
            t = tr[: args.trials + 1]
            g, p, f_ = score(t, ystar, task, args.trials)
            fin.append(f_); g100.append(g); pag.append(p)
            g20.append(score(t, ystar, task, min(20, args.trials))[0])
            g50.append(score(t, ystar, task, min(50, args.trials))[0])
            rT.append(float(simple_regret(t, ystar, task)[-1]))
            RT.append(cumulative_regret(t, ystar, task))
        if not fin:
            continue
        out["methods"][m] = {
            "finalY": [float(np.mean(fin)), _sem(fin)],
            "GAP20": [float(np.mean(g20)), _sem(g20)],
            "GAP50": [float(np.mean(g50)), _sem(g50)],
            "GAP100": [float(np.mean(g100)), _sem(g100)],
            "PAGAP100": [float(np.mean(pag)), _sem(pag)],
            "regretT": [float(np.mean(rT)), _sem(rT)],
            "cumRegret": [float(np.mean(RT)), _sem(RT)]}
        print(f"{m:13s} {np.mean(fin):+9.3f} {np.mean(g20):7.3f} "
              f"{np.mean(g50):7.3f} {np.mean(g100):7.3f} {np.mean(pag):7.3f} "
              f"{np.mean(rT):7.3f} {np.mean(RT):8.2f}")

    # Realized price = QCBO-coarse plateau minus y* (final incumbent gap).
    if "QCBO-coarse" in out["methods"]:
        realized = out["methods"]["QCBO-coarse"]["finalY"][0] - ystar
        out["price_of_coarsening_realized"] = realized
        print(f"\nrealized price of coarsening (coarse finalY - y*) = "
              f"{realized:+.3f}  (oracle {price_oracle:.3f})")

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "clusterbench10L.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    if fails:
        print(f"FAILURES: {len(fails)}")


if __name__ == "__main__":
    main()
