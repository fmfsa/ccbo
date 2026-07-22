"""Damage experiment on ClusterBench10D (deleted-optimal-arm variant).

ClusterBench10D shares ClusterBench10's structure but its global optimum is the
singleton do(X1) (also attained by the cluster arm do(C1), so the coarse
partition stays lossless). The intra-cluster bow P1 (add X1->X2 on the
X1<->X2-confounded pair) therefore costs the cheap 1-d optimal arm at the
finest partition: full-DAG CBO must reach the same optimum through the
higher-dimensional do(X1,X2)/do(C1) arms -- a real convergence cost (dGAP < 0)
-- while QCBO-coarse is invariant (Prop. 2, intra edit).

Same fixed-objective protocol as scripts/run_misspec_fullfield.py: one
objective (the true ClusterBench10D SEM), shared seeds/budget, only the assumed
structure per method and perturbation. Writes results/clusterbench10D.json.

Run:
  PYTHONPATH=.:scripts conda run -n ccbo python scripts/run_clusterbench10D.py \
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
import misspec_inject  # noqa: F401  (kept for parity with the L runner's BO arm)

from ccbo.metrics import gap

DS = "ClusterBench10D"
RUNDIR = os.path.join(benchmark.BENCH_ROOT, "results", "_damage")
COARSE = [list(c) for c in cb.COARSE_CLUSTERS]
PIDS = ["P0", "P1"]


def score(traj, ystar, task, n):
    t = list(traj)[: n + 1]
    return gap(t, ystar, task), float(t[-1])


def run_qcbo(label, clusters, pid, seed, trials, ninit):
    csv = os.path.join(RUNDIR, f"{label}_{pid}_seed{seed}.csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv):
        tr = pd.read_csv(csv)["current_optimal"].tolist()
        if len(tr) >= trials:
            return tr
    benchmark.run_qcbo_benchmark(
        DS, coarse_clusters=clusters, seed=seed, num_trials=trials,
        num_interventions=ninit, out_csv=csv,
        method_label=label, assumed_graph_name=cb.variant_name(pid, prefix=DS))
    return pd.read_csv(csv)["current_optimal"].tolist()


def run_bo(seed, trials, ninit):
    csv = os.path.join(RUNDIR, f"BO_P0_seed{seed}.csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if os.path.exists(csv):
        tr = pd.read_csv(csv)["current_optimal"].tolist()
        if len(tr) >= trials:
            return tr
    from ccbo.cbo.bo import NonCausal_BO
    np.random.seed(seed)
    graph, obs, functions, config = misspec_inject.build_cbo_graph(DS, seed, "P0")
    manip = list(config["intervention"])
    task = config["task"]
    if task != "min":
        raise NotImplementedError("vendored NonCausal_BO is minimization-only")
    ranges = graph.get_interventional_ranges()
    dict_ranges = {v: (ranges[v][0], ranges[v][1]) for v in manip}
    costs = graph.get_cost_structure(1)
    dxl, dyl, bx, oy, _ = benchmark._initial_interventional_data(
        graph, [manip], ninit, task, seed)
    _, _, best_y, _ = NonCausal_BO(
        trials, graph, dict_ranges, dxl[0], dyl[0], costs, obs,
        functions, bx, oy, manip, Causal_prior=False)
    traj0 = np.minimum.accumulate(
        np.asarray(best_y, dtype=float).ravel()).tolist()
    benchmark._write_progress_csv(csv, traj0)
    return traj0


def run_unit(method, pid, seed, trials, ninit):
    cb.register_variants(prefix=DS)
    t0 = time.time()
    try:
        if method == "BO":
            tr = run_bo(seed, trials, ninit)
        elif method == "QCBO-coarse":
            tr = run_qcbo("QCBO-coarse", COARSE, pid, seed, trials, ninit)
        else:  # QCBO-finest (CBO reuses it, Prop. 1)
            tr = run_qcbo("QCBO-finest", None, pid, seed, trials, ninit)
        return dict(method=method, pid=pid, seed=seed, ok=True, traj=tr,
                    secs=time.time() - t0)
    except Exception as e:
        import traceback
        return dict(method=method, pid=pid, seed=seed, ok=False,
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

    cb.register_variants(prefix=DS)
    ybest = json.load(open(benchmark._paths(DS)["ybest"]))
    ystar = float(ybest["global_best_value"])
    task = benchmark.load_config(DS)["task"]
    methods = ["BO", "CBO", "QCBO-finest", "QCBO-coarse"]

    units = []
    for m in methods:
        if m == "CBO":
            continue  # CBO reuses QCBO-finest trajectories (Prop. 1)
        for pid in (["P0"] if m == "BO" else PIDS):
            units.extend((m, pid, s) for s in range(args.seeds))
    print(f"ClusterBench10D: {len(units)} units, y*={ystar:.4f}", flush=True)

    traj = {m: {p: {} for p in PIDS} for m in methods}
    fails = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_unit, m, p, s, args.trials, args.ninit)
                for (m, p, s) in units]
        for f in as_completed(futs):
            r = f.result()
            if r["ok"]:
                traj[r["method"]][r["pid"]][r["seed"]] = r["traj"]
                print(f"OK   {r['method']:12s} {r['pid']:3s} seed{r['seed']} "
                      f"finalY={r['traj'][-1]:+.3f} ({r['secs']:.0f}s)", flush=True)
            else:
                fails.append(r)
                print(f"FAIL {r['method']:12s} {r['pid']:3s} seed{r['seed']} :: "
                      f"{r['err']}", flush=True)
                print(r.get("tb", ""), flush=True)

    # BO is structure-blind (invariant); CBO == QCBO-finest (Prop. 1).
    traj["BO"]["P1"] = dict(traj["BO"]["P0"])
    traj["CBO"] = {p: dict(traj["QCBO-finest"][p]) for p in PIDS}

    out = {"dataset": DS, "y_star": ystar, "task": task,
           "trials": args.trials, "seeds": args.seeds, "methods": {}}
    print(f"\n{'method':13s} {'pert':4s} {'finalY':>9s} {'dFinalY':>9s} "
          f"{'GAP@100':>8s} {'dGAP':>8s} {'ident':>7s}")
    for m in methods:
        out["methods"][m] = {}
        base = traj[m]["P0"]
        for pid in PIDS:
            fin, dfin, gaps, dgap, ident = [], [], [], [], []
            rTs, cums = [], []
            for s in range(args.seeds):
                if s not in traj[m][pid] or s not in base:
                    continue
                t, t0 = traj[m][pid][s], base[s]
                g, f_ = score(t, ystar, task, args.trials)
                g0, f0 = score(t0, ystar, task, args.trials)
                fin.append(f_); dfin.append(f_ - f0)
                gaps.append(g); dgap.append(g - g0)
                rTs.append(float(simple_regret(
                    t[: args.trials + 1], ystar, task)[-1]))
                cums.append(cumulative_regret(t[: args.trials + 1], ystar, task))
                same_len = len(t) == len(t0)
                ident.append(same_len and max(
                    (abs(a - b) for a, b in zip(t, t0)), default=0.0) < 1e-7)
            if not fin:
                continue
            cell = {"finalY": [float(np.mean(fin)), _sem(fin)],
                    "dFinalY": [float(np.mean(dfin)), _sem(dfin)],
                    "GAP100": [float(np.mean(gaps)), _sem(gaps)],
                    "dGAP": [float(np.mean(dgap)), _sem(dgap)],
                    "regretT": [float(np.mean(rTs)), _sem(rTs)],
                    "cumRegret": [float(np.mean(cums)), _sem(cums)],
                    "identical_to_P0": all(ident)}
            out["methods"][m][pid] = cell
            print(f"{m:13s} {pid:4s} {np.mean(fin):+9.3f} {np.mean(dfin):+9.3f} "
                  f"{np.mean(gaps):8.3f} {np.mean(dgap):+8.3f} "
                  f"{str(all(ident)):>7s}")

    coarse_p1 = out["methods"].get("QCBO-coarse", {}).get("P1", {})
    if coarse_p1:
        assert coarse_p1["identical_to_P0"], \
            "Prop.2 VIOLATION: coarse not identical under the intra bow"
        print("\nPREMISE CHECK  QCBO-coarse P1 identical: OK  <- Prop.2")

    os.makedirs("results", exist_ok=True)
    outpath = os.path.join("results", "clusterbench10D.json")
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {outpath}")
    if fails:
        print(f"FAILURES: {len(fails)}")
    print("=== CLUSTERBENCH10D DONE ===", flush=True)


if __name__ == "__main__":
    main()
