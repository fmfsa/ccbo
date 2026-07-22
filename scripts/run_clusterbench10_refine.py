"""QCBO-refine experiment: plateau-triggered refinement on ClusterBench10L
(the lossy variant, where it must escape the coarse plateau) plus the lossless
ClusterBench10 control (where firing the trigger must cost ~nothing).

Reuses the existing baseline CSVs:
  * 10L:     third_party/.../results/_lossy/{BO,QCBO-coarse,QCBO-finest}_seed*.csv
  * control: third_party/.../results/_misspec/{QCBO-coarse,QCBO-finest}_P0_seed*.csv
New units are QCBO-refine x seeds on each dataset. C1 = {X1,X2,X3} refines to
singletons (the agnostic nested level); trigger k=10, delta=1e-3, chunk=10.

Writes results/clusterbench10_refine.json (stats + mean best-so-far curves +
trigger steps) and adds a "QCBO-refine" entry to results/clusterbench10L.json.

Run:
  PYTHONPATH=.:scripts conda run -n ccbo python scripts/run_clusterbench10_refine.py \
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

from ccbo.metrics import gap

COARSE = [list(c) for c in cb.COARSE_CLUSTERS]
REFINE_MAP = {tuple(cb.COARSE_CLUSTERS[0]):
              [[v] for v in cb.COARSE_CLUSTERS[0]]}

# Trigger setting: tuned once on ClusterBench10L seed 0 (sweep over
# k in {5,10,15} x delta in {1e-3,1e-2}; every config converged to y*,
# trigger step is the only thing that moves), then held fixed across every
# dataset and seed. The monitor is evaluated every step on the phase-1
# trajectory (phase 1 is byte-identical to the QCBO-coarse baseline).
TRIGGER = dict(plateau_k=5, plateau_delta=1e-3)

LOSSY = dict(ds="ClusterBench10L", seeds=10,
             coarse=COARSE, rmap=REFINE_MAP, assumed="ClusterBench10L",
             rundir=os.path.join(benchmark.BENCH_ROOT, "results", "_lossy"),
             csv_fmt="{label}_seed{seed}.csv")
CONTROL = dict(ds="ClusterBench10", seeds=10,
               coarse=COARSE, rmap=REFINE_MAP, assumed=cb.variant_name("P0"),
               rundir=os.path.join(benchmark.BENCH_ROOT, "results", "_misspec"),
               csv_fmt="{label}_P0_seed{seed}.csv")


def _suite(ds, coarse, rmap, seeds=5):
    """Survey-suite dataset: baselines live under results/<label>/<ds>/."""
    return dict(ds=ds, seeds=seeds, coarse=coarse, rmap=rmap, assumed=None,
                rundir=os.path.join(benchmark.BENCH_ROOT, "results"),
                csv_fmt="{label}/" + ds + "/{label}_" + ds +
                        "_100-trials_seed{seed}_progress.csv")


SYNTH2 = _suite("synthetic_2", [["X", "Z"]], {("X", "Z"): [["X"], ["Z"]]})
ECOLOGY = _suite("ecology", [["C", "N", "O"], ["D", "T"]],
                 {("C", "N", "O"): [["C"], ["N"], ["O"]],
                  ("D", "T"): [["D"], ["T"]]})
HEALTHCARE = _suite("healthcare", [["Aspirin", "Statin"]],
                    {("Aspirin", "Statin"): [["Aspirin"], ["Statin"]]})


def score(traj, ystar, task, n):
    t = list(traj)[: n + 1]
    return gap(t, ystar, task), float(t[-1])


def run_unit(setting, seed, trials, ninit):
    t0 = time.time()
    csv = os.path.join(setting["rundir"],
                       setting["csv_fmt"].format(label="QCBO-refine", seed=seed))
    info_path = csv.replace(".csv", "_info.json")
    try:
        if os.path.exists(csv) and os.path.exists(info_path):
            tr = pd.read_csv(csv)["current_optimal"].tolist()
            if len(tr) >= trials:
                return dict(ds=setting["ds"], seed=seed, ok=True, traj=tr,
                            info=json.load(open(info_path)), secs=0.0)
        os.makedirs(os.path.dirname(csv), exist_ok=True)
        coarse_csv = os.path.join(
            setting["rundir"],
            setting["csv_fmt"].format(label="QCBO-coarse", seed=seed))
        _, _, info = benchmark.run_qcbo_refine_benchmark(
            setting["ds"], setting["coarse"], setting["rmap"], seed=seed,
            num_trials=trials, num_interventions=ninit,
            out_csv=csv,
            assumed_graph_name=setting["assumed"],
            coarse_csv=coarse_csv, **TRIGGER)
        with open(info_path, "w") as f:
            json.dump(info, f)
        tr = pd.read_csv(csv)["current_optimal"].tolist()
        return dict(ds=setting["ds"], seed=seed, ok=True, traj=tr, info=info,
                    secs=time.time() - t0)
    except Exception as e:
        import traceback
        return dict(ds=setting["ds"], seed=seed, ok=False,
                    err=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()[-600:], secs=time.time() - t0)


def _sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def _baseline_traj(setting, label, seed):
    p = os.path.join(setting["rundir"],
                     setting["csv_fmt"].format(label=label, seed=seed))
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)["current_optimal"].astype(float).tolist()


def _stats(trajs, ystar, task, trials):
    fin, g100, rT, RT = [], [], [], []
    for tr in trajs:
        t = tr[: trials + 1]
        g, f_ = score(t, ystar, task, trials)
        fin.append(f_); g100.append(g)
        rT.append(float(simple_regret(t, ystar, task)[-1]))
        RT.append(cumulative_regret(t, ystar, task))
    return {"finalY": [float(np.mean(fin)), _sem(fin)],
            "GAP100": [float(np.mean(g100)), _sem(g100)],
            "regretT": [float(np.mean(rT)), _sem(rT)],
            "cumRegret": [float(np.mean(RT)), _sem(RT)]}


def _mean_curve(trajs, trials):
    # BO's harness logs one fewer step than the CBO loop; truncate every
    # trajectory to the method's common length rather than dropping any.
    n = min(min(len(t) for t in trajs), trials + 1)
    arr = np.array([t[:n] for t in trajs], dtype=float)
    return {"mean": arr.mean(axis=0).tolist(),
            "sem": (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])).tolist()
            if arr.shape[0] > 1 else [0.0] * n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--ninit", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=min(10, (os.cpu_count() or 8) - 2))
    ap.add_argument("--datasets",
                    default="lossy,control,synthetic_2,ecology,healthcare")
    args = ap.parse_args()

    settings = {"lossy": LOSSY, "control": CONTROL, "synthetic_2": SYNTH2,
                "ecology": ECOLOGY, "healthcare": HEALTHCARE}
    todo = [settings[k] for k in args.datasets.split(",")]

    units = [(st, s) for st in todo for s in range(st["seeds"])]
    print(f"QCBO-refine: {len(units)} units "
          f"({', '.join(st['ds'] for st in todo)})", flush=True)

    results = {st["ds"]: {} for st in todo}
    fails = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_unit, st, s, args.trials, args.ninit)
                for (st, s) in units]
        for f in as_completed(futs):
            r = f.result()
            if r["ok"]:
                results[r["ds"]][r["seed"]] = r
                print(f"OK   {r['ds']:16s} seed{r['seed']} "
                      f"finalY={r['traj'][-1]:+.3f} "
                      f"trigger={r['info'].get('trigger_step')} "
                      f"({r['secs']:.0f}s)", flush=True)
            else:
                fails.append(r)
                print(f"FAIL {r['ds']:16s} seed{r['seed']} :: {r['err']}",
                      flush=True)
                print(r.get("tb", ""), flush=True)

    out = {"trials": args.trials,
           "seeds": {st["ds"]: st["seeds"] for st in todo},
           "trigger": {"k": TRIGGER["plateau_k"],
                       "delta": TRIGGER["plateau_delta"]},
           "datasets": {}}

    for st in todo:
        ds = st["ds"]
        ybest = json.load(open(benchmark._paths(ds)["ybest"]))
        ystar = float(ybest["global_best_value"])
        task = benchmark.load_config(ds)["task"]
        seeds_done = sorted(results[ds])
        ref_trajs = [results[ds][s]["traj"] for s in seeds_done]
        if not ref_trajs:
            continue
        entry = {"y_star": ystar, "task": task,
                 "trigger_steps": {s: results[ds][s]["info"].get("trigger_step")
                                   for s in seeds_done},
                 "methods": {"QCBO-refine":
                             _stats(ref_trajs, ystar, task, args.trials)},
                 "curves": {"QCBO-refine": _mean_curve(ref_trajs, args.trials)}}
        for label in ("BO", "CBO", "QCBO-coarse", "QCBO-finest"):
            trajs = [t for t in (_baseline_traj(st, label, s)
                                 for s in range(st["seeds"])) if t]
            if trajs:
                entry["methods"][label] = _stats(trajs, ystar, task, args.trials)
                entry["curves"][label] = _mean_curve(trajs, args.trials)
        entry["price_realized"] = {
            m: entry["methods"][m]["finalY"][0] - ystar
            for m in entry["methods"]}
        out["datasets"][ds] = entry
        print(f"\n{ds}: y*={ystar:.4f}")
        for m, v in entry["methods"].items():
            print(f"  {m:13s} finalY={v['finalY'][0]:+.3f}"
                  f"±{v['finalY'][1]:.3f}  GAP100={v['GAP100'][0]:.3f}  "
                  f"price={entry['price_realized'][m]:+.3f}")

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "clusterbench10_refine.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")

    # Add the QCBO-refine row to the 10L price-experiment JSON (same scoring).
    lossy_path = os.path.join("results", "clusterbench10L.json")
    if os.path.exists(lossy_path) and "ClusterBench10L" in out["datasets"]:
        d = json.load(open(lossy_path))
        ref = out["datasets"]["ClusterBench10L"]["methods"]["QCBO-refine"]
        ystar = out["datasets"]["ClusterBench10L"]["y_star"]
        trajs = [results["ClusterBench10L"][s]["traj"]
                 for s in sorted(results["ClusterBench10L"])]
        g20 = [score(t, ystar, d["task"], min(20, args.trials))[0] for t in trajs]
        g50 = [score(t, ystar, d["task"], min(50, args.trials))[0] for t in trajs]
        d["methods"]["QCBO-refine"] = dict(
            ref, GAP20=[float(np.mean(g20)), _sem(g20)],
            GAP50=[float(np.mean(g50)), _sem(g50)],
            GAP100=ref["GAP100"])
        with open(lossy_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"Updated {lossy_path} with QCBO-refine")

    if fails:
        print(f"FAILURES: {len(fails)}")


if __name__ == "__main__":
    main()
