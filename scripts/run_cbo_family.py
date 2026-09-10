"""Family comparison runner: CBO (identity partition) vs QCBO (coarse partition).

Runs {ToyGraph, CompleteGraph, SimplifiedCoralGraph} x {CBO, QCBO} x seeds
through the single shared CoarsenedGraph + CBO backend (the identity partition
IS full-DAG CBO by Prop. 1, so both arms differ only in the partition), and
persists one CSV per (dataset, arm, seed):

    <outdir>/<dataset>_<arm>_seed<seed>.csv    columns: trial, best_y, cum_cost, arm, x_values
    <outdir>/<dataset>_<arm>_seed<seed>.decisions.json   full-precision decision log (engine v3)

Row convention: the vendored CBO loop appends to ``global_opt`` and
``current_cost`` in lockstep — one initial row (trial 0: the incumbent from the
initial interventional design, cost 0.0) plus exactly one row per trial
(observe trials repeat the previous value of both), so each CSV has
``trials + 1`` rows and ``best_y[i]`` / ``cum_cost[i]`` are aligned
index-for-index.

Arms
----
CBO   : CoarsenedGraph at the identity partition (all-singleton clusters).
QCBO  : the benchmark's domain coarse partition (mirrors the representative
        coarsenings in run_experiments.py):
          ToyGraph              {X,Z} | {Y}
          CompleteGraph         {B} | {D,E} | {Y}
          SimplifiedCoralGraph  {C,N,O} | {D,T} | {Y}

Exploration sets are the MIS of each arm's (C-)DAG; identifiability from
that graph decides each arm's *prior tier*, never its membership.  On
ToyGraph's QCBO arm the {X,Z} cluster forms a bow with Y through the latent
U, so ``do({X,Z})`` is not identifiable from the C-DAG and the arm runs on
the common uninformative prior (observational mean/var of Y) from
``CoarsenedGraph.get_all_do``; identifiable arms get do-calculus priors.

Protocol: unit costs (type_cost=1), 100 initial observational samples,
10 initial interventional points per arm, task = min — matching
run_experiments.py's main condition.

Run (from repo root):
    PYTHONPATH=. python scripts/run_cbo_family.py \
        --datasets ToyGraph,CompleteGraph,SimplifiedCoralGraph \
        --seeds 10 --trials 40 --outdir results/family_cbo --jobs 12

Resumable: a unit whose CSV already exists with the expected number of rows is
skipped, so a relaunch only fills the missing units.
"""

import os

# Cap BLAS/OpenMP threads to 1 *before* numpy import so every worker is
# single-threaded and many units run in parallel (and fork stays safe).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import csv
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


DATASETS = ("ToyGraph", "CompleteGraph", "SimplifiedCoralGraph")
ARMS = ("BO", "CBO", "QCBO")

# Partitions per (dataset, arm). CBO = identity on M (Prop. 1); QCBO = the
# domain coarse partition (the representative coarsenings in
# run_experiments.py lines 100-147).
PARTITIONS = {
    ("ToyGraph", "CBO"): [
        frozenset({"X"}), frozenset({"Z"}), frozenset({"Y"})],
    ("ToyGraph", "QCBO"): [
        frozenset({"X", "Z"}), frozenset({"Y"})],
    ("CompleteGraph", "CBO"): [
        frozenset({"B"}), frozenset({"D"}), frozenset({"E"}), frozenset({"Y"})],
    ("CompleteGraph", "QCBO"): [
        frozenset({"B"}), frozenset({"D", "E"}), frozenset({"Y"})],
    ("SimplifiedCoralGraph", "CBO"): [
        frozenset({"C"}), frozenset({"D"}), frozenset({"N"}),
        frozenset({"O"}), frozenset({"T"}), frozenset({"Y"})],
    ("SimplifiedCoralGraph", "QCBO"): [
        frozenset({"C", "N", "O"}), frozenset({"D", "T"}), frozenset({"Y"})],
}

# Standard BO reads no graph structure beyond the SEM, the intervention
# ranges, and the costs, so its partition is bookkeeping only: reuse the
# identity one and replace the exploration set with a single joint arm over
# every manipulable variable. Consequence for the cost axis: BO pays
# len(manip_vars) per trial under unit costs, against a causal method's
# per-arm cost.
for _ds in DATASETS:
    PARTITIONS[(_ds, "BO")] = PARTITIONS[(_ds, "CBO")]


def _csv_path(outdir, dataset, arm, seed):
    return os.path.join(outdir, f"{dataset}_{arm}_seed{seed}.csv")


def _load_graph(dataset, initial_num_obs_samples):
    """Load observational data and instantiate the fine-grained graph."""
    from ccbo.run_experiment import get_original_graph

    data_path = os.path.join(_REPO, "ccbo", "cbo", "data", dataset)
    full_obs = pd.read_pickle(os.path.join(data_path, "observations.pkl"))
    obs = full_obs[:initial_num_obs_samples]

    true_obs = None
    if dataset in ("CoralGraph", "SimplifiedCoralGraph"):
        true_obs = pd.read_pickle(
            os.path.join(data_path, "true_observations.pkl"))

    graph = get_original_graph(dataset, obs, true_obs)
    return graph, obs, full_obs


def run_unit(dataset, arm, seed, trials, num_interventions,
             initial_num_obs_samples, type_cost, outdir):
    """One (dataset, arm, seed) run; writes the per-unit CSV.

    Resumable: if a complete CSV (trials + 1 rows) already exists it is kept
    and the run is skipped.
    """
    t0 = time.time()
    csv_path = _csv_path(outdir, dataset, arm, seed)
    unit = dict(dataset=dataset, arm=arm, seed=seed, csv=csv_path)

    from ccbo.decision_log import sidecar_path
    if os.path.exists(csv_path) and os.path.exists(sidecar_path(csv_path)):
        try:
            n_rows = len(pd.read_csv(csv_path))
        except Exception:
            n_rows = -1
        if n_rows >= trials + 1:
            unit.update(ok=True, resumed=True, secs=0.0)
            return unit

    try:
        from ccbo.coarsened_graph import CoarsenedGraph
        from ccbo.data_generation import generate_interventional_data
        from ccbo.cbo.cbo import CBO
        from ccbo.cbo.utils import compute_coverage, define_initial_data_CBO
        from ccbo.minimal_suite import init_design_cost
        from ccbo import decision_log as dlog

        np.random.seed(seed)
        graph, obs, full_obs = _load_graph(dataset, initial_num_obs_samples)

        partition = PARTITIONS[(dataset, arm)]
        cg = CoarsenedGraph(graph, partition, dataset, obs,
                            num_mc_samples=2000)
        MIS, _, manip_vars = cg.get_sets()
        if arm == "BO":
            MIS = [list(manip_vars)]

        functions = cg.fit_all_models()
        dict_ranges = cg.get_interventional_ranges()
        costs = cg.get_cost_structure(type_cost)

        np.random.seed(seed)
        int_data = generate_interventional_data(
            cg.define_SEM, MIS, dict_ranges,
            num_points=20, num_mc_samples=2000, seed=seed)

        _, _, coverage = compute_coverage(obs, manip_vars, dict_ranges)
        x_list, y_list, best_x, opt_y, best_var = define_initial_data_CBO(
            int_data, num_interventions, MIS, 0, "min")
        init_cost = init_design_cost(MIS, x_list, costs)   # engine v3: charged
        max_N = initial_num_obs_samples + 50

        if arm == "BO":
            from ccbo.cbo.bo import NonCausal_BO
            cost_arr, current_best_x, best_y_arr, total_time, trial_log = NonCausal_BO(
                trials, cg, dict_ranges, x_list[0], y_list[0], costs, obs,
                functions, best_x, opt_y, list(manip_vars),
                Causal_prior=False, task="min", return_log=True)
            prior_mask = [False]
            global_opt = [float(v) for v in np.asarray(
                best_y_arr, dtype=float).ravel()]
            current_cost = [float(v) for v in np.asarray(
                cost_arr, dtype=float).ravel()]
            current_best_y = {"".join(manip_vars): global_opt}
            current_best_x = {"".join(manip_vars): [
                np.ravel(current_best_x)]}
            observed = 0
        else:
            (current_cost, current_best_x, current_best_y, global_opt,
             observed, total_time), state = CBO(
                trials, MIS, manip_vars, x_list, y_list, best_x, opt_y,
                best_var, dict_ranges, functions, obs, coverage, cg,
                20, costs, full_obs, "min",
                max_N, initial_num_obs_samples,
                num_interventions, Causal_prior=True, return_state=True)
            trial_log = state["trial_log"]
            prior_mask = state["prior_mask"]

        # The loop appends to global_opt and current_cost in lockstep:
        # 1 initial row + 1 row per trial.
        assert len(global_opt) == len(current_cost) == trials + 1, (
            f"trajectory length mismatch: |global_opt|={len(global_opt)} "
            f"|current_cost|={len(current_cost)} expected {trials + 1}")

        # Engine v3 rows: (trial, best_y, cum_cost, arm, x_values); row 0 is
        # the initial design at cum_cost = init_cost; decisions from the
        # full-precision trial log (observe rows blank).
        rows = [(0, float(global_opt[0]), float(init_cost), "init", "")]
        for i, e in enumerate(trial_log):
            if e.get("type") == "intervene":
                arm_lbl = e["arm"]
                xv = ";".join(repr(float(v)) for v in e["x"])
            else:
                arm_lbl, xv = "", ""
            rows.append((i + 1, float(global_opt[i + 1]),
                         float(current_cost[i + 1]) + float(init_cost),
                         arm_lbl, xv))
        assert len(rows) == trials + 1
        for e in trial_log:
            e["cum_cost"] = float(e["cum_cost"]) + float(init_cost)

        tmp_path = csv_path + ".tmp"
        with open(tmp_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["trial", "best_y", "cum_cost", "arm", "x_values"])
            for (t, y, c, a, xv) in rows:
                writer.writerow([t, repr(float(y)), repr(float(c)), a, xv])
        os.replace(tmp_path, csv_path)

        gate = {"+".join(k): v for k, v in getattr(cg, "_arm_gate_status", {}).items()}
        payload = dlog.build_payload(
            unit=dict(suite="family", dataset=dataset, cond="", arm=arm,
                      seed=int(seed), trials=int(trials),
                      n_init=int(num_interventions),
                      n_obs=int(initial_num_obs_samples),
                      type_cost=int(type_cost),
                      max_N=int(initial_num_obs_samples if arm == "BO" else max_N)),
            init=dict(partition=[sorted(p) for p in partition],
                      es=[list(e) for e in MIS], prior_mask=list(prior_mask),
                      x_list=[np.asarray(x, dtype=float).tolist() for x in x_list],
                      y_list=[np.asarray(y, dtype=float).ravel().tolist() for y in y_list],
                      init_cost=float(init_cost), incumbent=float(opt_y),
                      gate_status=gate),
            phases=[dlog.phase_record("main", MIS, prior_mask, trial_log)],
            rows=rows)
        dlog.write(sidecar_path(csv_path), payload)

        # Best arm so far (diagnostic only; not persisted).
        best_arm = min(current_best_y, key=lambda k: np.min(current_best_y[k]))
        best_idx = int(np.argmin(current_best_y[best_arm]))
        best_x = np.ravel(current_best_x[best_arm][
            0 if arm == "BO" else best_idx]).tolist()
        uninformative = [] if arm == "BO" else sorted(
            "".join(arm_vars) for arm_vars, ok_id
            in cg._arm_identifiable.items() if not ok_id)
        unit.update(
            ok=True, resumed=False, secs=time.time() - t0,
            final_y=float(global_opt[-1]), final_cost=float(rows[-1][2]),
            init_cost=float(init_cost),
            observed=int(observed), best_arm=best_arm, best_x=best_x,
            uninformative_arms=uninformative,
            partition=cg.get_partition_description(),
            es=cg.get_exploration_set_description())
        return unit
    except Exception as e:
        import traceback
        unit.update(ok=False, resumed=False, secs=time.time() - t0,
                    err=f"{type(e).__name__}: {e}\n"
                        f"{traceback.format_exc()[-800:]}")
        return unit


def main():
    ap = argparse.ArgumentParser(
        description="CBO (identity) vs QCBO (coarse) family runner")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help=f"Comma-separated subset of {DATASETS}")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"Comma-separated subset of {ARMS}")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Number of seeds (runs seeds 0..N-1)")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--num-interventions", type=int, default=10,
                    help="Initial interventional points per arm")
    ap.add_argument("--initial-obs", type=int, default=100,
                    help="Initial observational samples")
    ap.add_argument("--type-cost", type=int, default=1,
                    help="Cost structure (1 = unit costs)")
    ap.add_argument("--outdir", default="results/family_cbo")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Process-parallel workers (each single-threaded)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the expanded unit grid and exit.")
    args = ap.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    arms = [a for a in args.arms.split(",") if a]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"Unknown dataset {d!r}; choose from {DATASETS}")
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"Unknown arm {a!r}; choose from {ARMS}")

    os.makedirs(args.outdir, exist_ok=True)

    units = [(d, a, s, args.trials, args.num_interventions,
              args.initial_obs, args.type_cost, args.outdir)
             for d in datasets for a in arms for s in range(args.seeds)]

    print(f"family runner: {len(units)} units | datasets={datasets} "
          f"arms={arms} seeds={args.seeds} T={args.trials} "
          f"ninit={args.num_interventions} jobs={args.jobs}", flush=True)

    if args.dry_run:
        from collections import Counter
        for (d, a), n in sorted(Counter((u[0], u[1]) for u in units).items()):
            print(f"  {d:<22s} {a:<5s} {n:4d} units", flush=True)
        print(f"  {'TOTAL':<22s} {'':<5s} {len(units):4d} units "
              f"-> {args.outdir}", flush=True)
        return

    results, done = [], 0

    def _report(r):
        nonlocal done
        done += 1
        tag = f"{r['dataset']}/{r['arm']}/seed{r['seed']}"
        if not r["ok"]:
            print(f"[{done:3d}/{len(units)}] FAIL {tag} :: "
                  f"{r['err'].splitlines()[0]}", flush=True)
        elif r.get("resumed"):
            print(f"[{done:3d}/{len(units)}] SKIP {tag} (complete CSV exists)",
                  flush=True)
        else:
            uninf = r.get("uninformative_arms") or []
            note = (f" [uninformative prior: {','.join(uninf)}]"
                    if uninf else "")
            bx = ",".join(f"{v:.2f}" for v in r["best_x"])
            print(f"[{done:3d}/{len(units)}] OK   {tag} "
                  f"finalY={r['final_y']:+.3f} cost={r['final_cost']:.1f} "
                  f"bestArm={r['best_arm']}@[{bx}] obs={r['observed']} "
                  f"({r['secs']:.0f}s){note}", flush=True)

    if args.jobs <= 1:
        for u in units:
            r = run_unit(*u)
            results.append(r)
            _report(r)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_unit, *u) for u in units]
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                _report(r)

    failures = [r for r in results if not r["ok"]]
    print(f"\ndone: {len(results) - len(failures)}/{len(results)} units OK; "
          f"CSVs in {args.outdir}", flush=True)
    if failures:
        for r in failures:
            print(f"  FAILED {r['dataset']}/{r['arm']}/seed{r['seed']}: "
                  f"{r['err'].splitlines()[0]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
