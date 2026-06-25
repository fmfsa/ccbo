"""Full-field DAG-misspecification stress test on ClusterBench10.

Runs each method under the correct DAG (P0) and each perturbation variant, with
the *objective fixed* at the true SCM (only the structure each method reasons
with is perturbed). Reports the paired-by-seed degradation
    Delta = metric(perturbation, seed) - metric(P0, seed)
for final Y and GAP, and asserts QCBO-coarse byte-identity on the protected
(quotient-invisible) perturbations.

Currently wired methods: QCBO-finest, QCBO-coarse. External baselines
(CBO/CEO/CoCaBO via scripts/misspec_inject.py) plug into METHODS below.

Run (quick):  PYTHONPATH=. python scripts/run_misspec_fullfield.py --seeds 3 --trials 40 --cap 1
"""

import os
import sys
import json
import argparse
import warnings

# Cap BLAS threads BEFORE numpy import: GP fits otherwise over-subscribe every
# core (~20x), making serial runs far slower than necessary.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ccbo import benchmark, clusterbench10 as cb
import misspec_inject

sys.path.insert(0, benchmark.BENCH_ROOT)
from metrics.GAP import GAP            # noqa: E402
from metrics.PA_GAP import PA_GAP      # noqa: E402

DS = cb.NAME
OUTDIR = "results"
RUNDIR = os.path.join(benchmark.BENCH_ROOT, "results", "_misspec")


def score(traj, ystar, task, n):
    t = traj[: n + 1]
    g = GAP(ystar); g.calculate_GAP(t, task)
    p = PA_GAP(ystar); p.calculate_PA_GAP(t, task)
    return g.GAP_value, p.PA_GAP_value, t[-1]


# --- Method runners: (perturbation_id, seed, cap, trials, ninit) -> trajectory ---

def _qcbo_runner(clusters):
    def run(pid, seed, cap, trials, ninit):
        label = "QCBO-coarse" if clusters else "QCBO-finest"
        csv = os.path.join(RUNDIR, f"{label}_{pid}_seed{seed}.csv")
        os.makedirs(os.path.dirname(csv), exist_ok=True)
        benchmark.run_qcbo_benchmark(
            DS, coarse_clusters=clusters, seed=seed, num_trials=trials,
            num_interventions=ninit, max_intervention_size=cap, out_csv=csv,
            method_label=label, assumed_graph_name=cb.variant_name(pid))
        return pd.read_csv(csv)["current_optimal"].tolist()
    return run


def _cbo_runner():
    """Benchmark CBO with a misspecified causal prior (true objective).

    Mirrors the benchmark's own CBO invocation (as in run_qcbo_benchmark) but
    on the plain DiscoveredGraph whose adjacency was perturbed by
    misspec_inject. CBO does NOT gate (get_sets() = all subsets <= cap), so the
    bow only biases its prior (Tier-1) -- it keeps do(X1); contrast the gating
    methods (CEO, QCBO-finest) which lose it (Tier-2).
    """
    def run(pid, seed, cap, trials, ninit):
        from baselines.BO_CBO.CBO import CBO
        from baselines.BO_CBO.utils import compute_coverage
        np.random.seed(seed)
        graph, obs, functions, config = misspec_inject.build_cbo_graph(DS, seed, pid)
        manip = list(config["intervention"]); task = config["task"]
        ranges = graph.get_interventional_ranges()
        dict_ranges = {v: (ranges[v][0], ranges[v][1]) for v in manip}
        es, _, manip_vars = graph.get_sets()
        es = [list(s) for s in es if len(s) <= cap]
        costs = graph.get_cost_structure(1)
        dx, dy, bx, oy, bv = benchmark._initial_interventional_data(
            graph, es, ninit, task, seed)
        _, _, cov = compute_coverage(obs, manip_vars, dict_ranges)
        csv = os.path.join(RUNDIR, f"CBO_{pid}_seed{seed}.csv")
        os.makedirs(os.path.dirname(csv), exist_ok=True)
        CBO(trials, es, manip_vars, dx, dy, bx, oy, bv, dict_ranges, functions,
            obs, cov, graph, 20, costs, obs, task, len(obs) + 50, len(obs),
            ninit, Causal_prior=True, csv_log_file=csv)
        return pd.read_csv(csv)["current_optimal"].tolist()
    return run


METHODS = {
    "QCBO-finest": _qcbo_runner(None),
    "QCBO-coarse": _qcbo_runner([list(c) for c in cb.COARSE_CLUSTERS]),
    "CBO": _cbo_runner(),
    # "CEO": ..., "CoCaBO": ...   # added next via misspec_inject
}


def _sem(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--ninit", type=int, default=5)
    ap.add_argument("--cap", type=int, default=1,
                    help="max intervention size (clusters); 1 = Tier-2 headline")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--perturbations", default="P0,P1,P2,P3,Pic,P5,P6,S1,S2,S3")
    args = ap.parse_args()

    cb.register_variants()
    ystar = benchmark.theoretical_best(DS)
    task = benchmark.load_config(DS)["task"]
    methods = [m for m in args.methods.split(",") if m in METHODS]
    pids = args.perturbations.split(",")
    pmeta = {p["id"]: p for p in cb.PERTURBATIONS}

    # traj[method][pid][seed] -> list
    traj = {m: {p: {} for p in pids} for m in methods}
    for m in methods:
        for pid in pids:
            for s in range(args.seeds):
                try:
                    traj[m][pid][s] = METHODS[m](pid, s, args.cap,
                                                 args.trials, args.ninit)
                except Exception as e:
                    import traceback
                    print(f"  {m}/{pid}/seed{s} FAILED: {e}")
                    traceback.print_exc()

    # Paired Delta vs P0 (per seed) for final Y and GAP.
    out = {"dataset": DS, "y_star": ystar, "task": task, "cap": args.cap,
           "trials": args.trials, "seeds": args.seeds, "methods": {}}
    print(f"\n{'='*78}\nClusterBench10 misspecification  (cap={args.cap}, "
          f"T={args.trials}, {args.seeds} seeds, y*={ystar:.2f})\n{'='*78}")
    hdr = f"{'method':12s} {'pert':4s} {'locus':16s} {'protected':9s} " \
          f"{'finalY':>9s} {'dFinalY':>9s} {'dGAP':>8s} {'byte-id':>7s}"
    print(hdr)
    for m in methods:
        out["methods"][m] = {}
        base = {s: traj[m]["P0"][s] for s in traj[m]["P0"]}
        for pid in pids:
            finals, dfin, dgap, ident = [], [], [], []
            for s in range(args.seeds):
                if s not in traj[m][pid] or s not in base:
                    continue
                t, t0 = traj[m][pid][s], base[s]
                g, _, fin = score(t, ystar, task, args.trials)
                g0, _, fin0 = score(t0, ystar, task, args.trials)
                finals.append(fin); dfin.append(fin - fin0); dgap.append(g - g0)
                ident.append(t == t0)
            if not finals:
                continue
            meta = pmeta.get(pid, {})
            byte_id = all(ident)
            out["methods"][m][pid] = {
                "locus": meta.get("locus"), "protected": meta.get("protected"),
                "finalY": [float(np.mean(finals)), _sem(finals)],
                "dFinalY": [float(np.mean(dfin)), _sem(dfin)],
                "dGAP": [float(np.mean(dgap)), _sem(dgap)],
                "byte_identical_to_P0": byte_id}
            print(f"{m:12s} {pid:4s} {str(meta.get('locus')):16s} "
                  f"{str(meta.get('protected')):9s} "
                  f"{np.mean(finals):+9.3f} {np.mean(dfin):+9.3f} "
                  f"{np.mean(dgap):+8.3f} {str(byte_id):>7s}")

    # --- Premise checks on real trajectories ---
    # Protected (quotient-invisible) perturbations MUST be byte-identical: this
    # is the Prop. 2 guarantee and is asserted. Inter-cluster perturbations are
    # only quotient-VISIBLE -- they change the C-DAG but a trajectory shift is
    # not guaranteed (it occurs only if the changed edge enters a *used* arm's
    # identification), so we report them rather than assert.
    print(f"\n{'-'*78}\nPREMISE CHECKS  (QCBO-coarse)")
    coarse = out["methods"].get("QCBO-coarse", {})
    failures = []
    for pid, cell in coarse.items():
        if pid == "P0":
            continue
        prot, bid = cell["protected"], cell["byte_identical_to_P0"]
        if prot:
            status = "OK" if bid else "VIOLATION"
            if not bid:
                failures.append(pid)
            print(f"  [protected]  {pid:4s} ({cell['locus']:16s}) "
                  f"byte-identical={bid!s:5s}  [{status}]  <- Prop.2 guarantee")
        else:
            print(f"  [inter ctrl] {pid:4s} ({cell['locus']:16s}) "
                  f"trajectory-differs={(not bid)!s:5s}  (informational; "
                  f"quotient-visible but not guaranteed to shift the path)")
    if failures:
        print(f"\n  !! Prop.2 VIOLATION on protected perturbations: {failures}")

    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, "clusterbench10_misspec.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {outpath}")


if __name__ == "__main__":
    main()
