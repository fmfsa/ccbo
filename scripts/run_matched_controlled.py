#!/usr/bin/env python3
"""One immutable matched controlled unit; fresh process required per unit.

Pilot: PYTHONPATH=. python scripts/run_matched_controlled.py --scm ParallelParent \
 --cond A0 --method CBO --seed 1000 --stage pilot --max-purchases 3 --outdir /path/unit
Replication uses seeds 2000..2029 and no purchase cap; original seeds are excluded.
"""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccbo import minibench as mb
from ccbo.matched_protocol import (PROTOCOL_ID, SEED_NAMESPACE, DEFAULT_BUDGET, DEFAULT_N_INIT,
                                    SCALAR_METHODS, method_partition, observational_data,
                                    run_scalar, score_events)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scm", required=True, choices=list(DEFAULT_BUDGET))
    p.add_argument("--cond", required=True)
    p.add_argument("--method", required=True, choices=list(SCALAR_METHODS) + ["HQCBO", "CBO-FALLBACK"])
    p.add_argument("--partition", help="named partition for quotient methods (default: coarse)")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--stage", required=True, choices=["pilot", "replication"])
    p.add_argument("--max-purchases", type=int, help="pilot only: sequential purchases after initialization")
    p.add_argument("--budget", type=int, help="pilot override only; primary defaults 100/120/400")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    if args.stage == "replication" and (args.seed not in range(2000,2030) or args.max_purchases is not None or args.budget is not None):
        p.error("replication requires seed2000..2029, full default budget, no purchase cap")
    if args.stage == "pilot" and args.seed not in range(1000,1005):
        p.error("pilot requires seed1000..1004")
    if args.max_purchases is not None and args.max_purchases < 1:
        p.error("max-purchases must be positive")
    if not any(x["scm"] == args.scm and x["id"] == args.cond for x in mb.PERTURBATIONS):
        p.error("condition does not belong to selected SCM")
    out = Path(args.outdir)
    if out.exists() and any(out.iterdir()):
        p.error("outdir must be empty; stale results cannot be resumed implicitly")
    try:
        partition = method_partition(args.scm, "QCBO" if args.method == "HQCBO" else
                                     "CBO" if args.method == "CBO-FALLBACK" else args.method,
                                     args.partition)
    except ValueError as exc:
        p.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    obs = observational_data(args.scm, args.seed)
    observations = {"columns":list(obs.columns), "rows":obs.to_numpy().tolist()}
    obs_bytes = json.dumps(observations, sort_keys=True, separators=(",", ":")).encode()
    (out/"observations.json").write_bytes(obs_bytes)
    config = dict(vars(args), partition=partition, protocol_id=PROTOCOL_ID, seed_namespace=SEED_NAMESPACE,
                  n_obs=100, n_init=DEFAULT_N_INIT[args.scm],
                  feedback_mode="single_true_SCM_draw",
                  recommendation_rule="minimum_measured_Y_or_null_if_observational_mean_strictly_smaller;ties_canonical_arm_then_execution_index",
                  null_estimate=float(obs["Y"].mean()),
                  budget=DEFAULT_BUDGET[args.scm] if args.budget is None else args.budget,
                  observational_sha256=hashlib.sha256(obs_bytes).hexdigest())
    source_files = ["ccbo/matched_protocol.py", "scripts/run_matched_controlled.py", "ccbo/minibench.py", "ccbo/cbo/cbo.py", "ccbo/cbo/bo.py", "ccbo/cbo/utils/BO_functions.py", "scripts/matched_refinement.py", "scripts/matched_fallback.py", "ccbo/coarsened_graph.py", "ccbo/cbo/graphs/ClusterChain.py"]
    config["source_sha256"] = {f:digest(ROOT/f) for f in source_files}
    (out/"config.json").write_text(json.dumps(config, indent=2))
    started = time.time()
    try:
        if args.method == "CBO-FALLBACK":
            from matched_fallback import run_fallback
            experiment, meta = run_fallback(args.scm, args.cond, args.seed, args.budget, max_purchases=args.max_purchases)
        elif args.method == "HQCBO":
            from matched_refinement import run_hqcbo
            experiment, meta = run_hqcbo(args.scm, args.cond, args.seed, args.budget,
                                         max_purchases=args.max_purchases, partition_id=partition)
        else:
            experiment, meta = run_scalar(args.scm, args.cond, args.method, args.seed,
                                         args.budget, max_purchases=args.max_purchases,
                                         partition_id=partition)
        # Write measurement-only events BEFORE population scoring begins.
        events_path = out/"events.json"
        events_path.write_text(json.dumps(experiment.events, indent=2))
        scored = score_events(args.scm, json.loads(events_path.read_text()), experiment.budget,
                              null_estimate=config["null_estimate"])
        (out/"scores.json").write_text(json.dumps(scored, indent=2))
        status = "pilot_complete" if args.max_purchases is not None else "budget_complete"
        summary = dict(config, status=status, backend=meta, wall_seconds=time.time()-started,
                       actual_cost=experiment.cost, init_cost=sum(e["cost"] for e in experiment.events if e["phase"]=="init"),
                       split_init_cost=sum(e["cost"] for e in experiment.events if e["phase"]=="split_init"),
                       n_purchases=len(experiment.events), sequential_purchases=experiment.sequential_index,
                       final=scored["final"], cost_integrated_recommendation_regret=scored["cost_integrated_recommendation_regret"],
                       events_sha256=digest(events_path))
        (out/"summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k:summary[k] for k in ("status","actual_cost","sequential_purchases","final")}))
    except Exception as exc:
        (out/"summary.json").write_text(json.dumps(dict(config,status="failed", error=repr(exc), traceback=traceback.format_exc(), wall_seconds=time.time()-started), indent=2))
        raise

if __name__ == "__main__":
    main()
