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
from ccbo.matched_protocol import (PROTOCOL_ID, DEFAULT_BUDGET, observational_data,
                                    run_scalar, score_events)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scm", required=True, choices=list(DEFAULT_BUDGET))
    p.add_argument("--cond", required=True)
    p.add_argument("--method", required=True, choices=["CBO", "QCBO", "BO-S", "BO", "CBO-NP", "QCBO-NP", "CEO", "HQCBO", "CBO-FALLBACK"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--stage", required=True, choices=["pilot", "replication"])
    p.add_argument("--max-purchases", type=int, help="pilot only: sequential purchases after initialization")
    p.add_argument("--budget", type=int, help="pilot override only; primary defaults100/120")
    p.add_argument("--anchors", type=int, default=35)
    p.add_argument("--ceo-root", default=os.environ.get("CEO_ROOT"))
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
    if args.stage == "replication" and args.anchors != 35:
        p.error("replication freezes CEO anchors at35")
    out = Path(args.outdir)
    if out.exists() and any(out.iterdir()):
        p.error("outdir must be empty; stale results cannot be resumed implicitly")
    out.mkdir(parents=True, exist_ok=True)
    obs = observational_data(args.scm, args.seed)
    observations = {"columns":list(obs.columns), "rows":obs.to_numpy().tolist()}
    obs_bytes = json.dumps(observations, sort_keys=True, separators=(",", ":")).encode()
    (out/"observations.json").write_bytes(obs_bytes)
    config = dict(vars(args), protocol_id=PROTOCOL_ID, n_obs=100, n_init=3,
                  feedback_mode="single_true_SCM_draw", recommendation_rule="minimum_measured_Y;ties_canonical_arm_then_execution_index",
                  budget=DEFAULT_BUDGET[args.scm] if args.budget is None else args.budget,
                  observational_sha256=hashlib.sha256(obs_bytes).hexdigest())
    source_files = ["ccbo/matched_protocol.py", "scripts/run_matched_controlled.py", "scripts/matched_ceo_runtime.py", "ccbo/minibench.py", "ccbo/cbo/cbo.py", "ccbo/cbo/bo.py", "ccbo/cbo/utils/BO_functions.py", "scripts/ceo_adapter.py", "scripts/run_ceo_minimal.py", "scripts/matched_refinement.py", "scripts/matched_fallback.py"]
    config["source_sha256"] = {f:digest(ROOT/f) for f in source_files}
    if args.method == "CEO":
        ceo_path = Path(args.ceo_root) if args.ceo_root else ROOT/"third_party"/"CEO"
        config["ceo_source_sha256"] = {str(f.relative_to(ceo_path)):digest(f) for f in sorted((ceo_path/"src").rglob("*.py"))}
        if not config["ceo_source_sha256"]:
            p.error("CEO source directory is missing or empty")
    (out/"config.json").write_text(json.dumps(config, indent=2))
    started = time.time()
    try:
        if args.method == "CEO":
            from matched_ceo_runtime import run_ceo
            experiment, meta = run_ceo(args.scm, args.cond, args.seed, args.budget,
                                      anchors=args.anchors, ceo_root=args.ceo_root, max_purchases=args.max_purchases)
        elif args.method == "CBO-FALLBACK":
            from matched_fallback import run_fallback
            experiment, meta = run_fallback(args.scm, args.cond, args.seed, args.budget, max_purchases=args.max_purchases)
        elif args.method == "HQCBO":
            from matched_refinement import run_hqcbo
            experiment, meta = run_hqcbo(args.scm, args.cond, args.seed, args.budget, max_purchases=args.max_purchases)
        else:
            experiment, meta = run_scalar(args.scm, args.cond, args.method, args.seed,
                                         args.budget, max_purchases=args.max_purchases)
        # Write measurement-only events BEFORE population scoring begins.
        events_path = out/"events.json"
        events_path.write_text(json.dumps(experiment.events, indent=2))
        scored = score_events(args.scm, json.loads(events_path.read_text()), experiment.cost)
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
