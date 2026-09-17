#!/usr/bin/env python3
"""Small diagnostic or frozen unit; defaults to repaired protocol, no tuning."""
import argparse
import json
from ccbo.qmcbo.corrected import run_corrected
from ccbo.qmcbo.runner import E2_OPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["ToyGraph", "PSAGraph"], default="ToyGraph")
    p.add_argument("--algo", choices=["MCBO", "QMCBO"], default="MCBO")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--num-trials", type=int, default=3)
    p.add_argument("--menu", choices=["native", "full", "coarse"])
    p.add_argument("--misspec", default="")
    p.add_argument("--score-samples", type=int, default=100000)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    misspec = args.misspec
    if misspec == "e2":
        misspec = ",".join(f"{k}:{u}:{v}" for k,u,v in E2_OPS[args.env])
    print(json.dumps(run_corrected(args.env, args.algo, args.seed, args.num_trials,
        args.outdir, misspec=misspec, menu=args.menu, score_samples=args.score_samples), indent=2))

if __name__ == "__main__":
    main()
