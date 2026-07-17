"""Run MCBO or QMCBO on a vendored-MCBO environment (one unit = one process).

Mirrors baselines/mcbo/scripts/runner.py but: (i) dispatches QMCBO by
quotient-transforming the env_profile (see quotient.py), (ii) supports
misspecification injection into the model's DAG view (fixed objective),
(iii) is deterministic per seed (mcbo_trial's ``torch.seed()`` de-seeding is
neutralized at runtime — the vendored file is not edited), (iv) runs wandb in
disabled mode, and (v) writes the trial CSV into a caller-chosen directory
with the stock naming ``trial_results_<algo>_<env>_<seed>.csv``.

Usage (module is importable for the E-scripts; also runnable directly):
  PYTHONPATH=. conda run -n ccbo python -m ccbo.qmcbo.runner \
      --env ToyGraph --algo QMCBO --seed 0 --num_trials 50 --outdir <dir> \
      [--misspec del:0:1]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import contextlib
import json
import time

from ccbo.qmcbo.quotient import (ensure_mcbo_on_path, quotient_env_profile,
                                 perturb_parent_nodes)

# Partitions over env node indices; non-manipulable nodes and Y stay
# singletons. These mirror the QCBO paper's domain partitions.
PARTITIONS = {
    "ToyGraph":    [[0, 1], [2]],                      # {X, Z} | Y
    "Synthetic_2": [[0, 1], [2]],                      # {X, Z} | Y
    "PSAGraph":    [[0], [1], [2, 3], [4], [5]],       # {A(spirin), S(tatin)}
}

# The intra-cluster misspecification each env's E2 uses (model view only).
E2_OPS = {
    "ToyGraph":    [("del", 0, 1)],   # delete X->Z inside {X,Z}
    "Synthetic_2": [("rev", 0, 1)],   # reverse X->Z inside {X,Z}
    "PSAGraph":    [("add", 2, 3)],   # add spurious A->S inside {A,S}
}


def make_env(name: str, noise_scale: float = 0.0):
    ensure_mcbo_on_path()
    import functions as F
    ctors = {
        "ToyGraph": lambda: F.ToyGraph(noise_scales=noise_scale),
        "Synthetic_2": lambda: F.Synthetic_2(noise_scales=noise_scale),
        "PSAGraph": lambda: F.PSAGraph(),
        "EcologyH": lambda: F.EcologyH(noise_scales=noise_scale),
    }
    return ctors[name]()


def _parse_misspec(spec):
    """'del:0:1,add:2:3' -> [('del',0,1), ('add',2,3)]."""
    if not spec:
        return []
    ops = []
    for item in spec.split(","):
        kind, u, v = item.split(":")
        ops.append((kind, int(u), int(v)))
    return ops


def run_unit(env_name: str, algo: str, seed: int, num_trials: int,
             outdir: str, noise_scale: float = 0.0, beta: float = 10.0,
             misspec: str = "", initial_obs_samples: int = 5,
             initial_int_samples: int = 2) -> dict:
    """One (env, algo, seed) run. algo in {MCBO, QMCBO}. Returns run info."""
    ensure_mcbo_on_path()
    import torch
    import numpy as np
    import wandb
    from mcbo.utils.dag import DAG
    from mcbo import mcbo_trial as trial_mod

    t0 = time.time()
    import torch as _torch
    # The vendored samplers cast to double while the envs emit float32;
    # unify at the process level (BoTorch recommends float64 anyway).
    _torch.set_default_dtype(_torch.float64)
    env = make_env(env_name, noise_scale)
    profile = env.get_env_profile()

    # --- misspecification: perturb only the model's DAG view ---------------
    ops = _parse_misspec(misspec)
    fine_parents = [list(profile["dag"].get_parent_nodes(k))
                    for k in range(profile["dag"].get_n_nodes())]
    if ops:
        fine_parents = perturb_parent_nodes(fine_parents, ops)
        profile = dict(profile)
        profile["dag"] = DAG(fine_parents)

    # --- QMCBO: quotient the (possibly perturbed) model view ---------------
    if algo == "QMCBO":
        profile = quotient_env_profile(profile, PARTITIONS[env_name])
        run_algo = "MCBO"          # stock hallucination machinery
    elif algo == "MCBO":
        run_algo = "MCBO"
    else:
        raise ValueError(f"unknown algo {algo}")

    # --- determinism: neutralize mcbo_trial's torch.seed() de-seeding ------
    torch.manual_seed(seed)
    np.random.seed(seed)
    trial_mod.torch.seed = lambda: None   # runtime patch, vendored file untouched

    wandb.init(mode="disabled")
    # mcbo_trial reads wandb.config.env/seed for the CSV filename; in disabled
    # mode config is a plain settable object.
    with contextlib.suppress(Exception):
        wandb.config.env = env_name
        wandb.config.seed = seed

    from botorch.acquisition.objective import GenericMCObjective

    def function_network(X):
        return env.evaluate(X=X)

    objective = GenericMCObjective(lambda samples, X=None: samples[..., -1])

    algo_profile = {
        "algo": run_algo,
        "seed": seed,
        "n_init_evals": 2 * (profile["input_dim"] + 1),
        "n_bo_iter": num_trials,
        "beta": beta,
        "initial_obs_samples": initial_obs_samples,
        "initial_int_samples": initial_int_samples,
        "batch_size": 2,
    }

    # mcbo_trial writes its CSV to the CWD under a stock name that collides
    # across concurrent units (QMCBO runs as algo "MCBO"); isolate each unit
    # in its own scratch subdir, then move the CSV to its final name.
    os.makedirs(outdir, exist_ok=True)
    unit_dir = os.path.join(outdir, f".unit_{algo}_{env_name}_{seed}")
    os.makedirs(unit_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(unit_dir)
    try:
        trial_mod.mcbo_trial(
            algo_profile=algo_profile,
            env_profile=profile,
            function_network=function_network,
            network_to_objective_transform=objective,
            theoretical_optimum=getattr(env, "theoretical_optimum", None),
        )
    finally:
        os.chdir(cwd)

    stock_csv = os.path.join(unit_dir,
                             f"trial_results_{run_algo}_{env_name}_{seed}.csv")
    final_csv = os.path.join(outdir,
                             f"trial_results_{algo}_{env_name}_{seed}.csv")
    os.replace(stock_csv, final_csv)
    with contextlib.suppress(OSError):
        os.rmdir(unit_dir)

    info = {"env": env_name, "algo": algo, "seed": seed,
            "num_trials": num_trials, "misspec": misspec,
            "n_targets": len(profile["valid_targets"]),
            "parents_model_view": fine_parents if algo == "MCBO" else
            [list(profile["dag"].get_parent_nodes(k))
             for k in range(profile["dag"].get_n_nodes())],
            "secs": time.time() - t0, "csv": final_csv}
    with open(final_csv.replace(".csv", "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(PARTITIONS) + ["EcologyH"])
    ap.add_argument("--algo", required=True, choices=["MCBO", "QMCBO"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_trials", type=int, default=50)
    ap.add_argument("--noise_scale", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--misspec", default="",
                    help="edge ops on the model view, e.g. 'del:0:1' or 'e2' "
                         "for the env's canonical intra-cluster perturbation")
    ap.add_argument("--outdir", default="third_party/CausalBO_Benchmark/"
                                        "results/_qmcbo")
    args = ap.parse_args()
    misspec = args.misspec
    if misspec == "e2":
        misspec = ",".join(f"{k}:{u}:{v}" for k, u, v in E2_OPS[args.env])
    info = run_unit(args.env, args.algo, args.seed, args.num_trials,
                    args.outdir, args.noise_scale, args.beta, misspec)
    print(f"DONE {info['algo']:6s} {info['env']:12s} seed{info['seed']} "
          f"targets={info['n_targets']} ({info['secs']:.0f}s) -> {info['csv']}",
          flush=True)


if __name__ == "__main__":
    main()
