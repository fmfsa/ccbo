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
import numpy as np
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


def _run_unit(env_name: str, algo: str, seed: int, num_trials: int,
             outdir: str, noise_scale: float = 0.0, beta: float = 10.0,
             misspec: str = "", initial_obs_samples: int = 5,
             initial_int_samples: int = 2, mechanism: str = "joint",
             protocol: str = "historical", menu=None,
             score_samples: int = 100000) -> dict:
    """One (env, algo, seed) run. algo in {MCBO, QMCBO}; for QMCBO,
    mechanism in {joint, ind}. joint (JointQuotientGPNetwork, the paper's
    QMCBO) keeps the QMCBO label; the per-coordinate ind ablation is
    labeled QMCBO-ind and is a dev/debug tool only."""
    if protocol == "corrected":
        if mechanism != "joint":
            raise ValueError("Corrected protocol supports joint QMCBO only")
        from ccbo.qmcbo.corrected import run_corrected
        return run_corrected(env_name, algo, seed, num_trials, outdir,
                             noise_scale, beta, misspec, initial_obs_samples,
                             initial_int_samples, menu, score_samples)
    if protocol != "historical" or menu is not None:
        raise ValueError("Historical protocol retains its native menu")
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
    # mechanism="ind"  : v1 — stock per-node GPs on the quotient profile
    #                    (per-coordinate cluster mechanisms).
    # mechanism="joint": Q-Soundness formulation — one multi-output GP per
    #                    cluster (JointQuotientGPNetwork), injected via a
    #                    runtime get_model patch; profile keeps the (possibly
    #                    perturbed) fine dag + LIFTED targets.
    joint_ctx = None
    if algo == "QMCBO":
        from ccbo.qmcbo.quotient import lift_targets
        if mechanism == "joint":
            profile = dict(profile)
            profile["valid_targets"] = lift_targets(
                profile["valid_targets"], PARTITIONS[env_name])
            joint_ctx = (profile, PARTITIONS[env_name])
        else:
            profile = quotient_env_profile(profile, PARTITIONS[env_name])
        run_algo = "MCBO"          # stock hallucination machinery
    elif algo == "MCBO":
        run_algo = "MCBO"
    else:
        raise ValueError(f"unknown algo {algo}")

    # --- determinism: neutralize mcbo_trial's torch.seed() de-seeding ------
    stock_torch_seed = torch.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    trial_mod.torch.seed = lambda: None   # runtime patch, vendored file untouched

    # --- joint mechanisms: inject our model via a get_model patch ----------
    if joint_ctx is not None:
        from ccbo.qmcbo.qgp_network import JointQuotientGPNetwork
        j_profile, j_partition = joint_ctx
        stock_get_model = trial_mod.get_model

        def _patched_get_model(X, network_observation_at_X, observation_at_X,
                               algo_profile, env_profile):
            model = JointQuotientGPNetwork(
                train_X=X, train_Y=network_observation_at_X,
                algo_profile=algo_profile, env_profile=j_profile,
                partition=j_partition)
            input_dim = env_profile["input_dim"]
            if algo_profile["algo"] == "MCBO":
                input_dim += env_profile["dag"].get_n_nodes()
            if env_profile["interventional"]:
                input_dim -= env_profile["dag"].get_n_nodes()
            return model, input_dim

        trial_mod.get_model = _patched_get_model
    else:
        stock_get_model = None

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

    # The authors' mcbo_trial persists nothing itself — it only calls
    # wandb.log once per BO iteration with the running best_score (the
    # noiseless objective mean at the best point found). Record those calls
    # through a shim and write the trajectory CSV ourselves, in the same
    # (trial_number, current_optimal) format every downstream consumer
    # reads. Each unit runs in its own scratch subdir so any stray files
    # cannot collide across concurrent units.
    label = algo if not (algo == "QMCBO" and mechanism == "ind") else "QMCBO-ind"
    os.makedirs(outdir, exist_ok=True)
    unit_dir = os.path.join(outdir, f".unit_{label}_{env_name}_{seed}")
    os.makedirs(unit_dir, exist_ok=True)

    class _WandbRecorder:
        """Stands in for the wandb module inside mcbo_trial: swallows
        every call, keeps the best_score trajectory."""

        def __init__(self):
            self.best_scores = []
            self.records = []          # engine v3: per-iteration decisions

        @staticmethod
        def _tolist(v):
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    return v.detach().cpu().double().reshape(-1).tolist()
            except Exception:
                pass
            if v is None:
                return None
            try:
                return [float(x) for x in np.ravel(np.asarray(v, dtype=float))]
            except Exception:
                return None

        def log(self, payload, *a, **k):
            if "best_score" in payload:
                self.best_scores.append(float(payload["best_score"]))
                self.records.append({
                    "iter": len(self.records),
                    "best_score": float(payload["best_score"]),
                    "score": (float(payload["score"]) if "score" in payload
                              and payload["score"] is not None else None),
                    "average_score": (float(payload["average_score"])
                                      if "average_score" in payload else None),
                    "X": self._tolist(payload.get("X")),
                })

        def __getattr__(self, name):          # init/config/etc.: no-ops
            return lambda *a, **k: None

    recorder = _WandbRecorder()
    stock_wandb = getattr(trial_mod, "wandb", None)
    trial_mod.wandb = recorder
    cwd = os.getcwd()
    os.chdir(unit_dir)
    try:
        # The authors' mcbo_trial takes exactly four arguments; benchmark
        # forks added a theoretical_optimum kwarg — pass it only if present.
        import inspect
        trial_kwargs = dict(
            algo_profile=algo_profile,
            env_profile=profile,
            function_network=function_network,
            network_to_objective_transform=objective,
        )
        if "theoretical_optimum" in inspect.signature(
                trial_mod.mcbo_trial).parameters:
            trial_kwargs["theoretical_optimum"] = getattr(
                env, "theoretical_optimum", None)
        trial_mod.mcbo_trial(**trial_kwargs)
    finally:
        torch.seed = stock_torch_seed
        os.chdir(cwd)
        if stock_wandb is not None:
            trial_mod.wandb = stock_wandb
        if stock_get_model is not None:
            trial_mod.get_model = stock_get_model

    final_csv = os.path.join(outdir,
                             f"trial_results_{label}_{env_name}_{seed}.csv")
    stock_csv = os.path.join(unit_dir,
                             f"trial_results_{run_algo}_{env_name}_{seed}.csv")
    if os.path.exists(stock_csv):        # benchmark-fork compatibility
        os.replace(stock_csv, final_csv)
    else:
        if not recorder.best_scores:
            raise RuntimeError(
                "mcbo_trial produced no best_score log entries — cannot "
                "write the trajectory CSV")
        import pandas as pd
        pd.DataFrame({
            "trial_number": list(range(len(recorder.best_scores))),
            "current_optimal": recorder.best_scores,
        }).to_csv(final_csv, index=False)
    with contextlib.suppress(OSError):
        os.rmdir(unit_dir)

    # Engine v3: full-precision per-iteration decision log (the chosen
    # intervention X, its score and the running best) next to the CSV.
    with open(final_csv.replace(".csv", ".decisions.json"), "w") as f:
        json.dump({"schema": 1,
                   "unit": {"suite": "qmcbo", "env": env_name, "algo": label,
                            "seed": seed, "num_trials": num_trials,
                            "misspec": misspec,
                            "engine_sha": os.environ.get("CCBO_GIT_SHA")},
                   "iterations": recorder.records}, f, indent=1)
    info = {"env": env_name, "algo": label, "seed": seed,
            "num_trials": num_trials, "misspec": misspec,
            "mechanism": mechanism if algo == "QMCBO" else None,
            "n_targets": len(profile["valid_targets"]),
            "parents_model_view": fine_parents,
            "zero_range_guard": True,
            "engine_sha": os.environ.get("CCBO_GIT_SHA"),
            "secs": time.time() - t0, "csv": final_csv}
    with open(final_csv.replace(".csv", "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def run_unit(env_name: str, algo: str, seed: int, num_trials: int,
             outdir: str, noise_scale: float = 0.0, beta: float = 10.0,
             misspec: str = "", initial_obs_samples: int = 5,
             initial_int_samples: int = 2, mechanism: str = "joint",
             protocol: str = "historical", menu=None,
             score_samples: int = 100000) -> dict:
    """Run with exception-safe restoration of historical runtime patches."""
    ensure_mcbo_on_path()
    import torch
    import random
    from mcbo import mcbo_trial as trial_mod
    saved = (torch.seed, torch.get_default_dtype(), trial_mod.get_model,
             trial_mod.wandb, np.random.get_state(), random.getstate())
    with torch.random.fork_rng(devices=[]):
        try:
            return _run_unit(env_name, algo, seed, num_trials, outdir,
                             noise_scale, beta, misspec, initial_obs_samples,
                             initial_int_samples, mechanism, protocol, menu,
                             score_samples)
        finally:
            torch.seed = saved[0]
            torch.set_default_dtype(saved[1])
            trial_mod.get_model, trial_mod.wandb = saved[2:4]
            np.random.set_state(saved[4])
            random.setstate(saved[5])


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
    ap.add_argument("--mechanism", default="joint", choices=["joint", "ind"],
                    help="QMCBO cluster mechanism: joint multi-output GP "
                         "(the paper's QMCBO; default) or per-coordinate "
                         "(ind; dev/debug ablation only)")
    ap.add_argument("--protocol", choices=["historical", "corrected"], default="historical")
    ap.add_argument("--menu", choices=["native", "full", "coarse"], default=None)
    ap.add_argument("--score-samples", type=int, default=100000)
    ap.add_argument("--outdir", default="results/qmcbo")
    args = ap.parse_args()
    misspec = args.misspec
    if misspec == "e2":
        misspec = ",".join(f"{k}:{u}:{v}" for k, u, v in E2_OPS[args.env])
    info = run_unit(args.env, args.algo, args.seed, args.num_trials,
                    args.outdir, args.noise_scale, args.beta, misspec,
                    mechanism=args.mechanism, protocol=args.protocol,
                    menu=args.menu, score_samples=args.score_samples)
    print(f"DONE {info['algo']:6s} {info['env']:12s} seed{info['seed']} "
          f"targets={info.get('n_targets', len(info.get('targets', [])))} ({info['secs']:.0f}s) -> {info['csv']}",
          flush=True)


if __name__ == "__main__":
    main()
