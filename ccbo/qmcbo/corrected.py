"""Versioned, paired MCBO protocol; upstream reproduction remains in runner.py.

The scorer is called only after acquisition/measurement and never enters training.
Scores are oracle best-visited diagnostics; recommendations use measured outcomes.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import time

import numpy as np

PROTOCOL = "mcbo-corrected-v2"


def keyed_seed(seed, *parts):
    msg = json.dumps([PROTOCOL, int(seed), *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(msg.encode()).digest()[:4], "little")


@contextlib.contextmanager
def isolated_rng(seed):
    import torch
    np_state, py_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        random.seed(int(seed))
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


def scalar_gp(X, Y, noise_variance):
    """Exactly the released scalar model/fit policy, shared by both methods."""
    import torch
    from botorch.models import FixedNoiseGP
    from botorch.models.transforms import Standardize
    from botorch import fit_gpytorch_model
    from gpytorch.mlls import ExactMarginalLogLikelihood
    model = FixedNoiseGP(train_X=X, train_Y=Y,
                         train_Yvar=torch.ones_like(Y) * noise_variance,
                         outcome_transform=Standardize(m=1))
    fit_gpytorch_model(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def broadcast_noise(distribution, shape):
    """One residual per MC sample, shared over candidate batches and q points.

    The protocol uses q=1. Sharing across candidates makes the fixed SAA
    objective independent of candidate ordering and optimizer batch limits.
    """
    import torch
    base_shape = torch.Size([shape[0]] + [1] * (len(shape) - 1))
    return distribution.rsample(sample_shape=base_shape).expand(shape)


@contextlib.contextmanager
def corrected_backend(acquisition_seed=0):
    """Shared physical mapping, range guard and fixed per-call acquisition noise.

    All changes are restored, including failure paths. The deterministic noise is
    a sample-average approximation within each optimization (not fresh samples
    during each numerical objective/gradient call).
    """
    import torch
    from mcbo.models import gp_network as g
    from ccbo.qmcbo.qgp_network import JointMVNetwork, JointHallucinatedNetwork
    from ccbo.qmcbo.quotient import normalize_columns
    patches = []

    def patch(obj, name, value):
        patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def mapped_init(self, node_GPs, X, algo_profile, env_profile,
                    normalization_constants, target):
        g.MultivariateNormalNetwork.__init__(self, node_GPs, X, algo_profile,
                                            env_profile, normalization_constants, target)
        self.beta = algo_profile["beta"]
        self.X, self.eta = self._split_epistemic_control_actions_and_regular_actions(self.X)

    def features(self, k, is_root):
        active = self.env_profile["active_input_indices"]
        if g.check_no_inputs(k, is_root, active):
            return g.create_constant_feature_vector(self.train_X.shape[:-1])
        if is_root:
            return self.train_X[..., active[k]]
        aux = self.train_Y[..., self.dag.get_parent_nodes(k)].clone()
        self.normalization_constant_lower[k] = [aux[..., j].min() for j in range(aux.shape[-1])]
        self.normalization_constant_upper[k] = [aux[..., j].max() for j in range(aux.shape[-1])]
        aux = normalize_columns(aux, self.normalization_constant_lower[k], self.normalization_constant_upper[k])
        return torch.cat([self.train_X[..., active[k]], aux], -1)

    def parent_samples(self, k, samples):
        aux = samples[..., self.env_profile["dag"].get_parent_nodes(k)].clone()
        return normalize_columns(aux, self.normalization_constant_lower[k], self.normalization_constant_upper[k])

    def fit(self, k, is_root):
        mask = self._create_intervention_mask(self.train_X[..., k], self.env_profile["interventional"])
        X = self._construct_features_k(k, is_root)
        self.node_GPs[k] = scalar_gp(X[mask], self.train_Y[..., [k]][mask],
                                    self.env_profile["additive_noise_dists"][k].variance)

    def hallucinated_sample(self, k, posterior, root_node, nodes_samples):
        out = posterior.mean.squeeze(-1) + self.beta * self.eta[:, :, k] * posterior.variance.squeeze(-1).clamp_min(0).sqrt()
        if root_node:
            out = out.repeat((nodes_samples.shape[0], 1, 1))
        return out + broadcast_noise(self.env_profile["additive_noise_dists"][k],
                                     nodes_samples[..., k].shape)

    def deterministic(original):
        def draw(self, *args, **kwargs):
            with isolated_rng(acquisition_seed):
                return original(self, *args, **kwargs)
        return draw

    patch(g.HallucinatedGaussianProcessNetwork, "__init__", mapped_init)
    patch(g.GaussianProcessNetwork, "_construct_features_k", features)
    patch(g.GaussianProcessNetwork, "_fit_single_node", fit)
    patch(g.HallucinatedGaussianProcessNetwork, "_get_hallucinated_node_sample", hallucinated_sample)
    patch(g.MultivariateNormalNetwork, "_normalize_parent_sample", parent_samples)
    patch(g.HallucinatedGaussianProcessNetwork, "rsample", deterministic(g.HallucinatedGaussianProcessNetwork.rsample))
    patch(JointMVNetwork, "rsample", deterministic(JointMVNetwork.rsample))
    try:
        yield
    finally:
        for obj, name, original in reversed(patches):
            setattr(obj, name, original)


def action_menu(profile, partition, mode):
    """Explicit menus: native unchanged; full adds all manipulable subsets."""
    import itertools
    import torch
    from ccbo.qmcbo.quotient import lift_targets
    native = profile["valid_targets"]
    if mode == "native":
        return native
    if mode == "coarse":
        return lift_targets(native, partition)
    if mode != "full":
        raise ValueError(mode)
    movable = sorted({k for t in native for k, v in enumerate(t) if int(v)})
    targets = []
    for size in range(1, len(movable) + 1):
        for subset in itertools.combinations(movable, size):
            t = torch.zeros_like(native[0]); t[list(subset)] = 1
            targets.append(t)
    if any(not t.any() for t in native):
        targets.append(torch.zeros_like(native[0]))
    return targets


def initial_design(profile, seed, n_obs=5, n_per_target=2):
    import torch
    n = profile["dag"].get_n_nodes()
    blocks = []
    targets = [torch.zeros(n)] + [t for t in profile["valid_targets"] if t.any()]
    for target in targets:
        key = tuple(int(v) for v in target)
        count = n_per_target if any(key) else n_obs
        with isolated_rng(keyed_seed(seed, "init-levels", key)):
            values = torch.rand(count, n)
        blocks.append(torch.cat([target.repeat(count, 1), values], -1))
    return torch.cat(blocks)


def run_corrected(env_name, algo, seed, num_trials, outdir, noise_scale=0.0,
                  beta=10.0, misspec="", initial_obs_samples=5,
                  initial_int_samples=2, menu=None, score_samples=100000):
    from ccbo.qmcbo.quotient import ensure_mcbo_on_path, perturb_parent_nodes
    ensure_mcbo_on_path()
    import torch
    import importlib
    from mcbo import mcbo_trial as trial
    optimizer_module = importlib.import_module("mcbo.acquisition_function_optimization.optimize_acqf")
    from mcbo.utils.dag import DAG
    from botorch.acquisition.objective import GenericMCObjective
    from ccbo.qmcbo.runner import make_env, PARTITIONS, _parse_misspec
    from ccbo.qmcbo.qgp_network import JointQuotientGPNetwork
    if algo not in ("MCBO", "QMCBO"):
        raise ValueError(algo)
    menu = menu or ("coarse" if algo == "QMCBO" else "full")
    if algo == "QMCBO" and menu != "coarse":
        raise ValueError("QMCBO requires whole-cluster coarse menu")
    old_dtype, old_get_model, old_wandb = torch.get_default_dtype(), trial.get_model, trial.wandb
    old_optimize = trial.optimize_acqf_and_get_suggested_point
    old_optimizer_wandb = optimizer_module.wandb
    torch.set_default_dtype(torch.float64)
    t0 = time.time()
    events, fit_diagnostics = [], []
    class Recorder:
        def log(self, payload, *args, **kwargs):
            if "acq_value" in payload:
                fit_diagnostics.append({"acq_value": float(payload["acq_value"])})
    try:
        env = make_env(env_name, noise_scale)
        profile = dict(env.get_env_profile())
        parents = [list(profile["dag"].get_parent_nodes(k)) for k in range(profile["dag"].get_n_nodes())]
        if misspec:
            profile["dag"] = DAG(perturb_parent_nodes(parents, _parse_misspec(misspec)))
        profile["valid_targets"] = action_menu(profile, PARTITIONS[env_name], menu)
        config = {"algo": "MCBO", "seed": seed, "beta": beta, "batch_size": 2,
                  "scalar_fit_policy": "upstream-fixed", "n_bo_iter": num_trials}
        objective = GenericMCObjective(lambda samples, X=None: samples[..., -1])
        def get_model(X, network_observation_at_X, observation_at_X, algo_profile, env_profile):
            if algo == "QMCBO":
                model = JointQuotientGPNetwork(X, network_observation_at_X, algo_profile,
                                               env_profile, PARTITIONS[env_name])
                dim = env_profile["input_dim"]
            else:
                model, dim = old_get_model(X, network_observation_at_X, observation_at_X,
                                           algo_profile, env_profile)
            gps = model.cluster_GPs if algo == "QMCBO" else model.node_GPs
            fit_diagnostics.append({"models": [{"class": type(gp).__name__,
                "parameters": {name: p.detach().reshape(-1).tolist()
                               for name, p in gp.named_parameters()},
                "fit_rows": int(gp.train_inputs[0].shape[-2])} for gp in gps]})
            return model, dim
        def optimize(*args, **kwargs):
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate, value = old_optimize(*args, **kwargs)
            target_index = sum("candidate" in d for d in fit_diagnostics) % len(profile["valid_targets"])
            fit_diagnostics.append({"target_index": target_index,
                                    "target": profile["valid_targets"][target_index].tolist(),
                                    "candidate": candidate.detach().reshape(-1).tolist(),
                                    "acq_value": float(value),
                                    "warnings": [str(w.message) for w in caught]})
            return candidate, value
        trial.get_model = get_model
        trial.optimize_acqf_and_get_suggested_point = optimize
        recorder = Recorder()
        trial.wandb = recorder
        # Upstream optimizer owns a separate module binding. Intercept it too:
        # wandb.log otherwise requires initialization and can emit telemetry.
        optimizer_module.wandb = recorder
        X = initial_design(profile, seed, initial_obs_samples, initial_int_samples)
        n = profile["dag"].get_n_nodes()
        observations, means, counts = [], [], {}
        valid_target_keys = {tuple(int(v) for v in t) for t in profile["valid_targets"]}
        def evaluate(x, phase, index):
            target = tuple(int(v) for v in x[0, :n])
            key = ("init-measure", target, index) if phase == "init" else ("measure", index)
            with isolated_rng(keyed_seed(seed, *key)):
                y = env.evaluate(x)
            with isolated_rng(keyed_seed(seed, "score")):
                reps = 1 if env_name == "ToyGraph" and noise_scale == 0 else score_samples
                scores = env.evaluate(x.unsqueeze(0).repeat(reps, 1, 1))[..., -1]
                mean = float(scores.mean())
                se = float(scores.std(unbiased=True) / reps ** .5) if reps > 1 else 0.0
            events.append({"phase": phase, "index": index, "target": list(target),
                           "X": x.detach().reshape(-1).tolist(),
                           "physical_values": env.do_map(x[:, n:]).detach().reshape(-1).tolist(),
                           "network_observation": y.detach().reshape(-1).tolist(),
                           "measured_y": float(y[0,-1]), "population_estimate": mean,
                           "score_se": se, "score_samples": reps, "cost": sum(target)})
            events[-1]["recommendation_eligible"] = target in valid_target_keys
            return y, mean
        for x in X.split(1):
            target = tuple(int(v) for v in x[0,:n]); idx = counts.get(target, 0)
            counts[target] = idx + 1
            y, mean = evaluate(x, "init", idx); observations.append(y); means.append(mean)
        Y = torch.cat(observations)
        initial_best = max(v for e,v in zip(events,means) if e["recommendation_eligible"])
        trajectory = []
        for iteration in range(num_trials):
            before = len(fit_diagnostics)
            with isolated_rng(keyed_seed(seed, "algorithm", iteration)), corrected_backend(keyed_seed(seed, "acquisition", iteration)):
                new_x, _ = trial.get_new_suggested_point(X, Y, objective(Y), config, profile,
                                                        env.evaluate, objective, [])
            y, mean = evaluate(new_x, "sequential", iteration)
            X = torch.cat([X, new_x]); Y = torch.cat([Y, y]); means.append(mean)
            eligible = torch.tensor([e["recommendation_eligible"] for e in events])
            recommendation = int(objective(Y).masked_fill(~eligible, -torch.inf).argmax())
            oracle_best = max(v for e,v in zip(events,means) if e["recommendation_eligible"])
            events[-1].update(recommendation_event= recommendation,
                              recommendation_population_estimate=means[recommendation],
                              oracle_best_visited=oracle_best,
                              usable_node_rows=[int((X[:, k] == 0).sum()) for k in range(n)],
                              acquisition_diagnostics=fit_diagnostics[before:])
            trajectory.append((means[recommendation], oracle_best))
        os.makedirs(outdir, exist_ok=True)
        label = algo + ("-coarse-menu" if algo == "MCBO" and menu == "coarse" else "")
        stem = os.path.join(outdir, f"trial_results_{label}_{env_name}_{seed}")
        import csv
        with open(stem + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_number", "recommendation_population_estimate", "oracle_best_visited"])
            writer.writerows((i, recommended, oracle) for i, (recommended, oracle) in enumerate(trajectory))
        info = dict(protocol=PROTOCOL, env=env_name, algo=algo, menu=menu, seed=seed,
                    num_trials=num_trials, misspec=misspec, beta=beta, noise_scale=noise_scale,
                    partition=PARTITIONS[env_name] if algo == "QMCBO" else None,
                    scalar_fit_policy="upstream FixedNoiseGP + Standardize; shared fit helper",
                    environment_noise="intrinsically stochastic PSA; noise_scale ignored" if env_name == "PSAGraph" else noise_scale,
                    score_samples=score_samples, initial_best_population_estimate=initial_best,
                    initial_rows=sum(counts.values()), initial_cost=sum(e["cost"] for e in events if e["phase"]=="init"),
                    targets=[t.tolist() for t in profile["valid_targets"]],
                    n_targets=len(profile["valid_targets"]),
                    metric="data-selected recommendation population estimate (max measured outcome); oracle diagnostic separate",
                    secs=time.time()-t0, csv=stem + ".csv")
        with open(stem + ".decisions.json", "w") as f:
            json.dump({"schema": 2, "unit": info, "events": events}, f, indent=1)
        with open(stem + "_info.json", "w") as f:
            json.dump(info, f, indent=2)
        return info
    finally:
        trial.get_model, trial.wandb = old_get_model, old_wandb
        trial.optimize_acqf_and_get_suggested_point = old_optimize
        optimizer_module.wandb = old_optimizer_wandb
        torch.set_default_dtype(old_dtype)
