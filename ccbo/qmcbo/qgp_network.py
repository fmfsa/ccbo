"""Joint cluster mechanisms for QMCBO (the Q-Soundness formulation).

v1 (quotient.py) runs the stock per-node ``GaussianProcessNetwork`` on the
quotient profile — per-coordinate cluster mechanisms that ignore
within-cluster correlation. This module provides the joint formulation:
one multi-output GP per **cluster** (ICM task kernel across the cluster's
member coordinates; ``KroneckerMultiTaskGP``), composed along the C-DAG with
joint (correlated) within-cluster sampling. Singleton clusters reduce to the
stock ``SingleTaskGP`` per-node mechanism, so the finest partition recovers
MCBO exactly (Q-Identity).

Structure-free ceiling: by Q-Exposure, any cluster mechanism finer than the
joint conditional (e.g. an intra-cluster sequential factorization) requires
intra-cluster structure — exposure. The joint model is the best a
quotient-respecting method can do.
"""

from __future__ import annotations

from typing import List, Optional

from ccbo.qmcbo.quotient import (ensure_mcbo_on_path, _validate_partition,
                                 _cluster_parents, _assert_acyclic)

ensure_mcbo_on_path()

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from botorch.models.model import Model  # noqa: E402
from botorch.models.gp_regression import SingleTaskGP  # noqa: E402
from botorch.models.multitask import KroneckerMultiTaskGP  # noqa: E402
from botorch.models.transforms import Standardize  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

from mcbo.models.gp_network import (  # noqa: E402
    MultivariateNormalNetwork, create_constant_feature_vector)


def _topo_cluster_order(cpar: List[set]) -> List[int]:
    """Topological order of clusters (parents before children)."""
    n = len(cpar)
    indeg = [len(p) for p in cpar]
    children: List[set] = [set() for _ in range(n)]
    for c, pars in enumerate(cpar):
        for p in pars:
            children[p].add(c)
    order, queue = [], [c for c in range(n) if indeg[c] == 0]
    while queue:
        c = queue.pop()
        order.append(c)
        for ch in children[c]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                queue.append(ch)
    assert len(order) == n, "cyclic cluster graph"
    return order


class JointQuotientGPNetwork(Model):
    """One (joint) mechanism GP per cluster, composed along the C-DAG.

    Interface mirrors ``GaussianProcessNetwork`` (set_target / posterior) so
    the stock ``mcbo_trial`` machinery can drive it unchanged. ``env_profile``
    is the FINE profile (true node count, noise dists, do_map); the quotient
    enters through ``partition``.
    """

    def __init__(self, train_X, train_Y, algo_profile, env_profile,
                 partition) -> None:
        self.train_X = train_X
        self.train_Y = train_Y
        self.algo_profile = algo_profile
        self.env_profile = env_profile
        self.partition = [list(c) for c in partition]
        n_nodes = env_profile["dag"].get_n_nodes()

        fine_parents = [list(env_profile["dag"].get_parent_nodes(k))
                        for k in range(n_nodes)]
        self.node_to_cluster = _validate_partition(self.partition, n_nodes)
        self.cluster_parents = _cluster_parents(
            fine_parents, self.node_to_cluster, len(self.partition))
        _assert_acyclic(self.cluster_parents)
        self.cluster_order = _topo_cluster_order(self.cluster_parents)
        # Input coordinates (fine node indices) feeding each cluster's GP.
        self.cluster_inputs = [
            sorted(m for cp in self.cluster_parents[c]
                   for m in self.partition[cp])
            for c in range(len(self.partition))]

        self.cluster_GPs: List[Optional[Model]] = [None] * len(self.partition)
        self.norm_lo = [None] * len(self.partition)
        self.norm_hi = [None] * len(self.partition)
        self.set_target(env_profile["valid_targets"][0])
        for c in range(len(self.partition)):
            self._fit_cluster(c)

    # ------------------------------------------------------------------
    def _cluster_mask(self, c: int) -> Tensor:
        """Rows where NO member of cluster c was intervened (train on the
        mechanism, not on clamped values). Do-flags live in train_X[..., :n]."""
        if not self.env_profile["interventional"]:
            return torch.ones(self.train_X.shape[:-1], dtype=torch.bool)
        flags = self.train_X[..., self.partition[c]]
        return (flags.int() == 0).all(dim=-1)

    def _cluster_features(self, c: int, source: Tensor) -> Tensor:
        """Normalized parent-cluster member values for cluster c, gathered
        from ``source`` (train_Y at fit time; sampled nodes at inference)."""
        idx = self.cluster_inputs[c]
        if not idx:
            return create_constant_feature_vector(source.shape[:-1])
        aux = source[..., idx].clone()
        if self.norm_lo[c] is None:
            self.norm_lo[c] = [torch.min(aux[..., j]).detach()
                               for j in range(len(idx))]
            self.norm_hi[c] = [torch.max(aux[..., j]).detach()
                               for j in range(len(idx))]
        for j in range(len(idx)):
            span = self.norm_hi[c][j] - self.norm_lo[c][j]
            aux[..., j] = (aux[..., j] - self.norm_lo[c][j]) / span
        return aux

    def _fit_cluster(self, c: int):
        mask = self._cluster_mask(c)
        X_c = self._cluster_features(c, self.train_Y)[mask]
        Y_c = self.train_Y[..., self.partition[c]][mask]
        if len(self.partition[c]) == 1:
            gp = SingleTaskGP(train_X=X_c, train_Y=Y_c,
                              outcome_transform=Standardize(m=1))
        else:
            # Full-rank task noise: within-cluster residual correlation (the
            # footprint of intra-cluster edges) is aleatoric at fixed cluster
            # inputs, so it must live in the LIKELIHOOD, not the latent ICM —
            # with diagonal task noise it would be structurally
            # unrepresentable (posterior cross-covariance exactly 0).
            from gpytorch.likelihoods import MultitaskGaussianLikelihood
            m = len(self.partition[c])
            gp = KroneckerMultiTaskGP(
                train_X=X_c, train_Y=Y_c,
                likelihood=MultitaskGaussianLikelihood(num_tasks=m, rank=m))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        self.cluster_GPs[c] = gp

    # ------------------------------------------------------------------
    def set_target(self, target):
        self.target = target
        return None

    def posterior(self, X: Tensor, posterior_transform=None):
        if self.algo_profile["algo"] == "MCBO":
            return JointHallucinatedNetwork(self, X)
        return JointMVNetwork(self, X)

    def forward(self, x: Tensor):
        raise NotImplementedError

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs):
        raise NotImplementedError


class JointMVNetwork(MultivariateNormalNetwork):
    """EIFN-style posterior: ancestral sampling over clusters with JOINT
    within-cluster draws (correlated coordinates)."""

    def __init__(self, net: JointQuotientGPNetwork, X):
        self.net = net
        self.algo_profile = net.algo_profile
        self.env_profile = net.env_profile
        self.node_GPs = net.cluster_GPs        # unused; kept for base props
        self.X = X
        self.target = net.target
        self.normalization_constant_lower = net.norm_lo
        self.normalization_constant_upper = net.norm_hi
        self.intervention_input_map()

    # -- cluster-level machinery ---------------------------------------
    def _clamp_cluster(self, c, nodes_samples):
        for k in self.net.partition[c]:
            nodes_samples[..., k] = self.X[..., k].repeat(
                (nodes_samples.shape[0], 1, 1))

    def _cluster_X(self, c, nodes_samples, sample_shape):
        idx = self.net.cluster_inputs[c]
        if not idx:
            return create_constant_feature_vector(self.event_shape[:-1])
        aux = nodes_samples[..., idx].clone()
        for j in range(len(idx)):
            span = self.net.norm_hi[c][j] - self.net.norm_lo[c][j]
            aux[..., j] = (aux[..., j] - self.net.norm_lo[c][j]) / span
        return aux

    def _sample_cluster(self, c, nodes_samples, sample_shape):
        members = self.net.partition[c]
        X_c = self._cluster_X(c, nodes_samples, sample_shape)
        is_root = not self.net.cluster_inputs[c]
        if len(members) == 1:
            # Singleton: stock MCBO semantics (latent draw + env noise) so
            # the finest partition recovers MCBO exactly (Q-Identity).
            post = self.net.cluster_GPs[c].posterior(X_c)
            draw = (post.rsample(sample_shape) if is_root
                    else post.rsample()[0])
            nodes_samples[..., members[0]] = self._with_noise(
                members[0], draw[..., 0], nodes_samples, is_root)
            return
        # Multi-member: sample the joint PREDICTIVE (observation-noise)
        # posterior — the full-rank task noise carries the within-cluster
        # residual correlation — and do NOT add env noise (it is part of the
        # learned predictive; adding it again would double-count).
        post = self.net.cluster_GPs[c].posterior(X_c, observation_noise=True)
        draw = post.rsample(sample_shape) if is_root else post.rsample()[0]
        for j, k in enumerate(members):
            vals = draw[..., j]
            if is_root and vals.shape != nodes_samples[..., k].shape:
                vals = vals.repeat((nodes_samples.shape[0], 1, 1))
            nodes_samples[..., k] = vals

    def _with_noise(self, k, vals, nodes_samples, is_root):
        if is_root and vals.shape != nodes_samples[..., k].shape:
            vals = vals.repeat((nodes_samples.shape[0], 1, 1))
        noise = self.env_profile["additive_noise_dists"][k].rsample(
            sample_shape=nodes_samples[..., k].shape)
        return vals + noise

    def _cluster_targeted(self, c):
        return all(int(self.target[k]) == 1 for k in self.net.partition[c])

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        nodes_samples, _ = self._create_nodes_sample_tensor(sample_shape)
        for c in self.net.cluster_order:
            if self._cluster_targeted(c):
                self._clamp_cluster(c, nodes_samples)
            else:
                self._sample_cluster(c, nodes_samples, sample_shape)
        return nodes_samples


class JointHallucinatedNetwork(JointMVNetwork):
    """MCBO-style optimistic realization: per-coordinate eta on the marginal
    std (box relaxation of the joint ellipsoid, matching the stock per-node
    semantics); within-cluster correlation enters through the jointly fitted
    means/stds."""

    def __init__(self, net: JointQuotientGPNetwork, X):
        super().__init__(net, X)   # maps do-values once (stock semantics)
        self.beta = net.algo_profile["beta"]
        n_nodes = net.env_profile["dag"].get_n_nodes()
        # Split AFTER the do_map applied in super().__init__ (do_map only
        # touches the leading value coords, never the eta tail) — mirrors
        # the stock HallucinatedGaussianProcessNetwork exactly.
        self.X, self.eta = (self.X[:, :, 0:(-n_nodes)],
                            (self.X[:, :, -n_nodes:] - 0.5) * 2)

    def _sample_cluster(self, c, nodes_samples, sample_shape):
        members = self.net.partition[c]
        X_c = self._cluster_X(c, nodes_samples, sample_shape)
        is_root = not self.net.cluster_inputs[c]
        post = self.net.cluster_GPs[c].posterior(X_c)
        mean, var = post.mean, post.variance
        multi = len(members) > 1
        if multi:
            # Joint correlated residual draw from the learned full-rank task
            # noise (replaces the env noise for multi-member clusters).
            lik = self.net.cluster_GPs[c].likelihood
            noise_cov = lik.task_noise_covar
            noise_cov = (noise_cov.to_dense() if hasattr(noise_cov, "to_dense")
                         else noise_cov).detach()
            if getattr(lik, "has_global_noise", False):
                noise_cov = noise_cov + lik.noise.detach() * torch.eye(
                    len(members))
            chol = torch.linalg.cholesky(
                noise_cov + 1e-9 * torch.eye(len(members)))
            z = torch.randn(nodes_samples[..., members].shape)
            joint_noise = z @ chol.T
        for j, k in enumerate(members):
            m_k = mean[..., j] if multi else mean.squeeze(-1)
            s_k = (var[..., j] if multi
                   else var.squeeze(-1)).clamp_min(1e-12).sqrt()
            out = m_k + self.beta * self.eta[:, :, k] * s_k
            if is_root and out.shape != nodes_samples[..., k].shape:
                out = out.repeat((nodes_samples.shape[0], 1, 1))
            if multi:
                noise = joint_noise[..., j]
            else:
                noise = self.env_profile["additive_noise_dists"][k].rsample(
                    sample_shape=nodes_samples[..., k].shape)
            nodes_samples[..., k] = out + noise
