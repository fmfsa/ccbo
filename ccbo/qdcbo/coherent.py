"""Opt-in coherent temporal conditional model for both DCBO resolutions.

This replaces separately fitted full-response emission/transition sums with
one model conditioned on the UNION of contemporaneous and lagged parents.
Predictive uncertainty is integrated ancestrally, not propagated as a second
set of physical parent values. It is a fitted GP predictive approximation,
not a claim of nonparametric identification or exact finite-sample inference.

Historical QDCBO remains the default. CoherentDCBO is the singleton version
of precisely the same estimator/acquisition stack as CoherentQDCBO.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np

from ccbo.qdcbo.qdcbo_base import QDCBO, _columns, fit_joint_gp, JointClusterGP


def _seed(seed, *parts):
    digest = hashlib.sha256(repr((int(seed), parts)).encode()).digest()
    return int.from_bytes(digest[:4], 'little')


def fit_conditional_mechanisms(spec, observations, fitter=None):
    """One (joint) child regression on all its temporal parents.

    Root clusters retain the joint empirical marginal, so within-cluster
    dependence is not replaced by independent member draws. Graph/data are
    the only inputs; the true structural equations are never consulted.
    """
    fitter = fitter or fit_joint_gp
    mechanisms = {t: {} for t in range(spec.T)}
    for t in range(spec.T):
        for cluster in spec.cluster_order:
            parents = tuple(dict.fromkeys(spec.trans_input_nodes(cluster, t) +
                                          spec.emit_input_nodes(cluster, t)))
            children = spec.member_nodes([cluster], t)
            y = _columns(observations, children)
            if parents:
                # Existing fitters deliberately reseed numpy. Keep that
                # deterministic fit policy local to model construction.
                rng_state = np.random.get_state()
                try:
                    model = fitter(_columns(observations, parents), y)
                finally:
                    np.random.set_state(rng_state)
                if isinstance(model, JointClusterGP):
                    model = ResidualJointConditional(model, y.shape[1])
                mechanisms[t][cluster] = dict(inputs=parents, model=model,
                                               members=spec.members(cluster))
            else:
                mechanisms[t][cluster] = dict(inputs=(), empirical=y.copy(),
                                               members=spec.members(cluster))
    return mechanisms


class ResidualJointConditional:
    """Joint latent GP + 10%-shrunk leave-one-joint-row-out residual covariance.

    This is a declared residual predictive approximation. LOO residuals
    include finite-data mean-estimation error and misspecification, so their
    covariance is not asserted to recover pure structural/aleatoric noise.
    The stock diagonal likelihood is excluded at prediction, not added twice.
    """
    def __init__(self, base, outputs, shrinkage=.1):
        self.base, self.outputs = base, outputs
        fitted = base.model
        alpha = np.asarray(fitted.posterior.woodbury_vector).reshape(-1)
        precision = np.asarray(fitted.posterior.woodbury_inv)
        if precision.ndim == 3:
            precision = precision[..., 0]
        diagonal = np.diag(precision)
        if np.any(diagonal <= 0):
            raise ValueError('Invalid GP precision for leave-one-out residuals')
        tasks = np.asarray(fitted.X)[:, -1].astype(int)
        blocks = [np.flatnonzero(tasks == output) for output in range(outputs)]
        if len({len(block) for block in blocks}) != 1:
            raise ValueError('Joint residual covariance requires aligned output rows')
        # fit_joint_gp uses the same X rows for every output in their original
        # order; verify this instead of assuming arbitrary stacked layout.
        reference = np.asarray(fitted.X)[blocks[0], :-1]
        for block in blocks[1:]:
            if not np.array_equal(reference, np.asarray(fitted.X)[block, :-1]):
                raise ValueError('Joint output training rows are not aligned')
        # Remove all task outputs of one original row together. Scalar
        # alpha_i/Kinv_ii would condition on sibling outputs from that row,
        # understating joint held-out residuals when tasks are correlated.
        paired = []
        for row in range(len(blocks[0])):
            indices = np.asarray([block[row] for block in blocks])
            held_precision = precision[np.ix_(indices, indices)]
            paired.append(np.linalg.solve(held_precision, alpha[indices]))
        paired = np.asarray(paired)
        centered = paired - paired.mean(axis=0)
        empirical = centered.T @ centered / max(len(paired)-1, 1)
        diagonal_cov = np.diag(np.diag(empirical))
        scale = max(float(np.trace(empirical)/outputs), 1.)
        self.residual_covariance = ((1-shrinkage)*empirical + shrinkage*diagonal_cov
                                    + 1e-8*scale*np.eye(outputs))
        self.residual_policy = 'joint-row-LOO-covariance-10pct-diagonal-shrinkage-v1'

    def predict_joint(self, inputs):
        mean, latent = predictive_moments(self.base, inputs, self.outputs,
                                          include_likelihood=False)
        return mean, latent + self.residual_covariance[None, :, :]


def predictive_moments(model, inputs, outputs, include_likelihood=True):
    """Return mean (n,m) and pointwise covariance (n,m,m), incl. likelihood.

    For the existing ICM model, make one joint prediction and retain the
    within-input task covariance. Correlations between distinct Monte Carlo
    particles are not needed for the one-path predictive marginal integral.
    Test fixtures can supply predict_joint without requiring GPy fitting.
    """
    if hasattr(model, 'predict_joint'):
        mean, covariance = model.predict_joint(inputs)
        return np.asarray(mean), np.asarray(covariance)
    if outputs == 1:
        mean, variance = model.predict(inputs)
        return np.asarray(mean).reshape(-1, 1), np.asarray(variance).reshape(-1, 1, 1)
    if not isinstance(model, JointClusterGP):
        raise TypeError('Multi-output mechanism must expose joint predictive covariance')
    n = len(inputs)
    # Row-major point/task layout, matching the block extraction below.
    tasks = np.tile(np.arange(outputs), n)
    xx = np.column_stack((np.repeat(inputs, outputs, axis=0), tasks))
    mean, covariance = model.model.predict(
        xx, full_cov=True, include_likelihood=include_likelihood,
        Y_metadata={'output_index': tasks[:, None]})
    covariance = np.asarray(covariance)
    if covariance.ndim == 3:
        covariance = covariance[..., 0]
    indices = np.arange(n)[:, None] * outputs + np.arange(outputs)[None, :]
    blocks = covariance[indices[:, :, None], indices[:, None, :]]
    return np.asarray(mean).reshape(n, outputs), blocks


def _gaussian_draw(mean, covariance, standard_normal):
    covariance = (covariance + covariance.swapaxes(-1, -2)) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    if np.min(values) < -1e-6:
        raise ValueError('Predictive covariance is materially indefinite')
    root = vectors * np.sqrt(np.maximum(values, 0.0))[:, None, :]
    return mean + np.einsum('nij,nj->ni', root, standard_normal)


def integrate_predictive(spec, mechanisms, temporal_index, interventions,
                         samples=256, seed=0, target='Y'):
    """Ancestral predictive moments under one fixed intervention blanket.

    Local arm-independent base streams give common random numbers across
    candidate values and cannot perturb optimizer/environment RNG state.
    Interventions replace mechanisms. Past target blanket values are fixed
    context, not observations entered twice into a transition/emission sum.
    Final target integration is Rao–Blackwellized for lower sampling error.
    """
    if samples < 2:
        raise ValueError('At least two integration particles are required')
    if not 0 <= temporal_index < spec.T:
        raise ValueError('Invalid target time')
    state = {}
    for t in range(temporal_index + 1):
        for cluster in spec.cluster_order:
            item = mechanisms[t][cluster]
            members = tuple(item['members'])
            values = [interventions.get(m, [None] * spec.T)[t] for m in members]
            mask = np.asarray([value is not None for value in values])
            if mask.all():
                draw = np.tile(np.asarray(values, dtype=float), (samples, 1))
                mean, covariance = draw, np.zeros((samples, len(members), len(members)))
            elif mask.any():
                # Whole-cluster intervention semantics do not identify a
                # partial intervention from a joint cluster conditional.
                raise ValueError(f'Partial clamp of cluster {cluster} at slice {t}')
            else:
                rng = np.random.RandomState(_seed(seed, t, cluster, members))
                if item['inputs']:
                    inputs = np.column_stack([state[node] for node in item['inputs']])
                    mean, covariance = predictive_moments(item['model'], inputs, len(members))
                    normal = rng.standard_normal((samples, len(members)))
                    draw = _gaussian_draw(mean, covariance, normal)
                else:
                    empirical = np.asarray(item['empirical'])
                    draw = empirical[rng.randint(len(empirical), size=samples)].copy()
                    # Target roots can be integrated exactly under the
                    # empirical marginal instead of sampled unnecessarily.
                    mu = empirical.mean(axis=0)
                    residual = empirical - mu
                    cov = residual.T @ residual / len(empirical)
                    mean = np.tile(mu, (samples, 1))
                    covariance = np.tile(cov, (samples, 1, 1))
            if target in members and t == temporal_index:
                k = members.index(target)
                mu = mean[:, k]
                conditional_var = covariance[:, k, k]
                expectation = float(mu.mean())
                variance = float(np.mean(conditional_var) + np.var(mu, ddof=0))
                return expectation, max(variance, 0.0)
            for j, member in enumerate(members):
                state[f'{member}_{t}'] = draw[:, j]
    raise ValueError('Target not present in temporal model')


def make_prior_functions(spec, mechanisms, temporal_index, exploration_set,
                         assigned_blanket, samples=256, seed=0, target='Y'):
    """Cache joint mean/variance evaluations under the same integration law."""
    baseline = deepcopy(assigned_blanket)
    cache = {}

    def evaluate(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if len(x) != len(exploration_set):
            raise ValueError('Intervention coordinate mismatch')
        key = tuple(x)
        if key not in cache:
            blanket = deepcopy(baseline)
            for member, value in zip(exploration_set, x):
                blanket[member][temporal_index] = float(value)
            cache[key] = integrate_predictive(
                spec, mechanisms, temporal_index, blanket, samples, seed, target)
        return cache[key]

    def moment(which):
        def function(x_values):
            return np.asarray([evaluate(x)[which] for x in np.atleast_2d(x_values)])[:, None]
        return function
    return moment(0), moment(1)


class CoherentQDCBO(QDCBO):
    """Explicit alternative using union-parent conditionals and integration.

    Acquisitions, interventional GP updates and true-SEM evaluations retain
    the corrected DCBO engine. Unsupported online/refit paths still reject.
    """
    def __init__(self, *args, predictive_samples=256, predictive_seed=0,
                 objective_protocol='zero-disturbance', feedback_samples=2048,
                 population_seed=0, **kwargs):
        if kwargs.get('stock_quirks', False):
            raise ValueError('Coherent protocol cannot combine historical stock quirks')
        self.predictive_samples = int(predictive_samples)
        self.predictive_seed = int(predictive_seed)
        self.mechanism_protocol = 'coherent-union-parent-v1'
        if objective_protocol not in ('zero-disturbance', 'population-mc'):
            raise ValueError('Unknown objective protocol')
        self.objective_protocol = objective_protocol
        self.feedback_samples = int(feedback_samples)
        self.population_seed = int(population_seed)
        self.population_events = []
        self.policy_commits = []
        if objective_protocol == 'population-mc' and kwargs.get('optimal_assigned_blankets') is not None:
            raise ValueError('Population protocol uses only explicit committed action history')
        if objective_protocol == 'population-mc' and kwargs.get('number_of_trials', 10) < 2:
            raise ValueError('Population protocol needs at least one intervention per slice')
        super().__init__(*args, **kwargs)
        self._policy_blanket = {v: [None]*self.T for v in self._spec.base_order}
        if self.objective_protocol == 'population-mc':
            from ccbo.qdcbo.population import make_training_target
            for t in range(self.T):
                for arm in self.exploration_sets:
                    self.target_functions[t][arm] = make_training_target(self, arm)

    def _make_sem_estimator(self, spec):
        # DCBO.run passes this object to our overridden statistics updater;
        # no obsolete emission/transition SEM-hat dispatch is used.
        return lambda G=None, emission_fncs=None, transition_fncs=None: emission_fncs

    def _fit_mechanisms(self, spec, observations):
        self.conditional_mechanisms = fit_conditional_mechanisms(spec, observations)
        self.sem_emit_fncs = self.conditional_mechanisms
        self.sem_trans_fncs = {}

    def _update_sufficient_statistics(self, target, temporal_index, dynamic,
                                      assigned_blanket, updated_sem):
        target_name, target_time = target.rsplit('_', 1)
        if int(target_time) != temporal_index:
            raise ValueError('Target time mismatch')
        for arm in self.exploration_sets:
            mean, variance = make_prior_functions(
                self._spec, self.conditional_mechanisms, temporal_index, arm,
                assigned_blanket, samples=self.predictive_samples,
                seed=self.predictive_seed, target=target_name)
            self.mean_function[temporal_index][arm] = mean
            self.variance_function[temporal_index][arm] = variance

    def _get_assigned_blanket(self, temporal_index):
        if self.objective_protocol != 'population-mc':
            return super()._get_assigned_blanket(temporal_index)
        return deepcopy(self._policy_blanket)

    def _per_trial_computations(self, t, it, target, assigned_blanket, method=None):
        previous_count = len(self.population_events)
        super()._per_trial_computations(t, it, target, assigned_blanket, method)
        if self.objective_protocol == 'population-mc':
            if len(self.population_events) != previous_count + 1:
                raise RuntimeError('Expected exactly one purchased population response per trial')
            event = self.population_events[-1]
            eligible = [e for e in self.population_events if e['t'] == t]
            key = (lambda e: e['training_mean']) if self.task == 'min' else (lambda e: -e['training_mean'])
            recommendation = min(eligible, key=key)
            event.update(trial=it, recommendation_event=recommendation['event_id'])

    def _post_optimisation_assignments(self, target, t, DCBO=False):
        if self.objective_protocol != 'population-mc':
            return super()._post_optimisation_assignments(target, t, DCBO)
        events = [event for event in self.population_events if event['t'] == t]
        if not events:
            raise RuntimeError('Cannot commit an unexecuted intervention')
        recommendation_id = events[-1]['recommendation_event']
        selected = self.population_events[recommendation_id]
        arm = tuple(selected['arm'])
        self.optimal_intervention_sets[t] = arm
        for variable, value in zip(arm, selected['x']):
            self._policy_blanket[variable][t] = float(value)
        # Only actual manipulable actions become historical interventions.
        # Past outcomes and unselected mediators must respond naturally.
        self.assigned_blanket = deepcopy(self._policy_blanket)
        self.assigned_blanket_hat = deepcopy(self._policy_blanket)
        self.optimal_blanket = deepcopy(self._policy_blanket)
        self.policy_commits.append(dict(t=t, recommendation_event=recommendation_id,
            arm=list(arm), x=list(selected['x']), history=deepcopy(self._policy_blanket),
            training_mean=selected['training_mean']))
        self._check_optimization_results(t)


class CoherentDCBO(CoherentQDCBO):
    """Fine-resolution baseline of exactly the same coherent estimator."""
    def __init__(self, *args, **kwargs):
        graph = kwargs.get('G', args[0] if args else None)
        target = kwargs.get('base_target_variable', 'Y')
        bases = list(dict.fromkeys(n.rsplit('_', 1)[0] for n in graph.nodes()))
        if 'partition' in kwargs:
            raise ValueError('Fine coherent baseline fixes the singleton partition')
        kwargs['partition'] = [[v] for v in bases if v != target]
        super().__init__(*args, **kwargs)
