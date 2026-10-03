import time
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
import time

from ccbo.cbo.utils import *
import logging

logger = logging.getLogger(__name__)



def _check_observation_prefix(observational_samples, full_observational_samples,
								cursor):
	"""Validate the cursor contract: the current observational frame must be
	the first ``cursor`` rows of the pool (all runners build it as
	``full_obs[:n]``).  Compared on shared columns and by value, because the
	frame is re-indexed on every observe action."""
	if cursor > len(full_observational_samples):
		raise ValueError(
			f"observation cursor {cursor} exceeds the pool "
			f"({len(full_observational_samples)} rows)")
	if len(observational_samples) != cursor:
		raise ValueError(
			f"observational frame has {len(observational_samples)} rows but "
			f"the cursor says {cursor}; the initial frame must be a prefix of "
			f"full_observational_samples")
	cols = [c for c in observational_samples.columns
			if c in full_observational_samples.columns]
	if not cols:
		raise ValueError("observational frame shares no columns with the pool")
	left = np.asarray(observational_samples[cols], dtype=float)
	right = np.asarray(full_observational_samples[cols].iloc[:cursor], dtype=float)
	if not np.allclose(left, right, equal_nan=True):
		raise ValueError(
			"observational frame is not the first "
			f"{cursor} rows of full_observational_samples; pass "
			"observation_cursor explicitly")


def CBO(num_trials, exploration_set, manipulative_variables, data_x_list, data_y_list,  best_intervention_value, opt_y,
					best_variable, dict_ranges, functions, observational_samples, coverage_total, graph,
					num_additional_observations, costs, full_observational_samples, task = 'min', max_N = 200,
					initial_num_obs_samples =100, num_interventions=10, Causal_prior=False,
					state=None, return_state=False, intervention_callback=None,
					target_evaluator=None, observation_cursor=None,
					force_observe_on_entry=None):
	"""
	Causal Bayesian Optimization loop.

	Parameters
	----------
	[... original parameters unchanged ...]
	state : dict, optional
		If provided, resume from this saved state instead of initializing.
		Obtained from a previous call with return_state=True.
	return_state : bool
		If True, return (results, state_dict) instead of just results.
		The state_dict can be passed back to resume the loop.
	Causal_prior : bool or sequence of bool
		True/False for every arm, or one flag per arm (engine v3 prior mask).
	intervention_callback : callable, optional
		Called after each intervention with signature:
		  callback(intervention_vars, x_new, y_new, sem_fn)
		Used by RCCBO to record full-variable samples for RePaRe.
	target_evaluator : callable, optional
		Exact population evaluator with signature ``(arm, values) -> float``.
		Used by MinimalBench to avoid simulation-based target evaluations.
	observation_cursor : int, optional
		Position of the first not-yet-revealed row of
		``full_observational_samples``.  Resolution order: this argument,
		then ``state['obs_cursor']``, then ``len(observational_samples)``
		(valid only because the initial frame is a prefix of the pool --
		checked below).  Pass it explicitly to carry the observation
		budget across a phase boundary (HQCBO) without re-revealing rows.
	force_observe_on_entry : bool, optional
		Whether trial 0 of *this call* forces an observation (budget
		permitting).  Defaults to ``state is None``: a fresh run or a
		fresh optimizer phase observes first, a resumed state does not.

	Observation protocol
	--------------------
	Each observe action reveals a fresh, previously unseen batch of
	``num_additional_observations`` rows; the final batch may be shorter.
	The effective cap is ``min(max_N, len(full_observational_samples))``;
	once it is reached the optimizer must intervene, forced or not.  Every
	observation refreshes the graph's observational data, invalidates the
	do-estimator cache, clears the point-wise prior caches, and rebuilds
	all arm GPs -- see the observe branch below.
	"""

	if state is not None:
		# === RESUME FROM SAVED STATE ===
		current_cost = state['current_cost']
		global_opt = state['global_opt']
		current_best_x = state['current_best_x']
		current_best_y = state['current_best_y']
		x_dict_mean = state['x_dict_mean']
		x_dict_var = state['x_dict_var']
		dict_interventions = state['dict_interventions']
		observed = state['observed']
		trial_intervened = state['trial_intervened']
		cumulative_cost = state['cumulative_cost']
		target_function_list = state['target_function_list']
		space_list = state['space_list']
		model_list = state['model_list']
		type_trial = state['type_trial']
		mean_functions_list = state['mean_functions_list']
		var_functions_list = state['var_functions_list']
		functions = state['functions']
		observational_samples = state['observational_samples']
		index = state.get('index', 0)
		data_x_list = state['data_x_list']
		data_y_list = state['data_y_list']
		obs_cursor = state.get('obs_cursor')
		initial_obs_cursor = state.get('initial_obs_cursor')
		num_observations_collected = state.get('num_observations_collected', 0)
		models_fresh = state.get('models_fresh', False)
		trial_log = list(state.get('trial_log', []))
		opening_forced_observe = bool(state.get('opening_forced_observe', False))
		fixed_priors = dict(state.get('fixed_priors', {}))
		resumed = True
	else:
		# === ORIGINAL INITIALIZATION (unchanged) ===
		## Initialise dicts to store values over trials and assign initial values
		current_cost = []
		global_opt = []
		current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions = initialise_dicts(exploration_set, task)
		current_best_y[best_variable].append(opt_y)
		current_best_x[best_variable].append(best_intervention_value)
		global_opt.append(opt_y)
		current_cost.append(0.)

		## Initialise variables
		observed = 0
		trial_intervened = 0.
		cumulative_cost = 0.
		cumulative_cost_mf = 0.

		## Define list to store info
		target_function_list = [None]*len(exploration_set)
		space_list = [None]*len(exploration_set)
		model_list = [None]*len(exploration_set)
		type_trial = []
		index = 0

		## Define intervention function
		for s in range(len(exploration_set)):
			target_function_list[s], space_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),
												model = graph.define_SEM(), target_variable = 'Y',
												min_intervention = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[0],
												max_intervention = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[1])
			if target_evaluator is not None:
				_arm = tuple(exploration_set[s])
				target_function_list[s] = lambda value, arm=_arm: np.asarray(
					target_evaluator(arm, value), dtype=float)[np.newaxis, np.newaxis]

		mean_functions_list = [None]*len(exploration_set)
		var_functions_list = [None]*len(exploration_set)
		obs_cursor = None
		initial_obs_cursor = None
		num_observations_collected = 0
		models_fresh = False
		trial_log = []
		opening_forced_observe = False
		fixed_priors = {}
		resumed = False


	############################# OBSERVATION BUDGET
	## Effective cap: the declared maximum, or the pool, whichever is smaller.
	obs_cap = int(min(max_N, len(full_observational_samples)))

	## Cursor contract. `len(observational_samples)` is a valid fallback only
	## because every runner slices the initial frame as a *prefix* of the pool
	## (`obs = full_obs[:n]`). Check that rather than assume it: a silently
	## wrong cursor reproduces the very bug this replaces.
	if observation_cursor is not None:
		obs_cursor = int(observation_cursor)
	elif obs_cursor is None:
		obs_cursor = int(len(observational_samples))
	obs_cursor = int(obs_cursor)

	_check_observation_prefix(observational_samples, full_observational_samples,
								obs_cursor)

	if initial_obs_cursor is None:
		initial_obs_cursor = obs_cursor

	## Phase-entry policy: a fresh run (or a fresh optimizer phase) forces one
	## observation; a resumed state does not.
	fresh_phase = state is None
	if force_observe_on_entry is None:
		force_observe_on_entry = fresh_phase

	############################# PRIOR MASK (engine v3)
	## ``Causal_prior`` may be a bool (all arms) or a per-arm sequence: arms
	## with ``False`` run on a plain RBF GP (no do-calculus mean / variance),
	## which is what the graph-free refinement and the no-prior ablations use.
	if isinstance(Causal_prior, (list, tuple, np.ndarray)):
		prior_mask = [bool(b) for b in Causal_prior]
	else:
		prior_mask = [bool(Causal_prior)] * len(exploration_set)
	assert len(prior_mask) == len(exploration_set), (
		f"prior mask length {len(prior_mask)} != {len(exploration_set)} arms")
	any_prior = any(prior_mask)

	## Arms whose prior is supplied directly (``state['fixed_priors']``: arm
	## index -> (mean, var) functions) are never sent to the graph; the
	## graph-free refinement uses this for newly exposed arms.
	def refresh_priors():
		graph_mask = [m and j not in fixed_priors for j, m in enumerate(prior_mask)]
		if any(graph_mask):
			means, variances = update_all_do_functions(
				graph, exploration_set, functions, dict_interventions,
				observational_samples, x_dict_mean, x_dict_var, prior_mask=graph_mask)
		else:
			means, variances = [None] * len(exploration_set), [None] * len(exploration_set)
		for j, (mean_fn, var_fn) in fixed_priors.items():
			means[j], variances[j] = mean_fn, var_fn
		return means, variances

	############################# LOOP
	start_time = time.perf_counter()
	for i in range(num_trials):
		logger.debug('Optimization step %s', i)
		## Decide to observe or intervene and then recompute the obs coverage.
		## The budget guard sits OUTSIDE the probabilistic decision: an
		## exhausted pool always means intervene.
		can_observe = obs_cursor < obs_cap
		if can_observe:
			epsilon_raw, epsilon_coverage = observation_probability(
				observational_samples, manipulative_variables,
				coverage_total, obs_cap)
		else:
			epsilon_raw, epsilon_coverage = 0.0, 0.0

		## Drawn unconditionally so the guard never perturbs the RNG stream.
		uniform = np.random.uniform(0.,1.)

		## At least observe and intervene once (only on first-ever run/phase,
		## not on resume). Expressed as explicit decisions rather than through
		## doctored `uniform` values, so force_observe_on_entry=False really
		## does permit an intervention at local step 0.
		force_observe = bool(force_observe_on_entry) and i == 0 and can_observe
		## The forced opening (observe, then intervene) spans resumed one-trial
		## phases too, so a phase-by-phase run opens exactly like a single run.
		force_intervene = (fresh_phase and i == 1) or (
			not fresh_phase and opening_forced_observe and len(type_trial) == 1)
		if force_observe and fresh_phase:
			opening_forced_observe = True

		if not can_observe:
			choose_observe = False
		elif force_intervene:
			choose_observe = False
		elif force_observe:
			choose_observe = True
		else:
			choose_observe = uniform < epsilon_coverage

		_log = {
			'trial': i,
			'epsilon_raw': float(epsilon_raw),
			'epsilon': float(epsilon_coverage),
			'u': float(uniform),
			'exhausted': bool(not can_observe),
			'n_obs_before': int(len(observational_samples)),
			'forced': ('observe' if force_observe else
						('intervene' if force_intervene else None)),
		}

		if choose_observe:
			observed += 1
			type_trial.append(0)
			## Collect a FRESH batch of previously unseen rows and append it to
			## the current observational dataset. Batches are disjoint
			## positional slices; concatenate with ignore_index because the
			## correctness property is disjointness, not label uniqueness.
			_obs_start = obs_cursor
			new_observational_samples, obs_cursor = observe(
				full_observational_samples, obs_cursor,
				num_additional_observations, obs_cap)
			num_observations_collected += int(len(new_observational_samples))

			observational_samples = pd.concat(
				[observational_samples, new_observational_samples],
				ignore_index=True)

			## Refit the models for the conditional distributions. For a
			## CoarsenedGraph this also refreshes the graph's observational
			## data and invalidates its memoized do-estimator cache.
			functions = graph.refit_models(observational_samples)

			## Clear the point-wise causal-prior caches BEFORE rebuilding the
			## mean/var functions: they are keyed by str(x) and would otherwise
			## return pre-observation values at every already-queried point.
			for _arm_cache in x_dict_mean.values():
				_arm_cache.clear()
			for _arm_cache in x_dict_var.values():
				_arm_cache.clear()

			## Update the mean functions and var functions given the current set of observational data. This is updating the prior.
			if any_prior:
				mean_functions_list, var_functions_list = refresh_priors()
			else:
				mean_functions_list = [None] * len(exploration_set)
				var_functions_list = [None] * len(exploration_set)

			## Rebuild every arm's causal GP now, so the returned state stays
			## internally consistent even if the run ends right here. The
			## intervene branch below skips its own all-arm rebuild while this
			## flag holds, so an observation costs exactly one rebuild.
			for s in range(len(exploration_set)):
				model_list[s] = update_BO_models(mean_functions_list[s], var_functions_list[s],
													data_x_list[s], data_y_list[s], prior_mask[s])
			models_fresh = True

			## Update current optimal solution. If I observe the cost and the optimal y are the same of the previous trial
			global_opt.append(global_opt[-1])
			current_cost.append(current_cost[-1])

			_log.update(type='observe', obs_start=int(_obs_start),
						obs_end=int(obs_cursor),
						n_obs_after=int(len(observational_samples)),
						arm=None, x=None,
						cum_cost=float(current_cost[-1]),
						incumbent=float(global_opt[-1]))
			trial_log.append(_log)


		else:
			type_trial.append(1)
			trial_intervened += 1
			## When we decide to intervene we need to compute the acquisition functions based on the GP models and decide the variable/variables to intervene
			## together with their interventional data

			## Define list to store info
			y_acquisition_list = [None]*len(exploration_set)
			x_new_list = [None]*len(exploration_set)

			## This is the global opt from previous iteration
			current_global = find_current_global(current_best_y, dict_interventions, task)


			## If in the previous trial we have observed we want to update all the BO models as the mean functions and var functions computed
			## via the DO calculus are changed
			## If in the previous trial we have intervened we want to update only the BO model for the intervention for which we have collected additional data
			## force_rebuild_all is set by RCCBO on resumed phases (and after partition refinement) to invalidate stale GP priors from the previous phase.
			## The causal prior is normally built by the first observe trial.
			## When the observational budget is already spent at phase entry
			## there is no such trial, so build it here from the data we have
			## -- otherwise update_BO_models receives mean_function=None.
			## Lazy on purpose: for every run that does observe at step 0 this
			## is a no-op, so existing trajectories are untouched.
			if any_prior and any(f is None for f, m in zip(mean_functions_list, prior_mask) if m):
				mean_functions_list, var_functions_list = refresh_priors()

			_force_rebuild = (state or {}).pop('force_rebuild_all', False) if state else False
			## `models_fresh` guards against rebuilding every arm twice: when the
			## previous trial was an observation, its branch already rebuilt
			## every arm from the enlarged dataset.
			if _force_rebuild:
				for s in range(len(exploration_set)):
					model_list[s] = update_BO_models(mean_functions_list[s], var_functions_list[s], data_x_list[s], data_y_list[s], prior_mask[s])
			elif models_fresh:
				pass
			elif len(type_trial) < 2 or type_trial[-2] == 0:
				for s in range(len(exploration_set)):
					model_list[s] = update_BO_models(mean_functions_list[s], var_functions_list[s], data_x_list[s], data_y_list[s], prior_mask[s])
			else:
				model_list[index] = update_BO_models(mean_functions_list[index],
																var_functions_list[index],
																data_x_list[index], data_y_list[index], prior_mask[index])


			## Compute acquisition function given the updated BO models for the interventional data
			## Notice that we use current_global and the costs to compute the acquisition functions
			for s in range(len(exploration_set)):
				y_acquisition_list[s], x_new_list[s] = find_next_y_point(space_list[s], model_list[s], current_global, exploration_set[s], costs, task = task)


			## Selecting the variable to intervene based on the values of the acquisition functions
			var_to_intervene = exploration_set[np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]]
			index = np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]


			## Evaluate the target function at the new point
			y_new = target_function_list[index](x_new_list[index])

			logger.debug('Selected intervention: %s', var_to_intervene)
			logger.debug('Selected point: %s', x_new_list[index])
			logger.debug('Target function at selected point: %s', y_new)

			## Callback for RCCBO: record full-variable sample
			if intervention_callback is not None:
				intervention_callback(var_to_intervene, x_new_list[index], y_new, graph.define_SEM)

			## Append the new data and set the new dataset of the BO model
			data_x, data_y_x = add_data([data_x_list[index], data_y_list[index]],
												  [x_new_list[index], y_new])

			data_x_list[index] = np.vstack((data_x_list[index], x_new_list[index]))
			data_y_list[index] = np.vstack((data_y_list[index], y_new))

			model_list[index].set_data(data_x, data_y_x)


			## Compute cost
			# Use the real ordered variable collection (exploration_set[index], still
			# held by var_to_intervene here) rather than dict_interventions[index],
			# which is a name-concatenated label string ('P1', 'RGB', 'P1P2'). The
			# label only works as a per-variable index when every variable name is a
			# single character; multi-char labels (the coarse clusters, e.g. 'BC',
			# 'ABC') make get_new_dict_x walk the string's characters and break.
			x_new_dict = get_new_dict_x(x_new_list[index], var_to_intervene)
			cumulative_cost += total_cost(var_to_intervene, costs, x_new_dict)
			var_to_intervene = dict_interventions[index]
			current_cost.append(cumulative_cost)


	 		## Update the dict storing the current optimal solution
			current_best_x[var_to_intervene].append(x_new_list[index][0][0])
			current_best_y[var_to_intervene].append(y_new[0][0])


			## Find the new current global optima
			current_global = find_current_global(current_best_y, dict_interventions, task)
			global_opt.append(current_global)

			logger.debug('Current global optimum: %s', current_global)

			## Optimise BO model given the new data
			model_list[index].optimize()

			models_fresh = False
			_log.update(type='intervene', obs_start=None, obs_end=None,
						n_obs_after=int(len(observational_samples)),
						arm='+'.join(exploration_set[index]),
						x=[float(v) for v in np.ravel(x_new_list[index])],
						y_new=float(np.ravel(y_new)[0]),
						cum_cost=float(current_cost[-1]),
						incumbent=float(global_opt[-1]))
			trial_log.append(_log)

	## Compute total time for the loop
	total_time = time.perf_counter() - start_time

	results = (current_cost, current_best_x, current_best_y, global_opt, observed, total_time)

	if return_state:
		state_out = {
			'current_cost': current_cost,
			'global_opt': global_opt,
			'current_best_x': current_best_x,
			'current_best_y': current_best_y,
			'x_dict_mean': x_dict_mean,
			'x_dict_var': x_dict_var,
			'dict_interventions': dict_interventions,
			'observed': observed,
			'trial_intervened': trial_intervened,
			'cumulative_cost': cumulative_cost,
			'target_function_list': target_function_list,
			'space_list': space_list,
			'model_list': model_list,
			'type_trial': type_trial,
			'mean_functions_list': mean_functions_list,
			'var_functions_list': var_functions_list,
			'functions': functions,
			'observational_samples': observational_samples,
			'index': index,
			'data_x_list': data_x_list,
			'data_y_list': data_y_list,
			'obs_cursor': int(obs_cursor),
			'initial_obs_cursor': int(initial_obs_cursor),
			'obs_cap': int(obs_cap),
			'num_observations_collected': int(num_observations_collected),
			'models_fresh': bool(models_fresh),
			'trial_log': trial_log,
			'prior_mask': list(prior_mask),
			'opening_forced_observe': bool(opening_forced_observe),
			'fixed_priors': dict(fixed_priors),
		}
		return results, state_out

	return results
