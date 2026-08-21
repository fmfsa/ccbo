## Import basic python packages
import time
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer

## Import defined functions
from ccbo.cbo.utils import *


def NonCausal_BO(num_trials, graph, dict_ranges, interventional_data_x, interventional_data_y, costs, 
			observational_samples, functions, min_intervention_value, min_y, intervention_variables, Causal_prior=False,
			target_evaluator=None, task='min', intervention_callback=None):
	"""Standard (non-causal) BO baseline over a single joint arm.

	The arm is ``intervention_variables`` intervened on jointly, so under
	per-variable unit costs this pays ``len(intervention_variables)`` per
	trial -- more than a per-arm causal method. That is a property of the
	baseline, not an accounting bug, and matters on cost-indexed plots.

	Parameters
	----------
	intervention_callback : callable, optional
		Called after each intervention with ``(intervention_vars, x_new,
		y_new, sem_fn)`` -- the same signature ``CBO`` uses -- so runners can
		record the played arm and values per trial.
	target_evaluator : callable, optional
		Exact population evaluator with signature ``(arm, values) -> float``,
		the same hook ``CBO`` takes. Without it the target is evaluated by
		simulation, which would make BO rows noisy while CBO/QCBO rows on the
		same figure are exact.
	task : {'min'}
		Minimization only: the incumbent tracking below is hardcoded to
		``np.min``. Asserted rather than generalized.
	"""
	if task != 'min':
		raise NotImplementedError(
			"vendored NonCausal_BO is minimization-only; got task=%r" % (task,))

	## Compute input space dimension
	input_space = len(intervention_variables)

	## Initialise matrices for storing
	current_best_x= np.zeros((num_trials + 1, input_space))
	current_best_y = np.zeros((num_trials + 1, 1))
	current_cost = np.zeros((num_trials + 1, 1))

	## Get do function and mean/var functions only when causal prior is used
	mean_function_do, var_function_do = None, None
	if Causal_prior:
		function_name = get_do_function_name(intervention_variables)
		do_function = graph.get_all_do()[function_name]
		mean_function_do, var_function_do = mean_var_do_functions(do_function, observational_samples, functions)

	## Get interventional data
	data_x = interventional_data_x.copy()
	data_y = interventional_data_y.copy()

	
	## Assign the initial values 
	current_cost[0] = 0.
	current_best_y[0] = min_y
	current_best_x[0] = min_intervention_value
	cumulative_cost = 0.


	## Compute target function and space parameters
	target_function, space_parameters = Intervention_function(get_interventional_dict(intervention_variables),
																model = graph.define_SEM(), target_variable = 'Y', 
																min_intervention = list_interventional_ranges(graph.get_interventional_ranges(), intervention_variables)[0],
																max_intervention = list_interventional_ranges(graph.get_interventional_ranges(), intervention_variables)[1])

	## Exact population target, mirroring ccbo/cbo/cbo.py's target_evaluator hook.
	if target_evaluator is not None:
		_arm = tuple(intervention_variables)
		target_function = lambda value, arm=_arm: np.asarray(
			target_evaluator(arm, value), dtype=float)[np.newaxis, np.newaxis]


	if Causal_prior==False:
		#### Define the model without Causal prior
		gpy_model = GPy.models.GPRegression(data_x, data_y, GPy.kern.RBF(input_space, lengthscale=1., variance=1.), noise_var=1e-10)
		emukit_model= GPyModelWrapper(gpy_model)
	else:
		#### Define the model with Causal prior
		mf = GPy.core.Mapping(input_space, 1)
		mf.f = lambda x: mean_function_do(x)
		mf.update_gradients = lambda a, b: None
		kernel = CausalRBF(input_space, variance_adjustment=var_function_do, lengthscale=1., variance=1., rescale_variance = 1., ARD = False)
		gpy_model = GPy.models.GPRegression(data_x, data_y, kernel, noise_var=1e-10, mean_function=mf)
		emukit_model = GPyModelWrapper(gpy_model)


	## BO loop
	start_time = time.perf_counter()
	for j in range(num_trials):
		print('Iteration', j)
		## Optimize model and get new evaluation point
		emukit_model.optimize()
		acquisition = ExpectedImprovement(emukit_model)
		optimizer = GradientAcquisitionOptimizer(space_parameters)
		x_new, _ = optimizer.optimize(acquisition)
		y_new = target_function(x_new)

		if intervention_callback is not None:
			intervention_callback(intervention_variables, x_new, y_new,
									graph.define_SEM)

		## Append the data
		data_x = np.append(data_x, x_new, axis=0)
		data_y = np.append(data_y, y_new, axis=0)
		emukit_model.set_data(data_x, data_y)

		## Compute cost
		x_new_dict = get_new_dict_x(x_new, intervention_variables)
		cumulative_cost += total_cost(intervention_variables, costs, x_new_dict)
		current_cost[j + 1] = cumulative_cost

		## Get current optimum
		results = np.concatenate((emukit_model.X, emukit_model.Y), axis =1)
		current_best_y[j + 1 ] = np.min(results[:,input_space])
		if results[results[:,input_space] == np.min(results[:,input_space]), :input_space].shape[0] > 1:
			best_x = results[results[:,input_space] == np.min(results[:,input_space]), :input_space][0]
		else:
			best_x = results[results[:,input_space] == np.min(results[:,input_space]), :input_space]
		print('Current best Y', np.min(results[:,input_space]))

	total_time = time.perf_counter() - start_time

	return (current_cost, current_best_x, current_best_y, total_time)


