"""
Standard Bayesian Optimization baseline using botorch.

Runs vanilla BO (GP + Expected Improvement) on the manipulative variables
of a causal graph, without any causal knowledge. This is the same algorithm
as mcbo's "EI" mode (which uses botorch's FixedNoiseGP + EI internally),
but without mcbo's hard wandb dependency.

The baseline treats the SEM as a black-box function:
    f(B, D, E) → Y    (via do-intervention + forward sampling)

This provides a fair comparison point: how well does standard BO do
when it has NO causal structure, versus CBO/CCBO/RCCBO which exploit it.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll

from ccbo.generic_do import intervene, sample_from_model


def _evaluate_sem(sem_fn, manip_vars, x_values, num_mc=200, seed=None):
    """
    Evaluate E[Y | do(manip_vars = x_values)] via MC sampling from the SEM.

    Parameters
    ----------
    sem_fn : callable
        Returns the SEM OrderedDict (e.g., CompleteGraph.define_SEM).
    manip_vars : list of str
        Variable names to intervene on.
    x_values : array-like
        Values for the intervention, same order as manip_vars.
    num_mc : int
        Number of MC samples for estimating E[Y].
    seed : int, optional

    Returns
    -------
    float
        Estimated E[Y | do(manip_vars = x_values)].
    """
    model = sem_fn()
    intervention_dict = {var: float(val)
                         for var, val in zip(manip_vars, x_values)}
    mutilated = intervene(intervention_dict, model)

    if seed is not None:
        np.random.seed(seed)

    y_samples = []
    for _ in range(num_mc):
        sample = sample_from_model(mutilated)
        y_samples.append(sample['Y'])

    return float(np.mean(y_samples))


def _fit_gp(X, Y):
    """Fit a SingleTaskGP model (equivalent to mcbo's fit_gp_model)."""
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def bo_baseline(experiment, total_trials, num_init=5,
                task='min', num_mc=200, seed=9):
    """
    Run standard BO (GP + EI) on the manipulative variables of a graph.

    Parameters
    ----------
    experiment : str
        Graph name (e.g. 'CompleteGraph').
    total_trials : int
        Total optimization budget.
    num_init : int
        Number of initial random evaluations (Latin hypercube).
    task : str
        'min' or 'max'.
    num_mc : int
        MC samples per SEM evaluation.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        'global_opt': list of best Y values per trial
        'current_cost': list of cumulative costs (uniform cost = 1 per trial)
        'total_time': float
        'method': str
        'experiment': str
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Load graph ---
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'cbo', 'data', experiment)
    observational_samples = pd.read_pickle(
        os.path.join(data_path, 'observations.pkl'))[:100]

    from ccbo.run_experiment import get_original_graph
    original_graph = get_original_graph(experiment, observational_samples)

    sem_fn = original_graph.define_SEM
    _, _, manip_vars = original_graph.get_sets()
    dict_ranges = original_graph.get_interventional_ranges()

    manip_vars = sorted(manip_vars)
    d = len(manip_vars)

    # Bounds in original space
    lower = torch.tensor([dict_ranges[v][0] for v in manip_vars], dtype=torch.double)
    upper = torch.tensor([dict_ranges[v][1] for v in manip_vars], dtype=torch.double)
    # Botorch works in [0,1]^d internally; we'll map back and forth
    bounds_unit = torch.stack([torch.zeros(d, dtype=torch.double),
                               torch.ones(d, dtype=torch.double)])

    def to_original(x_unit):
        """Map [0,1]^d → original ranges."""
        return lower + x_unit * (upper - lower)

    def evaluate(x_unit):
        """Evaluate the SEM at a point in [0,1]^d."""
        x_orig = to_original(x_unit)
        return _evaluate_sem(sem_fn, manip_vars, x_orig.numpy(), num_mc=num_mc)

    t_start = time.perf_counter()

    # --- Initial design (random in [0,1]^d) ---
    X = torch.rand(num_init, d, dtype=torch.double)
    Y = torch.tensor([evaluate(X[i]) for i in range(num_init)],
                      dtype=torch.double).unsqueeze(-1)

    # Track best
    if task == 'min':
        best_vals = [float(Y[:i+1].min()) for i in range(num_init)]
    else:
        best_vals = [float(Y[:i+1].max()) for i in range(num_init)]

    print(f"BO-botorch: {num_init} initial evals, best so far: {best_vals[-1]:.4f}")

    # --- BO loop ---
    for trial in range(num_init, total_trials):
        # Fit GP
        try:
            gp = _fit_gp(X, Y)
        except Exception as e:
            print(f"  GP fitting failed at trial {trial}: {e}. Using random point.")
            x_new = torch.rand(1, d, dtype=torch.double)
            y_new = evaluate(x_new[0])
            X = torch.cat([X, x_new])
            Y = torch.cat([Y, torch.tensor([[y_new]], dtype=torch.double)])
            if task == 'min':
                best_vals.append(min(best_vals[-1], y_new))
            else:
                best_vals.append(max(best_vals[-1], y_new))
            continue

        # Acquisition function
        # For minimization: fit GP on -Y and maximize EI (standard trick)
        if task == 'min':
            gp_acq = _fit_gp(X, -Y)
            best_f = (-Y).max().item()
        else:
            gp_acq = gp
            best_f = Y.max().item()
        acq = LogExpectedImprovement(model=gp_acq, best_f=best_f)

        # Optimize acquisition
        try:
            x_new, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds_unit,
                q=1,
                num_restarts=10 * d,
                raw_samples=100 * d,
            )
        except Exception as e:
            print(f"  Acq optimization failed at trial {trial}: {e}. Using random.")
            x_new = torch.rand(1, d, dtype=torch.double)

        # Evaluate
        y_new = evaluate(x_new[0])
        X = torch.cat([X, x_new])
        Y = torch.cat([Y, torch.tensor([[y_new]], dtype=torch.double)])

        if task == 'min':
            best_vals.append(min(best_vals[-1], y_new))
        else:
            best_vals.append(max(best_vals[-1], y_new))

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{total_trials}: best = {best_vals[-1]:.4f}")

    t_elapsed = time.perf_counter() - t_start

    return {
        'global_opt': best_vals,
        'current_cost': list(range(1, total_trials + 1)),
        'total_time': t_elapsed,
        'method': 'BO-botorch',
        'experiment': experiment,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='CompleteGraph')
    parser.add_argument('--total-trials', default=40, type=int)
    parser.add_argument('--seed', default=9, type=int)
    args = parser.parse_args()

    result = bo_baseline(
        experiment=args.experiment,
        total_trials=args.total_trials,
        seed=args.seed,
    )

    print(f"\nFinal best Y: {result['global_opt'][-1]:.4f}")
    print(f"Time: {result['total_time']:.1f}s")
