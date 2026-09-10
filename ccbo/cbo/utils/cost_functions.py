"""Intervention cost as an emukit acquisition (denominator of EI / cost).

Engine v3 (2026-09): ``evaluate_with_gradients`` returns central-difference
gradients, exact (zero) for constant per-variable costs and correct for
value-dependent costs (``type_cost`` 3 / 4), instead of a hard-coded zero.
"""

import numpy as np

from emukit.core.acquisition import Acquisition

FD_STEP = 1e-6


class Cost(Acquisition):
    def __init__(self, costs_functions, evaluated_set):
        self.costs_functions = costs_functions
        self.evaluated_set = evaluated_set

    def evaluate(self, x):
        """Cost of every row of ``x`` as an ``(N, 1)`` array (a scalar or
        ``(N,)`` return of the per-variable cost functions is broadcast), so
        emukit's ``Quotient`` acquisition divides row by row."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(self.evaluated_set)
        if n == 1:
            cost = self.costs_functions[self.evaluated_set[0]](x)
        else:
            cost = sum(self.costs_functions[self.evaluated_set[i]](x[:, i])
                       for i in range(n))
        cost = np.asarray(cost, dtype=float)
        return np.reshape(cost * np.ones(x.shape[0]), (x.shape[0], 1))

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        x = np.asarray(x, dtype=float)
        x = np.atleast_2d(x)
        f = self.evaluate(x)
        grad = np.zeros(x.shape)
        for q in range(x.shape[1]):
            h = FD_STEP * np.maximum(1.0, np.abs(x[:, q]))
            xp = x.copy(); xp[:, q] += h
            xm = x.copy(); xm[:, q] -= h
            fp = np.ravel(np.asarray(self.evaluate(xp), dtype=float))
            fm = np.ravel(np.asarray(self.evaluate(xm), dtype=float))
            grad[:, q] = (fp - fm) / (2.0 * h)
        return f, grad


def total_cost(intervention_variables, costs, x_new_dict):
  total_cost = 0.
  for i in range(len(intervention_variables)):
    total_cost += costs[intervention_variables[i]](x_new_dict[intervention_variables[i]])
  return total_cost
