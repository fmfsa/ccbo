"""MediatedChain GraphStructure (MinimalBench, Experiment C).

The minimal mediated-chain SCM behind the price-of-coarsening and
refinement result. All structure, coefficients, the policy box, and the
closed-form oracles are owned by ``ccbo/minibench.py``.

Causal model
------------
Manipulable: X1 in [-3, 3];  X2 in [-2, 2]  (the POLICY BOX)
Target:      Y
No latent confounders.

    X1 -> X2 -> Y
    X2 = BETA X1 + noise            BETA = 2
    Y  = (X2 - C)^2 + noise         C = 4

The fine optimum do(X1 = C/BETA = 2) drives X2's natural response to C = 4,
outside the clampable box, reaching y* = SIGMA_2^2 = 0.04. The whole-cluster
arm do(X1, X2) clamps X2 into the box, so the best coarse value is
V(Pi) = (2 - 4)^2 = 4: the fixed coarse partition pays the analytic price
V(Pi) - y* = 3.96, and refinement (splitting {X1,X2}) buys it back.

Note the fine MIS is {X1}, {X2} only — the joint arm is NOT minimal on a
chain (mutilating X2 severs X1 from An(Y)); the joint intervention exists
only as the quotient cluster arm.

This SEM doubles as the data-generating process for observations.pkl (see
ccbo/cbo/data/MediatedChain/generate_observations.py).
"""

import numpy as np
from collections import OrderedDict

from . import graph
from ccbo.cbo.utils.utils import fit_single_GP_model
from ccbo.generic_do import make_do_function
from ccbo import minibench as mb

from .MediatedChain_CostFunctions import define_costs


class MediatedChain(graph.GraphStructure):
    """GraphStructure for the minimal mediated-chain SCM."""

    def __init__(self, observational_samples):
        self.X1 = np.asarray(observational_samples['X1'])[:, np.newaxis]
        self.X2 = np.asarray(observational_samples['X2'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

    # ------------------------------------------------------------------
    # SEM
    # ------------------------------------------------------------------
    def define_SEM(self):
        # Epsilon layout: 0: eX1   1: eX2   2: eY
        def fX1(epsilon, **kw):
            return float(mb.MC_SIGMA_1 * epsilon[0])

        def fX2(epsilon, X1, **kw):
            return float(mb.MC_BETA * X1 + mb.MC_SIGMA_2 * epsilon[1])

        def fY(epsilon, X2, **kw):
            return float((X2 - mb.MC_C) ** 2 + mb.SIGMA_Y * epsilon[2])

        return OrderedDict([
            ('X1', fX1), ('X2', fX2),
            ('Y', fY),
        ])

    # ------------------------------------------------------------------
    # Exploration / interventional metadata
    # ------------------------------------------------------------------
    def get_sets(self):
        # The joint {X1, X2} is not an MIS on a chain: once X2 is mutilated,
        # X1 is no longer an ancestor of Y.
        MIS = [['X1'], ['X2']]
        POMIS = MIS
        manipulative_variables = list(mb.MC_MANIPULATIVE)
        return MIS, POMIS, manipulative_variables

    def get_set_BO(self):
        return list(mb.MC_MANIPULATIVE)

    def get_interventional_ranges(self):
        return OrderedDict([
            ('X1', list(mb.DOMAIN)),
            ('X2', list(mb.MC_X2_BOX)),
        ])

    # ------------------------------------------------------------------
    # GP fits
    # ------------------------------------------------------------------
    def fit_all_models(self):
        return self._fit_models_from(self.X1, self.X2, self.Y)

    def refit_models(self, observational_samples):
        X1 = np.asarray(observational_samples['X1'])[:, np.newaxis]
        X2 = np.asarray(observational_samples['X2'])[:, np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        return self._fit_models_from(X1, X2, Y)

    @staticmethod
    def _fit_models_from(X1, X2, Y):
        functions = {}
        spec = [
            ('gp_Y_X1', np.hstack((X1,))),
            ('gp_Y_X2', np.hstack((X2,))),
            ('gp_Y_X1_X2', np.hstack((X1, X2))),
        ]
        params = [1.0, 1.0, 1.0, False]
        for name, X in spec:
            functions[name] = fit_single_GP_model(X, Y, params)
        return functions

    # ------------------------------------------------------------------
    # Costs & do-functions
    # ------------------------------------------------------------------
    def get_cost_structure(self, type_cost):
        return define_costs(type_cost)

    def get_all_do(self):
        sem = self.define_SEM
        return {
            'compute_do_X1': make_do_function(sem, ['X1'], num_mc_samples=2000),
            'compute_do_X2': make_do_function(sem, ['X2'], num_mc_samples=2000),
            'compute_do_X1X2': make_do_function(sem, ['X1', 'X2'],
                                                num_mc_samples=2000),
        }
