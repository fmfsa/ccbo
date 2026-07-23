"""FrontDoor GraphStructure (MinimalBench, Experiment B).

The minimal front-door SCM behind the prior-corruption-without-arm-loss
result. All structure, coefficients, and closed-form oracles are owned by
``ccbo/minibench.py``.

Causal model
------------
Manipulable: X1, M  (each in [-3, 3])
Target:      Y
Hidden:      U      (latent confounder on X1 and Y)

    X1 -> M -> Y     X1 <-> Y     (from projecting out U)

    M = B X1 + noise                                        B = 2
    Y = AMP (1 - exp(-(M-C)^2 / 2 W^2)) + G U + noise       AMP=2, C=1, W=0.3

The target is a narrow well, not a bowl — prior corruption only matters in
a global-search regime (see ccbo/minibench.py). Fine MIS = {X1}, {M} (the
joint is not minimal on a chain). do(M) is the optimal arm (y* = 0 at
M = C; backdoor adjustment through X1); do(X1) is identified by the
front-door functional through M (best value ~0.336, the noise-smoothed
well). The
coarse cluster {X1, M} forms the ToyGraph bow with Y (C1 -> Y, C1 <-> Y):
the whole-cluster arm runs on the uninformative tier yet still reaches
y* = 0 (the partition is lossless), in every condition.

Condition B1 assumes a spurious intra-cluster confounder X1 <-> M, which
costs BOTH fine arms their identification (prior tier drops, membership
untouched) while the quotient is unchanged. The misspecification lives
only in the assumed graph metadata; this SEM is the same for both
conditions and doubles as the data-generating process for
observations.pkl (see ccbo/cbo/data/FrontDoor/generate_observations.py).
"""

import numpy as np
from collections import OrderedDict

from . import graph
from ccbo.cbo.utils.utils import fit_single_GP_model
from ccbo.generic_do import make_do_function
from ccbo import minibench as mb

from .FrontDoor_CostFunctions import define_costs


class FrontDoor(graph.GraphStructure):
    """GraphStructure for the minimal front-door SCM."""

    def __init__(self, observational_samples):
        self.X1 = np.asarray(observational_samples['X1'])[:, np.newaxis]
        self.M = np.asarray(observational_samples['M'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

    # ------------------------------------------------------------------
    # SEM
    # ------------------------------------------------------------------
    def define_SEM(self):
        # Epsilon layout: 0: U   1: eX1   2: eM   3: eY
        def fU(epsilon, **kw):
            return mb.FD_SIGMA_U * epsilon[0]

        def fX1(epsilon, U, **kw):
            return float(U + mb.FD_SIGMA_1 * epsilon[1])

        def fM(epsilon, X1, **kw):
            return float(mb.FD_B * X1 + mb.FD_SIGMA_M * epsilon[2])

        def fY(epsilon, M, U, **kw):
            well = 1.0 - np.exp(-(M - mb.FD_C) ** 2 / (2 * mb.FD_W ** 2))
            return float(mb.FD_AMP * well + mb.FD_G * U
                         + mb.SIGMA_Y * epsilon[3])

        return OrderedDict([
            ('U', fU),
            ('X1', fX1), ('M', fM),
            ('Y', fY),
        ])

    # ------------------------------------------------------------------
    # Exploration / interventional metadata
    # ------------------------------------------------------------------
    def get_sets(self):
        # Chain: the joint {X1, M} is not an MIS (mutilating M severs X1).
        MIS = [['X1'], ['M']]
        POMIS = MIS
        manipulative_variables = list(mb.FD_MANIPULATIVE)
        return MIS, POMIS, manipulative_variables

    def get_set_BO(self):
        return list(mb.FD_MANIPULATIVE)

    def get_interventional_ranges(self):
        return OrderedDict([
            ('X1', list(mb.DOMAIN)),
            ('M', list(mb.DOMAIN)),
        ])

    # ------------------------------------------------------------------
    # GP fits
    # ------------------------------------------------------------------
    def fit_all_models(self):
        return self._fit_models_from(self.X1, self.M, self.Y)

    def refit_models(self, observational_samples):
        X1 = np.asarray(observational_samples['X1'])[:, np.newaxis]
        M = np.asarray(observational_samples['M'])[:, np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        return self._fit_models_from(X1, M, Y)

    @staticmethod
    def _fit_models_from(X1, M, Y):
        functions = {}
        spec = [
            ('gp_Y_X1', np.hstack((X1,))),
            ('gp_Y_M', np.hstack((M,))),
            ('gp_Y_X1_M', np.hstack((X1, M))),
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
            'compute_do_M': make_do_function(sem, ['M'], num_mc_samples=2000),
            'compute_do_X1M': make_do_function(sem, ['X1', 'M'],
                                               num_mc_samples=2000),
        }
