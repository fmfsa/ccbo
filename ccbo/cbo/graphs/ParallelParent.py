"""ParallelParent GraphStructure (MinimalBench, Experiments A + B).

The minimal parallel-parent SCM behind the paper's headline exploration-set
corruption result and the intra-cluster bow (prior corruption) result. All
structure, coefficients, and closed-form oracles are owned by
``ccbo/minibench.py`` — this class only realizes them as a GraphStructure
so the SCM plugs into the shared CoarsenedGraph + CBO backend.

Causal model
------------
Manipulable: X1, X2  (each in [-3, 3]; both direct parents of Y)
Target:      Y
Hidden:      U       (latent confounder on X1 and X2; induces observational
                      corr(X1, X2) = 0.64 without any causal X1 -> X2 edge)

    X1 -> Y    X2 -> Y    X1 <-> X2   (from projecting out U)

    Y = LAM (X1 - A)^2 + (X2 - B)^2 + noise,   A = 2, B = -2

The joint arm do(X1=A, X2=B) is uniquely optimal (y* = 0); each singleton
pays the free co-parent's bowl: best do(X1) = 4.25, best do(X2) = 4.25 LAM.
See ccbo/minibench.py for the misspecification conditions A1-A3 and the
analytic gap G(LAM) = 4.25 LAM.

The misspecifications live only in the *assumed graph metadata* (registered
by ``minibench.register_variants``); the data-generating SEM below is the
same for every condition (the fixed-objective seam). This SEM doubles as the
data-generating process for observations.pkl (see
ccbo/cbo/data/ParallelParent/generate_observations.py).
"""

import numpy as np
from collections import OrderedDict

from . import graph
from ccbo.cbo.utils.utils import fit_single_GP_model
from ccbo.generic_do import make_do_function
from ccbo import minibench as mb

from .ParallelParent_CostFunctions import define_costs


class ParallelParent(graph.GraphStructure):
    """GraphStructure for the minimal parallel-parent SCM."""

    def __init__(self, observational_samples):
        self.X1 = np.asarray(observational_samples['X1'])[:, np.newaxis]
        self.X2 = np.asarray(observational_samples['X2'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

    # ------------------------------------------------------------------
    # SEM
    # ------------------------------------------------------------------
    def define_SEM(self):
        # Epsilon layout: 0: U   1: eX1   2: eX2   3: eY
        def fU(epsilon, **kw):
            return mb.PP_SIGMA_U * epsilon[0]

        def fX1(epsilon, U, **kw):
            return float(U + mb.PP_SIGMA_IN * epsilon[1])

        def fX2(epsilon, U, **kw):
            return float(U + mb.PP_SIGMA_IN * epsilon[2])

        def fY(epsilon, X1, X2, **kw):
            return float(mb.PP_LAM * (X1 - mb.PP_A) ** 2
                         + (X2 - mb.PP_B) ** 2
                         + mb.SIGMA_Y * epsilon[3])

        return OrderedDict([
            ('U', fU),
            ('X1', fX1), ('X2', fX2),
            ('Y', fY),
        ])

    # ------------------------------------------------------------------
    # Exploration / interventional metadata
    # ------------------------------------------------------------------
    def get_sets(self):
        MIS = [['X1'], ['X2'], ['X1', 'X2']]
        POMIS = MIS
        manipulative_variables = list(mb.PP_MANIPULATIVE)
        return MIS, POMIS, manipulative_variables

    def get_set_BO(self):
        return list(mb.PP_MANIPULATIVE)

    def get_interventional_ranges(self):
        return OrderedDict([
            ('X1', list(mb.DOMAIN)),
            ('X2', list(mb.DOMAIN)),
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
        """Oracle do-functions for the legacy (raw-graph) CBO path.

        Built by Monte Carlo through the true SEM (``generic_do``). QCBO
        derives its do-functions at the C-DAG level instead.
        """
        sem = self.define_SEM
        return {
            'compute_do_X1': make_do_function(sem, ['X1'], num_mc_samples=2000),
            'compute_do_X2': make_do_function(sem, ['X2'], num_mc_samples=2000),
            'compute_do_X1X2': make_do_function(sem, ['X1', 'X2'],
                                                num_mc_samples=2000),
        }
