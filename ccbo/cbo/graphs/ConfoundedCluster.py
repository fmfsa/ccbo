"""ConfoundedCluster GraphStructure.

A small, fully self-contained synthetic benchmark whose only purpose is the
Tier-2 (arm-deletion) robustness demonstration: an intra-cluster
misspecification that *deletes the optimal intervention target* from a
fine-grained method's exploration set, yet is provably invisible to QCBO under
a coarse partition.

Causal model
------------
Manipulable: A, B, C   (each in [0, 1]; all direct parents of Y)
Target:      Y
Hidden:      U         (latent confounder on B and C; induces observational
                        corr(B, C) > 0 *without* any causal B -> C edge)

Observable DAG (used by adjustment / C-DAG identification):
    A -> Y    B -> Y    C -> Y
    B <-> C   (bidirected, from projecting out the latent U)

The structural equation for Y is linear with the strongest (negative) weight
on B, so the best single-target intervention is do(B):

    E[Y | do(B=1)] = -3.75   (optimum among singletons)
    E[Y | do(C=1)] = -2.75
    E[Y | do(A=1)] = -2.50
    E[Y | do(B=1, C=1)] = -4.25   (the {B,C} cluster, which dominates)

The misspecified variant ``ConfoundedCluster_WrongBC`` adds a *spurious*
intra-cluster edge B -> C.  Together with the (real) U-induced bidirected
B <-> C this forms a bow, so P(Y | do(B)) is **not** identifiable from the
assumed fine DAG and the identity-partition exploration set loses its best
target {B} (Tier-2 arm damage; no amount of interventional data recovers it).

Under the coarse partition ``{A} | {B,C} | {Y}`` the bow lies *inside* the
{B,C} cluster.  Lee-2019 latent projection drops the B -> C edge, so the
quotient C-DAG is byte-identical under both assumed DAGs and QCBO's
trajectories coincide exactly (Prop. 2).  The cluster target do(B,C) also
dominates every singleton, so coarsening loses nothing here.

The misspecification lives only in the *assumed graph metadata*
(``get_dag_edges_from_sem`` in ccbo/coarsening.py); the data-generating SEM
below is the same for both DAGs.  This SEM doubles as the data-generating
process for observations.pkl (see
ccbo/cbo/data/ConfoundedCluster/generate_observations.py).
"""

import numpy as np
from collections import OrderedDict

from . import graph
from ccbo.cbo.utils.utils import fit_single_GP_model
from ccbo.generic_do import make_do_function

from .ConfoundedCluster_CostFunctions import define_costs


# Latent-confounder strength on B, C.  Larger => stronger observational
# corr(B, C), which is what gives the WrongBC misspecification bite.
_SIGMA_U = 0.15
_SIGMA_INPUT = 0.10
_SIGMA_Y = 0.02

# Y = W_A * A + W_B * B + W_C * C + noise.  |W_B| is the largest, so do(B) is
# the best single-target intervention; see module docstring for arm values.
_W_A = -0.5
_W_B = -3.0
_W_C = -1.0


class ConfoundedCluster(graph.GraphStructure):
    """GraphStructure for the synthetic bow benchmark (Tier-2 demonstration)."""

    def __init__(self, observational_samples):
        self.A = np.asarray(observational_samples['A'])[:, np.newaxis]
        self.B = np.asarray(observational_samples['B'])[:, np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

    # ------------------------------------------------------------------
    # SEM
    # ------------------------------------------------------------------
    def define_SEM(self):
        # Epsilon layout: 0: U   1: eA   2: eB   3: eC   4: eY
        def fU(epsilon, **kw):
            return _SIGMA_U * epsilon[0]

        def fA(epsilon, **kw):
            return float(np.clip(0.5 + _SIGMA_INPUT * epsilon[1], 0.0, 1.0))

        def fB(epsilon, U, **kw):
            return float(np.clip(0.5 + U + _SIGMA_INPUT * epsilon[2], 0.0, 1.0))

        def fC(epsilon, U, **kw):
            return float(np.clip(0.5 + U + _SIGMA_INPUT * epsilon[3], 0.0, 1.0))

        def fY(epsilon, A, B, C, **kw):
            return (_W_A * A + _W_B * B + _W_C * C
                    + _SIGMA_Y * epsilon[4])

        return OrderedDict([
            ('U', fU),
            ('A', fA), ('B', fB), ('C', fC),
            ('Y', fY),
        ])

    # ------------------------------------------------------------------
    # Exploration / interventional metadata
    # ------------------------------------------------------------------
    def get_sets(self):
        MIS = [['A'], ['B'], ['C'], ['B', 'C'], ['A', 'B', 'C']]
        POMIS = MIS  # every entry has a direct path to Y; none dominated
        manipulative_variables = ['A', 'B', 'C']
        return MIS, POMIS, manipulative_variables

    def get_set_BO(self):
        return ['A', 'B', 'C']

    def get_interventional_ranges(self):
        return OrderedDict([
            ('A', [0.0, 1.0]),
            ('B', [0.0, 1.0]),
            ('C', [0.0, 1.0]),
        ])

    # ------------------------------------------------------------------
    # GP fits
    # ------------------------------------------------------------------
    def fit_all_models(self):
        return self._fit_models_from(self.A, self.B, self.C, self.Y)

    def refit_models(self, observational_samples):
        A = np.asarray(observational_samples['A'])[:, np.newaxis]
        B = np.asarray(observational_samples['B'])[:, np.newaxis]
        C = np.asarray(observational_samples['C'])[:, np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        return self._fit_models_from(A, B, C, Y)

    @staticmethod
    def _fit_models_from(A, B, C, Y):
        functions = {}
        spec = [
            ('gp_Y_A', np.hstack((A,))),
            ('gp_Y_B', np.hstack((B,))),
            ('gp_Y_C', np.hstack((C,))),
            ('gp_Y_B_C', np.hstack((B, C))),
            ('gp_Y_A_B_C', np.hstack((A, B, C))),
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

        Built by Monte Carlo through the true SEM (``generic_do``).  QCBO does
        not use these — when this graph is wrapped in a CoarsenedGraph the
        do-functions are derived at the C-DAG level by ``make_cdag_do_function``
        — but we provide them so the GraphStructure interface is complete.
        """
        sem = self.define_SEM
        return {
            'compute_do_A': make_do_function(sem, ['A'], num_mc_samples=2000),
            'compute_do_B': make_do_function(sem, ['B'], num_mc_samples=2000),
            'compute_do_C': make_do_function(sem, ['C'], num_mc_samples=2000),
            'compute_do_BC': make_do_function(sem, ['B', 'C'], num_mc_samples=2000),
            'compute_do_ABC': make_do_function(sem, ['A', 'B', 'C'], num_mc_samples=2000),
        }
