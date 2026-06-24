"""Tier1Graph GraphStructure — the lossless-coarsening self-healing benchmark.

Purpose-built for the Tier-1 (prior-bias) robustness demonstration. It replaces
CompleteGraph in that role: CompleteGraph admits *no* lossless coarse partition
(its only non-trivial coarsening, {B}|{D,E}, splits the optimum do(B,D) and floors
far below the fine partition), so its Tier-1 figure conflates robustness with the
price of coarsening. Tier1Graph is designed so the coarse partition pays
**negligible** price of coarsening, isolating the self-healing claim.

Causal model
------------
Manipulable: B, D, E   (each in [0, 1])
Non-manip:   C         (an observed mediator)
Target:      Y
No latent confounders.

Observable DAG (used by adjustment / C-DAG identification):
    B -> C    D -> C        (C is a shared mediator of B and D)
    C -> Y    B -> Y    D -> Y    E -> Y

Key design properties
---------------------
* **Lossless coarse partition.** Every structural weight on Y is negative and C
  increases in B, D, so the global optimum is the joint arm do(B=1, D=1, E=1).
  Under the coarse partition ``{B,D} | {E}`` that optimum is the union of *whole*
  clusters, so by Prop. 5 coarsening loses (almost) nothing: coarse reaches the
  same final Y as the fine partition. (Contrast CompleteGraph, whose optimum
  splits a cluster and floors the coarse partition far below the fine one.)

* **Misspecification that bites the OPTIMAL arm yet is immune at coarse.** The
  variant ``Tier1Graph_NoBC`` deletes the edge B -> C. Because the mediator C
  feeds Y through C -> Y, which is *not* cut by intervening on {B,D,E}, deleting
  B -> C biases the prior mean of every B-containing arm — *including the optimal
  arm do(B,D,E)*: the assumed graph thinks C ignores B, so it mis-predicts Y. No
  arm is added or removed (|ES| unchanged), so this is a convergence-speed cost
  that the interventional data progressively correct — it self-heals.

* **Coarse immunity by redundancy.** Under ``{B,D} | {E}`` the cluster edge
  {B,D} -> C is kept alive by D -> C even when B -> C is deleted (exactly the
  redundancy that protects CompleteGraph's coarse partition under NoCD). The
  quotient C-DAG is therefore byte-identical under Tier1Graph and Tier1Graph_NoBC
  and the QCBO trajectories coincide exactly (Prop. 2).

The misspecification lives only in the *assumed graph metadata*
(``get_dag_edges_from_sem`` in ccbo/coarsening.py); the data-generating SEM below
is the same for both DAGs.
"""

import numpy as np
from collections import OrderedDict

from . import graph
from ccbo.cbo.utils.utils import fit_single_GP_model
from ccbo.generic_do import make_do_function

from .Tier1Graph_CostFunctions import define_costs


# Input/observation noise scales.
_SIGMA_INPUT = 0.15
_SIGMA_C = 0.08
_SIGMA_Y = 0.02

# Mediator C = clip(0.5 + A_BC*(B-0.5) + A_DC*(D-0.5) + noise). Both manipulable
# parents drive C; D -> C is what keeps the coarse {B,D} -> C edge alive when
# B -> C is deleted (redundancy => coarse immunity).
_A_BC = 0.8
_A_DC = 0.8

# Y = W_C*C + W_B*B + W_D*D + W_E*E + noise; all negative so do(B=1,D=1,E=1)
# is optimal. W_C dominates, so the B -> C -> Y path carries most of B's effect:
# deleting B -> C therefore visibly mis-primes the optimal arm's prior.
_W_C = -2.0
_W_B = -0.5
_W_D = -0.5
_W_E = -1.5


class Tier1Graph(graph.GraphStructure):
    """GraphStructure for the lossless-coarsening self-healing benchmark."""

    def __init__(self, observational_samples):
        self.B = np.asarray(observational_samples['B'])[:, np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:, np.newaxis]
        self.E = np.asarray(observational_samples['E'])[:, np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

    # ------------------------------------------------------------------
    # SEM
    # ------------------------------------------------------------------
    def define_SEM(self):
        # Epsilon layout: 0: eB  1: eD  2: eE  3: eC  4: eY
        def fB(epsilon, **kw):
            return float(np.clip(0.5 + _SIGMA_INPUT * epsilon[0], 0.0, 1.0))

        def fD(epsilon, **kw):
            return float(np.clip(0.5 + _SIGMA_INPUT * epsilon[1], 0.0, 1.0))

        def fE(epsilon, **kw):
            return float(np.clip(0.5 + _SIGMA_INPUT * epsilon[2], 0.0, 1.0))

        def fC(epsilon, B, D, **kw):
            return float(np.clip(0.5 + _A_BC * (B - 0.5) + _A_DC * (D - 0.5)
                                 + _SIGMA_C * epsilon[3], 0.0, 1.0))

        def fY(epsilon, C, B, D, E, **kw):
            return (_W_C * C + _W_B * B + _W_D * D + _W_E * E
                    + _SIGMA_Y * epsilon[4])

        return OrderedDict([
            ('B', fB), ('D', fD), ('E', fE),
            ('C', fC),
            ('Y', fY),
        ])

    # ------------------------------------------------------------------
    # Exploration / interventional metadata
    # ------------------------------------------------------------------
    def get_sets(self):
        MIS = [['B'], ['D'], ['E'],
               ['B', 'D'], ['B', 'E'], ['D', 'E'],
               ['B', 'D', 'E']]
        POMIS = MIS  # no confounding; every subset is identifiable
        manipulative_variables = ['B', 'D', 'E']
        return MIS, POMIS, manipulative_variables

    def get_set_BO(self):
        return ['B', 'D', 'E']

    def get_interventional_ranges(self):
        return OrderedDict([
            ('B', [0.0, 1.0]),
            ('D', [0.0, 1.0]),
            ('E', [0.0, 1.0]),
        ])

    # ------------------------------------------------------------------
    # GP fits
    # ------------------------------------------------------------------
    def fit_all_models(self):
        return self._fit_models_from(self.B, self.D, self.E, self.C, self.Y)

    def refit_models(self, observational_samples):
        B = np.asarray(observational_samples['B'])[:, np.newaxis]
        D = np.asarray(observational_samples['D'])[:, np.newaxis]
        E = np.asarray(observational_samples['E'])[:, np.newaxis]
        C = np.asarray(observational_samples['C'])[:, np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        return self._fit_models_from(B, D, E, C, Y)

    @staticmethod
    def _fit_models_from(B, D, E, C, Y):
        functions = {}
        spec = [
            ('gp_C_B_D', np.hstack((B, D))),     # mediator model C | B, D
            ('gp_Y_B', np.hstack((B,))),
            ('gp_Y_D', np.hstack((D,))),
            ('gp_Y_E', np.hstack((E,))),
            ('gp_Y_B_D', np.hstack((B, D))),
            ('gp_Y_B_E', np.hstack((B, E))),
            ('gp_Y_D_E', np.hstack((D, E))),
            ('gp_Y_B_D_E', np.hstack((B, D, E))),
        ]
        params = [1.0, 1.0, 1.0, False]
        for name, X in spec:
            out = C if name == 'gp_C_B_D' else Y
            functions[name] = fit_single_GP_model(X, out, params)
        return functions

    # ------------------------------------------------------------------
    # Costs & do-functions
    # ------------------------------------------------------------------
    def get_cost_structure(self, type_cost):
        return define_costs(type_cost)

    def get_all_do(self):
        """Oracle do-functions for the legacy (raw-graph) CBO path.

        QCBO does not use these — when this graph is wrapped in a
        CoarsenedGraph the do-functions are derived at the C-DAG level by
        ``make_cdag_do_function`` — but we provide them so the GraphStructure
        interface is complete.
        """
        sem = self.define_SEM
        return {
            'compute_do_B': make_do_function(sem, ['B'], num_mc_samples=2000),
            'compute_do_D': make_do_function(sem, ['D'], num_mc_samples=2000),
            'compute_do_E': make_do_function(sem, ['E'], num_mc_samples=2000),
            'compute_do_BD': make_do_function(sem, ['B', 'D'], num_mc_samples=2000),
            'compute_do_BE': make_do_function(sem, ['B', 'E'], num_mc_samples=2000),
            'compute_do_DE': make_do_function(sem, ['D', 'E'], num_mc_samples=2000),
            'compute_do_BDE': make_do_function(sem, ['B', 'D', 'E'], num_mc_samples=2000),
        }
