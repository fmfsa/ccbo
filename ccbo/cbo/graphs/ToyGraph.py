"""ToyGraph GraphStructure — the CBO-2020 toy benchmark (Aglietti et al. 2020).

Causal model
------------
Manipulable: X in [-5, 5],  Z in [-5, 20]
Target:      Y   (task: min)
Hidden:      U   (latent confounder on X and Y)

Observable DAG (matches ``get_dag_edges_from_sem('ToyGraph')`` in
ccbo/coarsening.py):
    X -> Z -> Y
    X <-> Y   (bidirected, from projecting out the latent U; declared as
               ('U', ['X', 'Y']) in ``get_hidden_confounders``)

SEM (same as every vendored reimplementation of the CBO-2020 toy, with the
latent confounder made explicit):
    U = eps_U                                (eps_U ~ N(0, 1))
    X = U + eps_X
    Z = exp(-X) + eps_Z
    Y = cos(Z) - exp(-Z/20) + U + eps_Y

Ground truth: E[Y | do(Z = z)] = cos(z) - exp(-z/20)  (E[U] = 0), minimised
at z* ~= -3.20 with E[Y] ~= -2.17.  Intervening on X alone is strictly worse
(its effect on Y is fully mediated by Z, and do(X) additionally cuts the
U -> X arm of the confounder without touching U -> Y).

Exploration sets: MIS = {X}, {Z}, {X,Z} (the CBO-2020 paper's exploration
set).  POMIS on the projected ADMG is {{Z}} — do(X) is dominated because
X's influence on Y is entirely mediated by Z, and do(X, Z) == do(Z) in
distribution — so the paper's optimum at {Z} is the POMIS arm.

This class follows the ``Tier1Graph`` / ``ConfoundedCluster`` pattern:
generic Monte-Carlo do-functions (``ccbo/generic_do.py``) for the legacy
raw-graph CBO path; when wrapped in a ``CoarsenedGraph`` the exploration set
and do-functions are derived at the C-DAG level instead.
"""

import numpy as np
from collections import OrderedDict

from . import graph
from ccbo.cbo.utils.utils import fit_single_GP_model
from ccbo.generic_do import make_do_function

from .ToyGraph_CostFunctions import define_costs


class ToyGraph(graph.GraphStructure):
    """GraphStructure for the CBO-2020 toy benchmark."""

    def __init__(self, observational_samples):
        self.X = np.asarray(observational_samples['X'])[:, np.newaxis]
        self.Z = np.asarray(observational_samples['Z'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

    # ------------------------------------------------------------------
    # SEM
    # ------------------------------------------------------------------
    def define_SEM(self):
        # Epsilon layout: 0: U   1: eX   2: eZ   3: eY
        def fU(epsilon, **kw):
            return float(epsilon[0])

        def fX(epsilon, U, **kw):
            return float(U + epsilon[1])

        def fZ(epsilon, X, **kw):
            return float(np.exp(-X) + epsilon[2])

        def fY(epsilon, Z, U, **kw):
            return float(np.cos(Z) - np.exp(-Z / 20.0) + U + epsilon[3])

        return OrderedDict([
            ('U', fU),
            ('X', fX),
            ('Z', fZ),
            ('Y', fY),
        ])

    # ------------------------------------------------------------------
    # Exploration / interventional metadata
    # ------------------------------------------------------------------
    def get_sets(self):
        # MIS as in the CBO-2020 paper.  POMIS = {{Z}} on the projected ADMG
        # (X -> Z -> Y, X <-> Y): X's effect on Y is fully mediated by Z, so
        # both {X} and {X, Z} are dominated by {Z}.
        MIS = [['X'], ['Z'], ['X', 'Z']]
        POMIS = [['Z']]
        manipulative_variables = ['X', 'Z']
        return MIS, POMIS, manipulative_variables

    def get_set_BO(self):
        return ['X', 'Z']

    def get_interventional_ranges(self):
        return OrderedDict([
            ('X', [-5.0, 5.0]),
            ('Z', [-5.0, 20.0]),
        ])

    # ------------------------------------------------------------------
    # GP fits
    # ------------------------------------------------------------------
    def fit_all_models(self):
        return self._fit_models_from(self.X, self.Z, self.Y)

    def refit_models(self, observational_samples):
        X = np.asarray(observational_samples['X'])[:, np.newaxis]
        Z = np.asarray(observational_samples['Z'])[:, np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        return self._fit_models_from(X, Z, Y)

    @staticmethod
    def _fit_models_from(X, Z, Y):
        functions = {}
        spec = [
            ('gp_Z_X', np.hstack((X,)), 'Z'),     # mediator model Z | X
            ('gp_Y_X', np.hstack((X,)), 'Y'),
            ('gp_Y_Z', np.hstack((Z,)), 'Y'),
            ('gp_Y_X_Z', np.hstack((X, Z)), 'Y'),
        ]
        params = [1.0, 1.0, 1.0, False]
        for name, inp, out_name in spec:
            out = Z if out_name == 'Z' else Y
            functions[name] = fit_single_GP_model(inp, out, params)
        return functions

    # ------------------------------------------------------------------
    # Costs & do-functions
    # ------------------------------------------------------------------
    def get_cost_structure(self, type_cost):
        return define_costs(type_cost)

    def get_all_do(self):
        """Oracle do-functions for the legacy (raw-graph) CBO path.

        Built by Monte Carlo through the true SEM (``generic_do``).  QCBO
        does not use these — when this graph is wrapped in a CoarsenedGraph
        the do-functions are derived at the C-DAG level by
        ``make_cdag_do_function`` — but we provide them so the
        GraphStructure interface is complete.
        """
        sem = self.define_SEM
        return {
            'compute_do_X': make_do_function(sem, ['X'], num_mc_samples=2000),
            'compute_do_Z': make_do_function(sem, ['Z'], num_mc_samples=2000),
            'compute_do_XZ': make_do_function(sem, ['X', 'Z'], num_mc_samples=2000),
        }
