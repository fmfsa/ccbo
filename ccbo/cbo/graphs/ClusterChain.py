"""ClusterChain GraphStructure (multi-cluster controlled benchmark).

Six manipulable variables in three confounded or chained pairs. All
structure, coefficients, ranges and closed-form values live in
``ccbo/minibench.py`` (``CC_*``, ``cc_do``, ``cc_mu_star``).

Causal model
------------
Manipulable: A1, A2, D1, D2 in [-3, 3];  B1, B2 in [-2, 2] (policy box)
Target:      Y
Latents:     UA (A1 <-> A2), UD (D1 <-> D2)

    A1 -> B1 -> Y,  A2 -> Y,  B2 -> Y,  D1 -> Y,  D2 -> Y
    Y = (A2+2)^2 + (B1-4)^2 + (B2-1)^2 + (D1-1)^2 + (D2+1)^2 + noise

The fine optimum do(A1=2, A2=-2, B2=1, D1=1, D2=-1) leaves B1 to respond
naturally around 4 (y* = 0.04). Clamping B1 inside its box costs at least 4,
so partitions that force B1 to be set together with B2 or A pay a price.

The vendored CBO loop only needs ranges, costs and the SEM (for its unused
default target); C-DAG priors are built by ``CoarsenedGraph`` from
observational data, so ``fit_all_models`` returns no fine conditionals.
"""

import numpy as np
from collections import OrderedDict

from . import graph
from ccbo import minibench as mb


def _unit(_value, **_kw):
    return 1.0


class ClusterChain(graph.GraphStructure):
    """GraphStructure for the ClusterChain SCM."""

    def __init__(self, observational_samples):
        for col in mb.CC_NODES:
            setattr(self, col, np.asarray(observational_samples[col])[:, np.newaxis])

    def define_SEM(self):
        # Epsilon layout: 0 UA, 1 UD, 2 eA1, 3 eA2, 4 eB1, 5 eB2, 6 eD1, 7 eD2, 8 eY.
        # Latents are folded into their children so the SEM keeps one entry per
        # observed variable (the vendored interface).
        def fA1(epsilon, **kw):
            return float(mb.CC_SIGMA_U * epsilon[0] + mb.CC_SIGMA_IN * epsilon[2])

        def fA2(epsilon, **kw):
            return float(mb.CC_SIGMA_U * epsilon[0] + mb.CC_SIGMA_IN * epsilon[3])

        def fB1(epsilon, A1, **kw):
            return float(mb.CC_BETA * A1 + mb.CC_SIGMA_B1 * epsilon[4])

        def fB2(epsilon, **kw):
            return float(mb.CC_SIGMA_B2 * epsilon[5])

        def fD1(epsilon, **kw):
            return float(mb.CC_SIGMA_U * epsilon[1] + mb.CC_SIGMA_IN * epsilon[6])

        def fD2(epsilon, **kw):
            return float(mb.CC_SIGMA_U * epsilon[1] + mb.CC_SIGMA_IN * epsilon[7])

        def fY(epsilon, A2, B1, B2, D1, D2, **kw):
            c = mb.CC_CENTRES
            return float((A2 - c["A2"]) ** 2 + (B1 - c["B1"]) ** 2 + (B2 - c["B2"]) ** 2
                         + (D1 - c["D1"]) ** 2 + (D2 - c["D2"]) ** 2
                         + mb.SIGMA_Y * epsilon[8])

        return OrderedDict([('A1', fA1), ('A2', fA2), ('B1', fB1), ('B2', fB2),
                            ('D1', fD1), ('D2', fD2), ('Y', fY)])

    def get_sets(self):
        # Exploration sets are derived by CoarsenedGraph from the C-DAG.
        raise NotImplementedError("use CoarsenedGraph.get_sets()")

    def get_set_BO(self):
        return list(mb.CC_MANIPULATIVE)

    def get_interventional_ranges(self):
        return OrderedDict((v, list(mb.domain(mb.CC_NAME, v))) for v in mb.CC_MANIPULATIVE)

    def fit_all_models(self):
        return {}

    def refit_models(self, observational_samples):
        self._refresh_observational_attrs(observational_samples)
        return {}

    def get_cost_structure(self, type_cost):
        if type_cost != 1:
            raise ValueError(f"Unknown type_cost: {type_cost}")
        return OrderedDict((v, _unit) for v in mb.CC_MANIPULATIVE)

    def get_all_do(self):
        raise NotImplementedError("C-DAG priors are built by CoarsenedGraph")
