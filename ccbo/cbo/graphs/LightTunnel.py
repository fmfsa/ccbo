"""
LightTunnel GraphStructure.

Causal model
------------
Manipulable: R, G, B, P1, P2  (each in [0, 1])
Target:      Y                (scaled vis_3 sensor reading from lt.Deterministic)
Hidden:      U_color          (latent confounder on R and G; creates
                               observational corr(R, G) > 0 without a
                               causal R -> G edge)

Observable DAG (used by adjustment / C-DAG identification):
    R -> Y     G -> Y     B -> Y     P1 -> Y     P2 -> Y
    R <-> G    (bidirected, from projecting out U_color)

This SEM doubles as the data-generating process for observations.pkl
(see ccbo/cbo/data/LightTunnel/generate_observations.py).
"""

import numpy as np
from collections import OrderedDict

from . import graph
from ccbo.cbo.utils.utils import fit_single_GP_model

from .LightTunnel_CostFunctions import define_costs


# Calibrated lt.Deterministic parameters (from the package tutorial). Kept
# inline here so this module has no run-time dependency on the experiments
# package, only on `causalchamber`.
_CALIBRATED_PARAMS = {
    "S": np.array(
        [[4.34165202, 2.8594438, 1.91565691],
         [0.89410569, 1.58679242, 2.68477163]]
    ),
    "d1": np.array(0.1), "d2": np.array(0.17), "d3": np.array(0.25),
    "Ts": np.array([0.42685688, 0.41231782, 0.46652526]),
    "Tp": np.array([0.2909342, 0.29624314, 0.30526271]),
    "Tc": np.array([0.00431032, 0.01447611, 0.02041075]),
    "Q":  np.array([0.12834946, 0.12870338, 0.13078523]),
    "C0": np.array(58.13307318503893),
    "A":  np.array(1.42261211),
    "a1": np.array(506.8049747171294),
    "a2": np.array(510.98343062774353),
}

# Diode/exposure/voltage knobs are held at the same defaults used by the
# tutorial. They're physical settings, not manipulable variables for BO.
_FIXED_SENSOR_INPUTS = {
    "diode_ir_1": 2, "diode_ir_2": 2, "diode_ir_3": 2,
    "diode_vis_1": 1, "diode_vis_2": 1, "diode_vis_3": 1,
    "t_ir_1": 3, "t_ir_2": 3, "t_ir_3": 3,
    "t_vis_1": 3, "t_vis_2": 3, "t_vis_3": 3,
    "v_c": 5, "v_angle_1": 5, "v_angle_2": 5,
}

# Y = vis_3 / _Y_SCALE.  vis_3 ranges ~[0, 1500]; rescaling keeps GP
# hyperparameters in a comfortable regime.
_Y_SCALE = 500.0

# Latent confounder strength on R, G.  Larger => stronger observational
# correlation between R and G, which is what makes the WrongRG misspec bite.
_SIGMA_U_COLOR = 0.15
_SIGMA_INPUT = 0.10
_SIGMA_Y = 0.02

# Pre-computed constants for the inlined fast vis_3 evaluation. Routing
# every SEM sample through pandas + lt.Deterministic.simulate_from_inputs
# adds ~1 ms of DataFrame overhead per call -- a single condition of the
# wrong_edge pipeline needs ~3e5 evals (generate_interventional_data),
# which would push wall time into hours. The math below is the exact
# vis_3 path from lt.Deterministic._simulate, specialised to our fixed
# diode_vis_3 = 1 and t_vis_3 = 3 defaults.
_S = _CALIBRATED_PARAMS["S"]
_d1 = float(_CALIBRATED_PARAMS["d1"])
_d3 = float(_CALIBRATED_PARAMS["d3"])
_TpTc = np.asarray(_CALIBRATED_PARAMS["Tp"] - _CALIBRATED_PARAMS["Tc"])
_Tc = np.asarray(_CALIBRATED_PARAMS["Tc"])
_GAIN_VIS3 = 2.0 ** (1 + 3)  # diode_vis_3=1, t_vis_3=3
_DIST_SCALE = (_d1 / _d3) ** 2


def _vis3_value(R, G, B, P1, P2):
    """Scaled vis_3 reading for scalar (R, G, B, P1, P2) in [0,1]^5.

    Mirrors lt.Deterministic._simulate's vis_3 path exactly.
    """
    R = float(np.clip(R, 0.0, 1.0)) * 255.0
    G = float(np.clip(G, 0.0, 1.0)) * 255.0
    B = float(np.clip(B, 0.0, 1.0)) * 255.0
    P1_deg = float(np.clip(P1, 0.0, 1.0)) * 180.0 - 90.0
    P2_deg = float(np.clip(P2, 0.0, 1.0)) * 180.0 - 90.0

    rgb = np.array([R, G, B])
    malus = np.cos(np.deg2rad(P1_deg - P2_deg)) ** 2
    crgb = _DIST_SCALE * rgb
    crgb = _TpTc * (malus * crgb) + _Tc * crgb
    srgb = _S @ crgb
    vis3 = _GAIN_VIS3 * srgb[1]
    return float(vis3) / _Y_SCALE


class LightTunnel(graph.GraphStructure):
    """
    GraphStructure wrapper around lt.Deterministic for CCBO experiments.
    """

    def __init__(self, observational_samples):
        self.R = np.asarray(observational_samples['R'])[:, np.newaxis]
        self.G = np.asarray(observational_samples['G'])[:, np.newaxis]
        self.B = np.asarray(observational_samples['B'])[:, np.newaxis]
        self.P1 = np.asarray(observational_samples['P1'])[:, np.newaxis]
        self.P2 = np.asarray(observational_samples['P2'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

    # ------------------------------------------------------------------
    # SEM
    # ------------------------------------------------------------------
    def define_SEM(self):
        # Epsilon layout (indices used below):
        #   0: U_color   1: eR   2: eG   3: eB   4: eP1   5: eP2   6: eY
        def fU(epsilon, **kw):
            return _SIGMA_U_COLOR * epsilon[0]

        def fR(epsilon, U_color, **kw):
            return float(np.clip(0.5 + U_color + _SIGMA_INPUT * epsilon[1], 0.0, 1.0))

        def fG(epsilon, U_color, **kw):
            return float(np.clip(0.5 + U_color + _SIGMA_INPUT * epsilon[2], 0.0, 1.0))

        def fB(epsilon, **kw):
            return float(np.clip(0.5 + _SIGMA_INPUT * epsilon[3], 0.0, 1.0))

        def fP1(epsilon, **kw):
            return float(np.clip(0.5 + _SIGMA_INPUT * epsilon[4], 0.0, 1.0))

        def fP2(epsilon, **kw):
            return float(np.clip(0.5 + _SIGMA_INPUT * epsilon[5], 0.0, 1.0))

        def fY(epsilon, R, G, B, P1, P2, **kw):
            return _vis3_value(R, G, B, P1, P2) + _SIGMA_Y * epsilon[6]

        return OrderedDict([
            ('U_color', fU),
            ('R', fR), ('G', fG), ('B', fB),
            ('P1', fP1), ('P2', fP2),
            ('Y', fY),
        ])

    # ------------------------------------------------------------------
    # Exploration / interventional metadata
    # ------------------------------------------------------------------
    def get_sets(self):
        MIS = [
            ['R'], ['G'], ['B'], ['P1'], ['P2'],
            ['P1', 'P2'],
            ['R', 'G', 'B'],
            ['R', 'G', 'B', 'P1', 'P2'],
        ]
        POMIS = MIS  # every entry has a direct path to Y; none structurally dominated
        manipulative_variables = ['R', 'G', 'B', 'P1', 'P2']
        return MIS, POMIS, manipulative_variables

    def get_set_BO(self):
        return ['R', 'G', 'B', 'P1', 'P2']

    def get_interventional_ranges(self):
        return OrderedDict([
            ('R', [0.0, 1.0]),
            ('G', [0.0, 1.0]),
            ('B', [0.0, 1.0]),
            ('P1', [0.0, 1.0]),
            ('P2', [0.0, 1.0]),
        ])

    # ------------------------------------------------------------------
    # GP fits
    # ------------------------------------------------------------------
    def fit_all_models(self):
        """
        Fit one GP per MIS entry, predicting Y from the intervened input
        variables. Used by the hand-coded do-functions in
        LightTunnel_DoFunctions.py. When the graph is wrapped in a
        CoarsenedGraph, these GPs are *not* used (CoarsenedGraph derives
        its own C-DAG-level do-functions via make_cdag_do_function); we
        still populate them so the legacy CBO path works if someone calls
        it.
        """
        return self._fit_models_from(
            self.R, self.G, self.B, self.P1, self.P2, self.Y
        )

    def refit_models(self, observational_samples):
        R = np.asarray(observational_samples['R'])[:, np.newaxis]
        G = np.asarray(observational_samples['G'])[:, np.newaxis]
        B = np.asarray(observational_samples['B'])[:, np.newaxis]
        P1 = np.asarray(observational_samples['P1'])[:, np.newaxis]
        P2 = np.asarray(observational_samples['P2'])[:, np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        return self._fit_models_from(R, G, B, P1, P2, Y)

    @staticmethod
    def _fit_models_from(R, G, B, P1, P2, Y):
        functions = {}
        # One GP per (input subset -> Y) used by the do-functions.
        spec = [
            ('gp_Y_R', np.hstack((R,))),
            ('gp_Y_G', np.hstack((G,))),
            ('gp_Y_B', np.hstack((B,))),
            ('gp_Y_P1', np.hstack((P1,))),
            ('gp_Y_P2', np.hstack((P2,))),
            ('gp_Y_P1_P2', np.hstack((P1, P2))),
            ('gp_Y_R_G_B', np.hstack((R, G, B))),
            ('gp_Y_R_G_B_P1_P2', np.hstack((R, G, B, P1, P2))),
        ]
        # ARD=False keeps each GP fit cheap; the inputs are all in [0,1] so a
        # single isotropic lengthscale is a sane prior.
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
        from .LightTunnel_DoFunctions import (
            compute_do_R, compute_do_G, compute_do_B,
            compute_do_P1, compute_do_P2,
            compute_do_P1P2, compute_do_RGB, compute_do_RGBP1P2,
        )
        return {
            'compute_do_R': compute_do_R,
            'compute_do_G': compute_do_G,
            'compute_do_B': compute_do_B,
            'compute_do_P1': compute_do_P1,
            'compute_do_P2': compute_do_P2,
            'compute_do_P1P2': compute_do_P1P2,
            'compute_do_RGB': compute_do_RGB,
            'compute_do_RGBP1P2': compute_do_RGBP1P2,
        }
