"""Cost functions for the LightTunnel benchmark.

Same layout as CompleteGraph_CostFunctions.py: one fixed-cost function per
manipulable variable, four cost types selectable via ``type_cost``.
For wrong_edge experiments we use ``type_cost=1`` (unit cost for every
intervention), so all five variables share a flat per-call cost.
"""

import numpy as np
from collections import OrderedDict


def _unit(_value, **_kw): return 1.0
def _var(_value, **_kw): return float(np.sum(np.abs(_value))) + 1.0


# Type 1: every variable costs 1 (used by wrong_edge runs).
cost_R_fix_equal = _unit
cost_G_fix_equal = _unit
cost_B_fix_equal = _unit
cost_P1_fix_equal = _unit
cost_P2_fix_equal = _unit

# Type 2: per-variable fixed costs (placeholder; matches CompleteGraph's
# pattern of having different fixed costs per variable).
def cost_R_fix_different(_v, **_kw): return 1.0
def cost_G_fix_different(_v, **_kw): return 1.0
def cost_B_fix_different(_v, **_kw): return 1.0
def cost_P1_fix_different(_v, **_kw): return 5.0
def cost_P2_fix_different(_v, **_kw): return 5.0

# Type 3: fix_different + magnitude penalty.
def cost_R_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 1.0
def cost_G_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 1.0
def cost_B_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 1.0
def cost_P1_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 5.0
def cost_P2_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 5.0

# Type 4: fix_equal + magnitude penalty.
cost_R_fix_equal_variable = _var
cost_G_fix_equal_variable = _var
cost_B_fix_equal_variable = _var
cost_P1_fix_equal_variable = _var
cost_P2_fix_equal_variable = _var


def define_costs(type_cost):
    if type_cost == 1:
        return OrderedDict([
            ('R', cost_R_fix_equal), ('G', cost_G_fix_equal),
            ('B', cost_B_fix_equal),
            ('P1', cost_P1_fix_equal), ('P2', cost_P2_fix_equal),
        ])
    if type_cost == 2:
        return OrderedDict([
            ('R', cost_R_fix_different), ('G', cost_G_fix_different),
            ('B', cost_B_fix_different),
            ('P1', cost_P1_fix_different), ('P2', cost_P2_fix_different),
        ])
    if type_cost == 3:
        return OrderedDict([
            ('R', cost_R_fix_different_variable), ('G', cost_G_fix_different_variable),
            ('B', cost_B_fix_different_variable),
            ('P1', cost_P1_fix_different_variable), ('P2', cost_P2_fix_different_variable),
        ])
    if type_cost == 4:
        return OrderedDict([
            ('R', cost_R_fix_equal_variable), ('G', cost_G_fix_equal_variable),
            ('B', cost_B_fix_equal_variable),
            ('P1', cost_P1_fix_equal_variable), ('P2', cost_P2_fix_equal_variable),
        ])
    raise ValueError(f"Unknown type_cost: {type_cost}")
