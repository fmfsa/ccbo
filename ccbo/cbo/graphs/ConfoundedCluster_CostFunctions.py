"""Cost functions for the ConfoundedCluster benchmark.

Same layout as CompleteGraph_CostFunctions.py: one fixed-cost function per
manipulable variable, four cost types selectable via ``type_cost``.  The
robustness (wrong_edge) experiments use ``type_cost=1`` (unit cost for every
intervention), so all three variables share a flat per-call cost.
"""

import numpy as np
from collections import OrderedDict


def _unit(_value, **_kw): return 1.0
def _var(_value, **_kw): return float(np.sum(np.abs(_value))) + 1.0


# Type 1: every variable costs 1 (used by the robustness runs).
cost_A_fix_equal = _unit
cost_B_fix_equal = _unit
cost_C_fix_equal = _unit

# Type 2: per-variable fixed costs (placeholder; matches CompleteGraph's
# pattern of having different fixed costs per variable).
def cost_A_fix_different(_v, **_kw): return 1.0
def cost_B_fix_different(_v, **_kw): return 1.0
def cost_C_fix_different(_v, **_kw): return 2.0

# Type 3: fix_different + magnitude penalty.
def cost_A_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 1.0
def cost_B_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 1.0
def cost_C_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 2.0

# Type 4: fix_equal + magnitude penalty.
cost_A_fix_equal_variable = _var
cost_B_fix_equal_variable = _var
cost_C_fix_equal_variable = _var


def define_costs(type_cost):
    if type_cost == 1:
        return OrderedDict([
            ('A', cost_A_fix_equal),
            ('B', cost_B_fix_equal),
            ('C', cost_C_fix_equal),
        ])
    if type_cost == 2:
        return OrderedDict([
            ('A', cost_A_fix_different),
            ('B', cost_B_fix_different),
            ('C', cost_C_fix_different),
        ])
    if type_cost == 3:
        return OrderedDict([
            ('A', cost_A_fix_different_variable),
            ('B', cost_B_fix_different_variable),
            ('C', cost_C_fix_different_variable),
        ])
    if type_cost == 4:
        return OrderedDict([
            ('A', cost_A_fix_equal_variable),
            ('B', cost_B_fix_equal_variable),
            ('C', cost_C_fix_equal_variable),
        ])
    raise ValueError(f"Unknown type_cost: {type_cost}")
