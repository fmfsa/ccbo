"""Cost functions for the Tier1Graph benchmark.

Same layout as ConfoundedCluster_CostFunctions.py: one fixed-cost function per
manipulable variable (B, D, E), four cost types selectable via ``type_cost``.
The robustness (wrong_edge) experiments use ``type_cost=1`` (unit cost for every
intervention).
"""

import numpy as np
from collections import OrderedDict


def _unit(_value, **_kw): return 1.0
def _var(_value, **_kw): return float(np.sum(np.abs(_value))) + 1.0


# Type 1: every variable costs 1 (used by the robustness runs).
cost_B_fix_equal = _unit
cost_D_fix_equal = _unit
cost_E_fix_equal = _unit

# Type 2: per-variable fixed costs.
def cost_B_fix_different(_v, **_kw): return 1.0
def cost_D_fix_different(_v, **_kw): return 1.0
def cost_E_fix_different(_v, **_kw): return 2.0

# Type 3: fix_different + magnitude penalty.
def cost_B_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 1.0
def cost_D_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 1.0
def cost_E_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 2.0

# Type 4: fix_equal + magnitude penalty.
cost_B_fix_equal_variable = _var
cost_D_fix_equal_variable = _var
cost_E_fix_equal_variable = _var


def define_costs(type_cost):
    if type_cost == 1:
        return OrderedDict([
            ('B', cost_B_fix_equal),
            ('D', cost_D_fix_equal),
            ('E', cost_E_fix_equal),
        ])
    if type_cost == 2:
        return OrderedDict([
            ('B', cost_B_fix_different),
            ('D', cost_D_fix_different),
            ('E', cost_E_fix_different),
        ])
    if type_cost == 3:
        return OrderedDict([
            ('B', cost_B_fix_different_variable),
            ('D', cost_D_fix_different_variable),
            ('E', cost_E_fix_different_variable),
        ])
    if type_cost == 4:
        return OrderedDict([
            ('B', cost_B_fix_equal_variable),
            ('D', cost_D_fix_equal_variable),
            ('E', cost_E_fix_equal_variable),
        ])
    raise ValueError(f"Unknown type_cost: {type_cost}")
