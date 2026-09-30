"""Cost functions for the ToyGraph benchmark (CBO-2020 toy).

One cost function per manipulable
variable (X, Z), four cost types selectable via ``type_cost``.  The family
comparison runs use ``type_cost=1`` (unit cost for every intervention).  The
``fix_different`` types mirror the CBO-2020 repository's ToyGraph costs
(X: 1, Z: 10).
"""

import numpy as np
from collections import OrderedDict


def _unit(_value, **_kw): return 1.0
def _var(_value, **_kw): return float(np.sum(np.abs(_value))) + 1.0


# Type 1: every variable costs 1 (used by the family-comparison runs).
cost_X_fix_equal = _unit
cost_Z_fix_equal = _unit

# Type 2: per-variable fixed costs (CBO-2020 ToyGraph convention).
def cost_X_fix_different(_v, **_kw): return 1.0
def cost_Z_fix_different(_v, **_kw): return 10.0

# Type 3: fix_different + magnitude penalty.
def cost_X_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 1.0
def cost_Z_fix_different_variable(v, **_kw): return float(np.sum(np.abs(v))) + 10.0

# Type 4: fix_equal + magnitude penalty.
cost_X_fix_equal_variable = _var
cost_Z_fix_equal_variable = _var


def define_costs(type_cost):
    if type_cost == 1:
        return OrderedDict([
            ('X', cost_X_fix_equal),
            ('Z', cost_Z_fix_equal),
        ])
    if type_cost == 2:
        return OrderedDict([
            ('X', cost_X_fix_different),
            ('Z', cost_Z_fix_different),
        ])
    if type_cost == 3:
        return OrderedDict([
            ('X', cost_X_fix_different_variable),
            ('Z', cost_Z_fix_different_variable),
        ])
    if type_cost == 4:
        return OrderedDict([
            ('X', cost_X_fix_equal_variable),
            ('Z', cost_Z_fix_equal_variable),
        ])
    raise ValueError(f"Unknown type_cost: {type_cost}")
