"""Cost functions for the FrontDoor SCM (MinimalBench).

Same layout as ConfoundedCluster_CostFunctions.py. The minimal suite uses
``type_cost=1`` (unit cost per intervention).
"""

import numpy as np
from collections import OrderedDict


def _unit(_value, **_kw): return 1.0
def _var(_value, **_kw): return float(np.sum(np.abs(_value))) + 1.0


def define_costs(type_cost):
    if type_cost == 1:
        return OrderedDict([('X1', _unit), ('M', _unit)])
    if type_cost == 4:
        return OrderedDict([('X1', _var), ('M', _var)])
    raise ValueError(f"Unknown type_cost: {type_cost}")
