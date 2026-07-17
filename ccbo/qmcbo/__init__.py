"""QMCBO: Quotient Model-based Causal Bayesian Optimization (v1 prototype).

Runs the vendored MCBO (Sussex et al.; third_party/.../baselines/mcbo) with a
quotient-transformed model view: cluster-level mechanisms composed along the
C-DAG instead of node-level mechanisms along the full DAG. See quotient.py.
"""

from ccbo.qmcbo.quotient import (  # noqa: F401
    quotient_env_profile,
    quotient_parent_nodes,
    lift_targets,
    perturb_parent_nodes,
    ensure_mcbo_on_path,
)
