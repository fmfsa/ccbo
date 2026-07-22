"""QDCBO: quotient Dynamic Causal Bayesian Optimisation.

Quotient transformation of the vendored DCBO stack
(third_party/DCBO, the authors' stack): the model's temporal graph,
arc mechanisms and SEM-hat all live on the cluster-level quotient of the
assumed DAG, while the optimisation runs on whole-cluster arms in member
coordinates. See quotient_dbn.py (structure), qdcbo_base.py (method) and
runner.py (CLI).
"""

from ccbo.qdcbo.quotient_dbn import (QuotientDBNSpec, build_quotient_dbn,
                                     ensure_dcbo_on_path,
                                     perturb_temporal_edges)

__all__ = [
    "QuotientDBNSpec",
    "build_quotient_dbn",
    "ensure_dcbo_on_path",
    "perturb_temporal_edges",
    "QDCBO",
    "fit_quotient_arcs",
]


def __getattr__(name):
    # Lazy: qdcbo_base imports GPy/emukit at module load; keep the package
    # importable for structure-only uses (tests of the quotient graph).
    if name in ("QDCBO", "fit_quotient_arcs"):
        from ccbo.qdcbo import qdcbo_base
        return getattr(qdcbo_base, name)
    raise AttributeError(name)
