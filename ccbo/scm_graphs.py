"""Fine-graph objects for the benchmark SCMs used by the static experiments."""

from ccbo.cbo.graphs import FrontDoor, MediatedChain, ParallelParent, ToyGraph


def get_original_graph(experiment, observational_samples):
    """Instantiate the vendored CBO graph object for a benchmark SCM.

    MinimalBench variants (e.g. ``ParallelParent_NoX1Y``) share the true SCM;
    the misspecification lives only in the assumed-graph metadata.
    """
    if experiment == 'ToyGraph':
        return ToyGraph(observational_samples)
    if experiment.startswith('ParallelParent'):
        return ParallelParent(observational_samples)
    if experiment.startswith('FrontDoor'):
        return FrontDoor(observational_samples)
    if experiment.startswith('MediatedChain'):
        return MediatedChain(observational_samples)
    raise ValueError(f"Unknown experiment: {experiment}")
