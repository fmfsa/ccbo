"""
Partition manipulation utilities for RCCBO.

Provides functions to find the coarsest valid partition, compare partitions,
and compute diffs between old and new partitions after refinement.

Imports from ccbo.coarsening (read-only).
"""

from ccbo.coarsening import (
    enumerate_valid_coarsenings_manip,
    get_dag_edges_from_sem,
)


def coarsest_valid_partition(graph_name):
    """
    Return the coarsest (fewest clusters) manipulable-only partition for a
    graph.  Under Lee-2019, the coarsest partition puts every manipulable
    variable into a single cluster ``M`` alongside ``{Y}`` and the atomic
    non-manipulable vertices.

    Returns
    -------
    list of frozenset
        The coarsest valid manipulable-only partition.  Non-manipulable
        observables are *not* included explicitly — they live as singleton
        atoms in the coarsened ADMG automatically.
    """
    coarsenings = enumerate_valid_coarsenings_manip(graph_name)
    if not coarsenings:
        raise ValueError(f"No valid coarsenings found for {graph_name}")
    # Sorted finest-first, so coarsest is last.
    return coarsenings[-1]['partition']


def partitions_equal(pi1, pi2):
    """
    Check if two partitions are the same (set-of-frozensets equality).

    Parameters
    ----------
    pi1, pi2 : list of frozenset

    Returns
    -------
    bool
    """
    return set(pi1) == set(pi2)


def partition_diff(pi_old, pi_new):
    """
    Compute the difference between an old and new (finer) partition.

    Returns which clusters were split and into what sub-clusters.

    Parameters
    ----------
    pi_old : list of frozenset
        The old (coarser) partition.
    pi_new : list of frozenset
        The new (finer) partition.

    Returns
    -------
    dict
        Keys: frozenset (old cluster that was split)
        Values: list of frozenset (the sub-clusters it was split into)
        Clusters that remain unchanged are not included.
    """
    old_set = set(pi_old)
    new_set = set(pi_new)

    # Unchanged clusters appear in both
    unchanged = old_set & new_set

    # Old clusters that were split (no longer in new partition)
    split_clusters = old_set - unchanged

    diff = {}
    for old_cluster in split_clusters:
        # Find which new clusters are subsets of this old cluster
        children = [nc for nc in new_set - unchanged if nc.issubset(old_cluster)]
        if children:
            diff[old_cluster] = children

    return diff


def is_refinement(pi_new, pi_old):
    """
    Check that pi_new is a refinement of pi_old.

    Every cluster in pi_new must be a subset of some cluster in pi_old.

    Parameters
    ----------
    pi_new : list of frozenset
        Candidate refined partition.
    pi_old : list of frozenset
        Original coarser partition.

    Returns
    -------
    bool
    """
    for new_cluster in pi_new:
        if not any(new_cluster.issubset(old_cluster) for old_cluster in pi_old):
            return False
    return True


def partition_description(partition):
    """Human-readable string for a partition."""
    parts = []
    for part in sorted(partition, key=lambda p: sorted(p)):
        parts.append('{' + ','.join(sorted(part)) + '}')
    return ' | '.join(parts)
