"""
Bridge between CCBO's interventional data and RePaRe's expected format.

RePaRe expects:
  data_dict = {
      "obs": (np.array of shape (n, d), ()),           # observational data
      "env_1": (np.array of shape (m, d), target_set),  # interventional env
      ...
  }
where each array has ALL d variables, and target_set is the set of
column indices that were intervened on (hard interventions).

CCBO stores interventional data per exploration set entry as
(data_x, data_y) where data_x has only the intervened variables.

The bridge handles this by storing full-variable samples collected
during the RCCBO loop.
"""

import numpy as np
import pandas as pd


class FullSampleStore:
    """
    Stores full-variable samples collected during RCCBO interventions.

    Each time CBO performs an intervention do(X_S = x_S) and observes Y,
    RCCBO also records the full-variable sample (all d variables) by
    re-running the SEM. This data is what RePaRe needs.

    Attributes
    ----------
    var_names : list of str
        Ordered variable names (defines column order).
    obs_data : np.ndarray
        Observational data, shape (n_obs, d).
    envs : dict
        {env_key: {'data': np.ndarray (n, d), 'targets': set of int}}
        Each env_key identifies an intervention environment.
    """

    def __init__(self, var_names, observational_samples):
        """
        Parameters
        ----------
        var_names : list of str
            Ordered variable names for column mapping.
        observational_samples : pd.DataFrame
            Initial observational data.
        """
        self.var_names = list(var_names)
        self.var_to_idx = {v: i for i, v in enumerate(self.var_names)}

        # Store observational data as numpy array in canonical column order
        self.obs_data = observational_samples[self.var_names].values.copy()
        self.envs = {}

    def record_intervention(self, intervention_vars, full_samples):
        """
        Record full-variable samples from an intervention.

        Parameters
        ----------
        intervention_vars : list of str
            Variable names that were intervened on.
        full_samples : pd.DataFrame or dict
            If DataFrame: multiple samples (rows) with variable columns.
            If dict: {var_name: value} for a single sample (legacy).
            Only variables in self.var_names are recorded (hidden vars filtered).
        """
        # Pool all interventions on the same variable set into one
        # environment.  This gives RePaRe fewer, larger environments
        # with more statistical power for KS / distance correlation
        # tests.  Each environment still records which variables were
        # targeted via the 'targets' set.
        env_key = tuple(sorted(intervention_vars))

        # Convert to array in canonical column order
        if isinstance(full_samples, pd.DataFrame):
            # Multiple samples — extract columns in canonical order
            available = [v for v in self.var_names if v in full_samples.columns]
            if len(available) != len(self.var_names):
                return  # missing tracked variables
            rows = full_samples[self.var_names].values
        else:
            # Legacy: single dict {var_name: value}
            row = np.array([full_samples[v] for v in self.var_names
                            if v in full_samples]).reshape(1, -1)
            if row.shape[1] != len(self.var_names):
                return
            rows = row

        # Target indices (column indices of intervened variables)
        targets = {self.var_to_idx[v] for v in intervention_vars}

        if env_key not in self.envs:
            self.envs[env_key] = {'data': rows, 'targets': targets}
        else:
            self.envs[env_key]['data'] = np.vstack([
                self.envs[env_key]['data'], rows
            ])

    def to_repare_format(self):
        """
        Convert stored data to RePaRe's expected format.

        Returns
        -------
        dict
            {"obs": (obs_array, ()), "env_0": (data_array, target_set), ...}
        """
        data_dict = {"obs": (self.obs_data, ())}

        for i, (env_key, env_data) in enumerate(self.envs.items()):
            data_dict[f"env_{i}"] = (env_data['data'], env_data['targets'])

        return data_dict

    def num_interventional_samples(self):
        """Total number of interventional samples collected."""
        return sum(env['data'].shape[0] for env in self.envs.values())


def run_repare(full_sample_store, alpha=0.05, beta=0.05, seed=42,
               assume='gaussian'):
    """
    Run RePaRe on the collected interventional data.

    Parameters
    ----------
    full_sample_store : FullSampleStore
        Contains observational + interventional full-variable samples.
    alpha : float
        Significance level for the refinement test (cluster splitting).
    beta : float
        Significance level for the adjacency test (edge determination).
    seed : int
        Random seed for RePaRe's RNG.
    assume : str or None
        Distribution assumption for the refinement test.
        'gaussian' uses a Gaussian LRT (faster, avoids KS test issues).
        None uses nonparametric KS test.

    Returns
    -------
    partition : list of frozenset
        The refined partition as frozensets of observable variable names.
    repare_model : PartitionDagModelIvn
        The fitted RePaRe model (for inspection).
    """
    from ccbo.repare_lib.repare import PartitionDagModelIvn

    data_dict = full_sample_store.to_repare_format()
    var_names = full_sample_store.var_names

    model = PartitionDagModelIvn(rng=np.random.default_rng(seed))
    model.fit(data_dict, alpha=alpha, beta=beta, assume=assume)

    # Convert RePaRe's column-index partition back to variable names
    partition = []
    for node in model.dag.nodes():
        cluster = frozenset(var_names[idx] for idx in node)
        partition.append(cluster)

    # Ensure Y is a singleton (RePaRe may have merged it)
    partition = _ensure_target_singleton(partition, 'Y')

    return partition, model


def _ensure_target_singleton(partition, target='Y'):
    """
    Ensure the target variable is a singleton cluster.

    If RePaRe merged Y with other variables, split it out.
    """
    new_partition = []
    for cluster in partition:
        if target in cluster and len(cluster) > 1:
            # Split: {Y} as singleton, rest as another cluster
            new_partition.append(frozenset([target]))
            new_partition.append(cluster - {target})
        else:
            new_partition.append(cluster)
    return new_partition


def sample_full_variables(sem_fn, intervention_dict, num_samples=100, seed=None):
    """
    Run the mutilated SEM and return individual samples for all variables.

    This is used to record full-variable samples when CBO performs
    an intervention. Returns ALL samples (not means) so that RePaRe
    can detect distributional differences via KS tests.

    Parameters
    ----------
    sem_fn : callable
        Returns the SEM OrderedDict.
    intervention_dict : dict
        {var_name: value} for intervened variables.
    num_samples : int
        Number of MC samples to return.
    seed : int, optional

    Returns
    -------
    pd.DataFrame
        DataFrame with num_samples rows and one column per variable.
    """
    from ccbo.cbo.utils.graph_functions import intervene_dict, sample_from_model

    model = sem_fn()
    mutilated = intervene_dict(model, **intervention_dict)

    if seed is not None:
        np.random.seed(seed)

    samples = [sample_from_model(mutilated) for _ in range(num_samples)]
    return pd.DataFrame(samples)
