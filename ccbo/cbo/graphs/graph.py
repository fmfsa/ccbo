import abc

import numpy as np


class GraphStructure:
    __metaclass__ = abc.ABCMeta

    def _refresh_observational_attrs(self, observational_samples):
        """Write newly revealed observational columns back onto the graph.

        ``refit_models`` implementations read their frame into locals and fit
        from those.  Without this write-back the cached ``self.<col>`` arrays
        set in ``__init__`` keep the *initial* rows, so any later
        ``fit_all_models()`` silently reverts the prior to them -- which is
        exactly what happened across the HQCBO phase boundary.
        """
        for col in observational_samples.columns:
            setattr(self, col,
                    np.asarray(observational_samples[col])[:, np.newaxis])

    @abc.abstractmethod
    def define_SEM():
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def fit_all_models(self):

        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def refit_models(self, observational_samples):
        raise NotImplementedError("Subclass should implement this.")


    @abc.abstractmethod
    def get_all_do(self):
        raise NotImplementedError("Subclass should implement this.")

