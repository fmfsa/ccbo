"""Expected improvement per unit cost for the arm-based CBO loop.

Engine v3 (2026-09): the maximisation branch is a proper EI
(``u = (mu - best)/sigma``, gradient ``phi(u) dsigma + Phi(u) dmu``) instead
of the negated minimisation EI, and ``sigma = 0`` is guarded.
"""

import numpy as np
import scipy.stats

from typing import Tuple, Union
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition


class CausalExpectedImprovement(Acquisition):
    def __init__(self, current_global_min, task, model: Union[IModel, IDifferentiable], jitter: float = float(0)) -> None:
        """
        Expected improvement over the current best observed value.

        :param current_global_min: incumbent value (best observed so far).
        :param task: 'min' or 'max'.
        :param model: model used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        self.model = model
        self.jitter = jitter
        self.current_global_min = current_global_min
        self.task = task

    def _u_pdf_cdf(self, mean, standard_deviation):
        if self.task == 'min':
            return get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        return get_standard_normal_pdf_cdf(mean, self.current_global_min, standard_deviation)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Computes the Expected Improvement at ``x``."""
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(np.maximum(variance, 0.0))
        mean = mean + self.jitter
        u, pdf, cdf = self._u_pdf_cdf(mean, standard_deviation)
        improvement = standard_deviation * (u * cdf + pdf)
        return np.where(standard_deviation > 0.0, improvement, 0.0)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """Computes the Expected Improvement and its derivative."""
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(np.maximum(variance, 0.0))
        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        safe_sd = np.where(standard_deviation > 0.0, standard_deviation, np.inf)
        dstandard_deviation_dx = dvariance_dx / (2 * safe_sd)

        mean = mean + self.jitter
        u, pdf, cdf = self._u_pdf_cdf(mean, standard_deviation)
        improvement = standard_deviation * (u * cdf + pdf)
        if self.task == 'min':
            dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx
        else:
            dimprovement_dx = dstandard_deviation_dx * pdf + cdf * dmean_dx
        zero = standard_deviation <= 0.0
        improvement = np.where(zero, 0.0, improvement)
        dimprovement_dx = np.where(zero, 0.0, dimprovement_dx)
        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)


def get_standard_normal_pdf_cdf(x: np.array, mean: np.array, standard_deviation: np.array) \
        -> Tuple[np.array, np.array, np.array]:
    """
    Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma

    :param x: Non-standardized input
    :param mean: Mean to normalize x with
    :param standard_deviation: Standard deviation to normalize x with
    :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        u = (x - mean) / standard_deviation
    u = np.where(np.isfinite(u), u, 0.0)
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf
