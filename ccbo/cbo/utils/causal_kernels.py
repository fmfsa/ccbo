"""Causal RBF kernel of the CBO prior (Aglietti et al. 2020), with correct
input and hyperparameter gradients (engine v3, 2026-09).

    k(x, x') = sigma_f^2 exp(-||x - x'||^2 / (2 l^2)) + s(x) s(x'),
    s(x)     = sqrt(v(x)),   v(x) = variance_adjustment(x) >= 0,

where ``v`` is the do-calculus prior variance ``Var[Y | do(A=x)]``.  The
rank-one term is positive semidefinite.  Its input derivative needs
``dv/dx``; ``v`` is a black-box (GP-derived, cached) function, so the
derivative is taken by central differences with a relative step
``fd_step``.  The estimators behind ``v`` are deterministic and continuous
in ``x`` (fixed quadrature nodes / common random numbers), which is what
makes the finite difference well defined.

Overrides relative to :class:`GPy.kern.src.stationary.Stationary`:
``K``, ``Kdiag``, ``gradients_X``, ``gradients_X_diag`` (inherited version
returns zeros), ``update_gradients_full`` (inherited version divides the
*full* K, including the rank-one term, by ``sigma_f^2``), and the
second-order ``gradients_XX*`` (not implemented: fail loudly rather than
silently inherit a wrong value).
"""

import numpy as np

from GPy.kern.src.stationary import Stationary
from GPy.kern.src.psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from GPy.core import Param
from paramz.transformations import Logexp
from GPy.kern.src.grid_kerns import GridRBF


class CausalRBF(Stationary):
    """RBF kernel plus the rank-one causal-prior variance term."""

    _support_GPU = True

    def __init__(self, input_dim, variance_adjustment, variance=1.,
                 lengthscale=None, rescale_variance=1., ARD=False,
                 active_dims=None, name='rbf', useGPU=False, inv_l=False,
                 fd_step=1e-4):
        super(CausalRBF, self).__init__(input_dim, variance, lengthscale, ARD,
                                        active_dims, name, useGPU=useGPU)
        if self.useGPU:
            self.psicomp = PSICOMP_RBF_GPU()
        else:
            self.psicomp = PSICOMP_RBF()
        self.use_invLengthscale = inv_l
        if inv_l:
            self.unlink_parameter(self.lengthscale)
            self.inv_l = Param('inv_lengthscale', 1. / self.lengthscale ** 2, Logexp())
            self.link_parameter(self.inv_l)
        self.variance_adjustment = variance_adjustment
        # Kept for signature compatibility with the vendored callers; the
        # rank-one term is not rescaled.
        self.rescale_variance = float(rescale_variance)
        self.fd_step = float(fd_step)

    # ------------------------------------------------------------------ v(x)
    def _v(self, X):
        """Prior variance ``v(x) >= 0`` as an ``(N, 1)`` array."""
        X = np.asarray(X, dtype=float)
        v = np.asarray(self.variance_adjustment(X), dtype=float)
        if v.size == X.shape[0]:
            v = v.reshape(X.shape[0], 1)
        else:  # scalar / constant adjustment
            v = np.full((X.shape[0], 1), float(np.ravel(v)[0]))
        return np.maximum(v, 0.0)

    def _s(self, X):
        return np.sqrt(self._v(X))

    def _dv_dX(self, X):
        """Central-difference ``dv/dx`` with a relative step, ``(N, Q)``."""
        X = np.asarray(X, dtype=float)
        N, Q = X.shape
        grad = np.zeros((N, Q))
        for q in range(Q):
            h = self.fd_step * np.maximum(1.0, np.abs(X[:, q]))
            Xp = X.copy(); Xp[:, q] += h
            Xm = X.copy(); Xm[:, q] -= h
            grad[:, q] = (self._v(Xp)[:, 0] - self._v(Xm)[:, 0]) / (2.0 * h)
        return grad

    def _ds_dX(self, X):
        s = self._s(X)[:, 0]
        return self._dv_dX(X) / (2.0 * np.maximum(s, 1e-12))[:, None]

    # --------------------------------------------------------------- kernel
    def K(self, X, X2=None):
        """``K(X, X2) = sigma_f^2 exp(-r^2/2) + s(X) s(X2)^T``."""
        if X2 is None:
            X2 = X
        r = self._scaled_dist(X, X2)
        value = self.variance * np.exp(-0.5 * r ** 2)
        return value + self._s(X) @ self._s(X2).T

    def Kdiag(self, X):
        """``diag K = sigma_f^2 + v(X)`` as an ``(N,)`` array."""
        X = np.asarray(X, dtype=float)
        return np.ravel(self.variance + self._v(X)[:, 0])

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r ** 2)

    def dK_dr(self, r):
        return -r * self.K_of_r(r)

    def dK2_drdr(self, r):
        return (r ** 2 - 1) * self.K_of_r(r)

    def dK2_drdr_diag(self):
        return -self.variance  # the diagonal of r is always zero

    # ------------------------------------------------------ input gradients
    def gradients_X(self, dL_dK, X, X2=None):
        """``d/dX sum_ik dL_dK_ik K_ik``: RBF part from ``Stationary`` plus
        the rank-one term ``(W s2)_i ds_i/dX`` (or ``((W + W^T) s)_i ds_i/dX``
        when ``X2 is None``)."""
        grad = super(CausalRBF, self).gradients_X(dL_dK, X, X2)
        X = np.asarray(X, dtype=float)
        W = np.asarray(dL_dK, dtype=float)
        if X2 is None:
            s = self._s(X)[:, 0]
            W = np.broadcast_to(W, (X.shape[0], X.shape[0]))
            c = W @ s + W.T @ s
        else:
            s2 = self._s(X2)[:, 0]
            W = np.broadcast_to(W, (X.shape[0], np.asarray(X2).shape[0]))
            c = W @ s2
        return grad + self._ds_dX(X) * c[:, None]

    def gradients_X_diag(self, dL_dKdiag, X):
        """``d/dX sum_i dL_dKdiag_i Kdiag_i = dL_dKdiag_i dv_i/dX``."""
        w = np.ravel(np.asarray(dL_dKdiag, dtype=float))
        return w[:, None] * self._dv_dX(X)

    def gradients_XX(self, dL_dK, X, X2=None):
        raise NotImplementedError(
            "CausalRBF: second-order input gradients are not implemented")

    def gradients_XX_diag(self, dL_dKdiag, X):
        raise NotImplementedError(
            "CausalRBF: second-order input gradients are not implemented")

    # ------------------------------------------------ parameter gradients
    def update_gradients_full(self, dL_dK, X, X2=None):
        """Hyperparameter gradients of the RBF part only: the rank-one term
        does not depend on ``sigma_f^2`` or the lengthscale."""
        r = self._scaled_dist(X, X2)
        K_rbf = self.K_of_r(r)
        self.variance.gradient = np.sum(K_rbf * dL_dK) / self.variance
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        if self.ARD:
            tmp = dL_dr * self._inv_dist(X, X2)
            if X2 is None:
                X2 = X
            self.lengthscale.gradient = self._lengthscale_grads_pure(tmp, X, X2)
        else:
            self.lengthscale.gradient = -np.sum(dL_dr * r) / self.lengthscale
        if self.use_invLengthscale:
            self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.)

    def update_gradients_diag(self, dL_dKdiag, X):
        # d Kdiag / d sigma_f^2 = 1; the rank-one term carries no parameter.
        super(CausalRBF, self).update_gradients_diag(dL_dKdiag, X)
        if self.use_invLengthscale:
            self.inv_l.gradient = self.lengthscale.gradient * (self.lengthscale ** 3 / -2.)

    # -------------------------------------------------------- housekeeping
    def to_dict(self):
        input_dict = super(CausalRBF, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.RBF"
        input_dict["inv_l"] = self.use_invLengthscale
        if input_dict["inv_l"]:
            input_dict["lengthscale"] = np.sqrt(1 / float(self.inv_l))
        return input_dict

    def __getstate__(self):
        dc = super(CausalRBF, self).__getstate__()
        if self.useGPU:
            dc['psicomp'] = PSICOMP_RBF()
            dc['useGPU'] = False
        return dc

    def __setstate__(self, state):
        self.use_invLengthscale = False
        return super(CausalRBF, self).__setstate__(state)

    def spectrum(self, omega):
        assert self.input_dim == 1
        return (self.variance * np.sqrt(2 * np.pi) * self.lengthscale
                * np.exp(-self.lengthscale * 2 * omega ** 2 / 2))

    def parameters_changed(self):
        if self.use_invLengthscale:
            self.lengthscale[:] = 1. / np.sqrt(self.inv_l + 1e-200)
        super(CausalRBF, self).parameters_changed()

    def get_one_dimensional_kernel(self, dim):
        return GridRBF(input_dim=1, variance=self.variance.copy(),
                       originalDimensions=dim)

    # PSI statistics (sparse-GP interface; unused by CBO, kept for parity)
    def psi0(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[0]

    def psi1(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior)[1]

    def psi2(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=False)[2]

    def psi2n(self, Z, variational_posterior):
        return self.psicomp.psicomputations(self, Z, variational_posterior, return_psi2_n=True)[2]

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        dL_dvar, dL_dlengscale = self.psicomp.psiDerivativecomputations(
            self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[:2]
        self.variance.gradient = dL_dvar
        self.lengthscale.gradient = dL_dlengscale
        if self.use_invLengthscale:
            self.inv_l.gradient = dL_dlengscale * (self.lengthscale ** 3 / -2.)

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(
            self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[2]

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return self.psicomp.psiDerivativecomputations(
            self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)[3:]
