# -*- coding: utf-8 -*-
"""Gaussian mixture Boltzman machine relaxations.

Helper classes for performing inference with Gaussian mixture Boltzmann machine
relaxations [1].

The original discrete-space Boltzmann machine is assumed to have weight
parameters `W` and bias parameters `b` and have a signed-binary state
`s[i] \in [-1, +1]` with the probability of a binary vector `s` being

    p(s) = exp(0.5 * s.dot(W).dot(s) + b.dot(s)) / Z

where `Z` is the normalisation constant.

Two alternative parameterisations for the relaxation distribution on continuous
vector state `x` are provided.

The unnormalised density of the `isotropic cosh` form is

    p(x) \propto exp(-0.5 * (x - b).T.dot(V).dot(x - b))) * cosh(x).prod()

where `V.dot(W + D) == I` and `D` is a diagonal matrix such that `(W + D)` is
(semi-)positive definite.

The unnormalised density of the `istropic covariance` form is

    p(x) \propto exp(-0.5 * x.dot(x)) * cosh(Q.dot(x) + b).prod()

where `Q.dot(Q.T) = W + D` with `D` as above.

References
----------
[1]: Zhang, Y., Ghahramani, Z., Storkey, A. J., & Sutton, C. A.
     Continuous relaxations for discrete Hamiltonian Monte Carlo.
     Advances in Neural Information Processing Systems 2012 (pp. 3194-3202).
"""

import numpy as np
import scipy.linalg as la
import itertools as it
import cvxpy as cvx
import bmtools.exact.moments as mom
import bmtools.exact.helpers as hlp


# Dimension of original binary state space above which to have user have to
# manually set `force` argument to `True` in functions which involve an
# exhaustive iteration over the 2**D binary state space.
FORCE_DIM = 10


def get_min_lambda_max_diagonal(W):
    """
    Compute maximum eigenvalue minimising additive diagonal term.

    Calculates additive diagonal term D which minimises maximum eigenvalue of
    W + D subject to the constrain W + D is semi-positive definite. This can
    be expressed as a semi-definite programme.
    """
    d_vct = cvx.Variable(W.shape[0])
    D_mtx = cvx.diag(d_vct)
    objc = cvx.Minimize(cvx.lambda_max(W + D_mtx))
    cnst = [W + D_mtx >> 0]
    prob = cvx.Problem(objc, cnst)
    prob.solve()
    return np.array(D_mtx.value)


class BaseGMRelaxation(object):
    """Abstract Gaussian mixture Boltzmann machine relaxation base class."""

    def __init__(self, W, b, optimise_eigvals=True, epsilon=0.,
                 eigval_thr=1e-6, precomputed_D=None):
        self.W = W
        self.b = b
        self.n_unit = W.shape[0]
        if precomputed_D is not None:
            self.D = precomputed_D
            e, R = la.eigh(W + self.D)
            assert np.all(e > -eigval_thr), 'Provided D does not make W PSD'
        elif optimise_eigvals:
            self.D = (get_min_lambda_max_diagonal(W) +
                      np.eye(self.n_unit) * epsilon)
            e, R = la.eigh(W + self.D)
        else:
            e, R = la.eigh(W)
            self.D = np.eye(n_unit) * (epsilon - e.min())
            e += (epsilon - e.min())
        idx_nz = e > eigval_thr
        self.n_dim_r = int(idx_nz.sum())
        self.R_r = R[:, idx_nz]
        self.e_r = e[idx_nz]

    def moments_s(self, force=False, num_threads=4):
        if self.n_unit > FORCE_DIM and not force:
            print('Size of state space is large ({0}) '
                  'set force=True to proceed anyway'.format(2**self.n_unit))
        norm_const_s, expc_s, expc_ss = mom.calculate_moments_parallel(
            self.W, self.b, force=force, num_threads=num_threads)
        return norm_const_s, np.array(expc_s), np.array(expc_ss)

    def neg_log_dens_x(self, x):
        raise NotImplementedError()

    def grad_neg_log_dens_x(self, x):
        raise NotImplementedError()

    def sample_x_gvn_s(self, s, n_sample, prng):
        raise NotImplementedError()

    def independent_samples(self, n_sample, force=False,
                            prng=np.random.RandomState(), num_threads=4,
                            return_probs_and_state_counts=True):
        if self.n_unit > FORCE_DIM and not force:
            print('Size of state space is large ({0}) '
                  'set force=True to proceed anyway'.format(2**self.n_unit))
        probs, norm_const = mom.calculate_probs_parallel(
            self.W, self.b, force=force, num_threads=num_threads)
        binary_state_counts = prng.multinomial(n_sample, probs)
        xs = np.empty((n_sample, self.n_dim))
        cum_count = 0
        s = np.empty(self.n_unit, dtype=np.int8)
        for i in np.nonzero(binary_state_counts)[0]:
            hlp.index_to_state(i, s)
            count = binary_state_counts[i]
            xs[cum_count:cum_count+count] = self.sample_x_gvn_s(s, count, prng)
            cum_count += count
        return xs, binary_state_counts, probs, norm_const


class IsotropicCovarianceGMRelaxation(BaseGMRelaxation):
    """Isotropic covariance parameterisation of Gaussian mixture relaxation.

    The unnormalised density is of the form

        p(x) \propto exp(-0.5 * x.dot(x)) * cosh(Q.dot(x) + b).prod()

    where `Q.dot(Q.T) = W + D` with `D` a diagonal matrix such that `(W + D)`
    is (semi-)positive definite.
    """

    def __init__(self, W, b, optimise_eigvals=True, epsilon=0.,
                 eigval_thr=1e-6, precomputed_D=None):
        super(IsotropicCovarianceGMRelaxation, self).__init__(
            W, b, optimise_eigvals, epsilon, eigval_thr, precomputed_D)
        self.n_dim = self.n_dim_r
        self.Q = self.R_r * self.e_r**0.5

    def neg_log_dens_x(self, x):
        return (0.5 * (x**2).sum(-1) -
                np.log(np.cosh(x.dot(self.Q.T) + self.b)).sum(-1))

    def grad_neg_log_dens_x(self, x):
        return x - np.tanh(x.dot(self.Q.T) + self.b).dot(self.Q)

    def sample_x_gvn_s(self, s, n_sample, prng):
        return (self.Q.T.dot(s)[None, :] +
                prng.normal(size=(n_sample, self.n_dim_r)))

    def moments_x(self, force=False, num_threads=4):
        norm_const_s, expc_s, expc_ss = self.moments_s(force, num_threads)
        log_norm_const_x = (
            0.5 * self.n_dim_r * np.log(2 * np.pi) - self.n_unit * np.log(2) +
            0.5 * self.D.diagonal().sum() + np.log(norm_const_s)
        )
        expc_x = self.Q.T.dot(expc_s)
        covar_x = (
            self.Q.T.dot(expc_ss - np.outer(expc_s, expc_s)).dot(self.Q) +
            np.eye(self.n_dim_r)
        )
        return log_norm_const_x, expc_x, covar_x


class IsotropicCoshGMRelaxation(BaseGMRelaxation):
    """Isotropic cosh parameterisation of Gaussian mixture relaxation.

    The unnormalised density of the relaxation is of the form

        p(x) \propto exp(-0.5 * (x - b).T.dot(V).dot(x - b))) * cosh(x).prod()

    where `V.dot(W + D) == I` and `D` is a diagonal matrix such that `(W + D)`
    is (semi-)positive definite.
    """

    def __init__(self, W, b, optimise_eigvals=True, epsilon=0.,
                 eigval_thr=1e-6, precomputed_D=None):
        super(IsotropicCoshGMRelaxation, self).__init__(
            W, b, optimise_eigvals, epsilon, eigval_thr, precomputed_D)
        self.n_dim = self.n_unit
        self.V_inv = self.W + self.D
        self.V_inv_sqrt = self.R_r * self.e_r**0.5
        self.V = (self.R_r / self.e_r).dot(self.R_r.T)
        self.V_sqrt = self.R_r / self.e_r**0.5

    def neg_log_dens_x(self, x):
        x_m_b = x - self.b
        return (0.5 * x_m_b.dot(self.V) * x_m_b - np.log(np.cosh(x))).sum(-1)

    def grad_neg_log_dens_x(self, x):
        return (x - self.b).dot(V) - np.tanh(x)

    def sample_x_gvn_s(self, s, n_sample, prng):
        n = prng.normal(size=(n_sample, self.n_dim_r))
        return (self.V_inv.dot(s) + self.b)[None, :] + n.dot(self.V_inv_sqrt.T)

    def moments_x(self, force=False, num_threads=4):
        norm_const_s, expc_s, expc_ss = self.moments_s(force, num_threads)
        log_norm_const_x = (
            0.5 * self.n_dim_r * np.log(2 * np.pi) - self.n_unit * np.log(2) +
            0.5 * self.D.diagonal().sum() + np.log(norm_const_s) +
            0.5 * self.e_r.sum()
        )
        expc_x = self.b + self.V.dot(expc_s)
        covar_x = (
            self.V.dot(expc_ss - np.outer(expc_s, expc_s)).dot(self.V) +
            self.V
        )
        return log_norm_const_x, expc_x, covar_x
