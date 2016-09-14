"""
=============================================
Gaussian mixture Boltzman machine relaxations
=============================================

Helper classes providing negative log densities (phi) and associated gradient
(dphi_dx) for various parametrisations of a Gaussian mixture relaxation of
a Boltzmann machine model. Methods allowing for computing independent samples
from the distributions are also provided - these involve an exhaustive
iteration over the 2^N binary states for a model with N units so can only
feasibly be used for small N.

References
----------
> Zhang, Y., Ghahramani, Z., Storkey, A. J., & Sutton, C. A.
> Continuous relaxations for discrete Hamiltonian Monte Carlo. 
> Advances in Neural Information Processing Systems 2012 (pp. 3194-3202).

"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

import numpy as np
import scipy.linalg as la
import itertools as it
import cvxpy as cvx


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
    return D_mtx.value

class ReducedSubspaceGMRelaxation(object):

    def __init__(self, W, b, optimise_eigvals=True, eigval_thr=1e-6):
        self.W = W
        self.b = b
        self.n_unit = W.shape[0]
        if optimise_eigvals:
            D = get_min_lambda_max_diagonal(W)
            self.e_p, self.R_p = la.eigh(W + D)
            idx_nz = self.e_p > eigval_thr
            self.n_dim_r = int(idx_nz.sum())
            self.R_r = self.R_p[:, idx_nz]
            self.e_r = self.e_p[idx_nz]
            self.Q = self.R_r * self.e_r**0.5
        else:
            self.e, self.R = la.eigh(W)
            min_e_mult = (np.abs(self.e - self.e[0]) < eigval_thr).sum()
            self.n_dim_r = self.n_unit - min_e_mult
            self.R_r = self.R[:, min_e_mult:]
            self.e_r = self.e[min_e_mult:] - self.e[0]
            self.Q = self.R_r * self.e_r**0.5

    def phi(self, x):
        return 0.5 * x.dot(x) - np.log(np.cosh(self.Q.dot(x) + self.b)).sum()

    def dphi_dx(self, x):
        return x - self.Q.T.dot(np.tanh(self.Q.dot(x) + self.b))

    def independent_samples(self, n_sample, force=False,
                            prng=np.random.RandomState()):
        if self.n_unit > 10 and not force:
            print('Size of state space is large ({0}) '
                  'set force=True to proceed anyway'.format(2**self.n_unit))
        ps = np.empty(2**self.n_unit)
        for i, s in enumerate(it.product(*[[-1, 1]] * self.n_unit)):
            s = np.array(s)
            ps[i] = np.exp(0.5 * s.dot(self.W).dot(s) + s.dot(self.b))
            if i % (2**(self.n_unit - 4)) == 0:
                print i
        ps /= ps.sum()
        binary_state_counts = prng.multinomial(n_sample, ps)
        xs = np.empty((n_sample, self.n_dim_r))
        j = 0
        for i, s in enumerate(it.product(*[[-1, 1]] * self.n_unit)):
            mean = self.Q.T.dot(s)
            count = binary_state_counts[i]
            xs[j:j+count] = (mean[None, :] +
                             prng.normal(size=(count, self.n_dim_r)))
            j += count
        return xs, binary_state_counts


class IsotropicCovarianceGMRelaxation(object):

    def __init__(self, W, b, epsilon=0.):
        self.W = W
        self.b = b
        self.e, self.R = la.eigh(W)
        self.e_p = self.e - self.e.min() + epsilon
        self.n_unit = W.shape[0]
        self.Q = self.R * self.e_p**0.5

    def phi(self, x):
        return 0.5 * x.dot(x) - np.log(np.cosh(self.Q.dot(x) + self.b)).sum()

    def dphi_dx(self, x):
        return x - self.Q.T.dot(np.tanh(self.Q.dot(x) + self.b))

    def independent_samples(self, n_sample, force=False,
                            prng=np.random.RandomState()):
        if self.n_unit > 10 and not force:
            print('Size of state space is large ({0}) '
                  'set force=True to proceed anyway'.format(2**self.n_unit))
        ps = np.empty(2**self.n_unit)
        for i, s in enumerate(it.product(*[[-1, 1]] * self.n_unit)):
            s = np.array(s)
            ps[i] = np.exp(0.5 * s.dot(self.W).dot(s) + s.dot(self.b))
        ps /= ps.sum()
        binary_state_counts = prng.multinomial(n_sample, ps)
        xs = np.empty((n_sample, self.n_unit))
        j = 0
        for i, s in enumerate(it.product(*[[-1, 1]] * self.n_unit)):
            mean = self.Q.T.dot(s)
            count = binary_state_counts[i]
            xs[j:j+count] = (mean[None, :] +
                             prng.normal(size=(count, self.n_unit)))
            j += count
        return xs, binary_state_counts


class IsotropicCoshGMRelaxation(object):

    def __init__(self, W, b, epsilon=0.):
        self.W = W
        self.b = b
        self.e, self.R = la.eigh(W)
        self.e_p = self.e - self.e.min() + epsilon
        self.n_unit = W.shape[0]
        self.V_inv = W + (epsilon - self.e.min()) * np.eye(self.n_unit)
        self.V_inv_sqrt = self.R * self.e_p**0.5
        if epsilon == 0.:
            self.V = la.pinv(self.V_inv)
            idx_e_min = self.e != self.e.min()
            self.e_nz = self.e[idx_e_min] - self.e.min()
            self.R_nz = self.R[:, idx_e_min]
            self.V_sqrt = self.R_nz / (self.e_nz)**0.5
        else:
            self.V = (self.R / self.e_p).dot(self.R.T)
            self.V_sqrt = self.R / self.e_p**0.5

    def phi(self, x):
        return (0.5 * (x - self.b).dot(self.V).dot(x - self.b) -
                np.log(np.cosh(x)).sum())

    def dphi_dx(self, x):
        return self.V.dot(x - self.b) - np.tanh(x)

    def independent_samples(self, n_sample, force=False,
                            prng=np.random.RandomState()):
        if self.n_unit > 10 and not force:
            print('Size of state space is large ({0}) '
                  'set force=True to proceed anyway'.format(2**self.n_unit))
        ps = np.empty(2**self.n_unit)
        for i, s in enumerate(it.product(*[[-1, 1]] * self.n_unit)):
            s = np.array(s)
            ps[i] = np.exp(0.5 * s.dot(self.W).dot(s) + s.dot(self.b))
        ps /= ps.sum()
        binary_state_counts = prng.multinomial(n_sample, ps)
        xs = np.empty((n_sample, self.n_unit))
        j = 0
        for i, s in enumerate(it.product(*[[-1, 1]] * self.n_unit)):
            mean = self.V_inv.dot(s) + self.b
            count = binary_state_counts[i]
            ns = prng.normal(size=(count, self.n_unit))
            xs[j:j+count] = (mean[None, :] + self.V_inv_sqrt.dot(ns.T).T)
            j += count
        return xs, binary_state_counts
