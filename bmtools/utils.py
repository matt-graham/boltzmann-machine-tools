# -*- coding: utf-8 -*-
"""Helper functions for working with Boltzmann machine distributions."""

import numpy as np


def random_orthogonal_matrix(num_dim, rng=np.random.RandomState()):
    """Generate a random orthogonal matrix.

    Based on qmult.m from Test Matrix Toolbox by N. Higham.

    Args:
        num_dim: Dimensionality of (square) matrix to be returned.
        rng (RandomState): Seeded random number generator object.

    Returns:
        Random orthogonal matrix of specified dimension.
    """
    orth_mtx = np.eye(num_dim)
    d = np.zeros(num_dim)
    for k in range(num_dim - 2, -1, -1):
        x = rng.randn(num_dim - k)
        s = x.dot(x)**0.5
        sgn = np.sign(x[0]) + float(x[0] == 0)
        s *= sgn
        d[k] = -sgn
        x[0] += s
        beta = s * x[0]
        y = x.dot(orth_mtx[k:])
        orth_mtx[k:] -= x[:, None] * y[None, :] / beta
    orth_mtx[:-1] = (orth_mtx[:-1].T * d[:-1]).T
    orth_mtx[-1] *= 2 * (rng.rand() > 0.5) - 1
    return orth_mtx


def random_sym_matrix_by_eigval(eigvals, rng=np.random.RandomState()):
    """Generate a random symmetric matrix with the specified eigenvalues.

    Args:
        eigvals (array): One-dimensional array of values to use as eigenvalues
            of generated symmetric matrix. The length of this array determines
            the dimension of the returned matrix.
        rng (RandomState): Seeded random number generator object.

    Returns:
        Tuple with first element a random symmetric matrix and the second the
        orthogonal matrix containing the eigenvectors of the symmetric matrix.
    """
    orth_mtx = random_orthogonal_matrix(eigvals.size, rng)
    return orth_mtx.dot(np.diag(eigvals)).dot(orth_mtx.T), orth_mtx
