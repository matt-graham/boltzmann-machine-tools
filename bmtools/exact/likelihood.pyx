# -*- coding: utf-8 -*-
"""Boltzmann machine likelihood.

Functions for calculating exact log-likelihoods for a set of binary data
vectors with respect to a Boltzmann machine disstribution.

Note log-likelihoods can be calculated from normalisation constants
already calculated during gradient updates so that should be preferred method
for tracking progression of log-likelihood during training rather than using
functions here. This intended for use in evaluating rather than training
models.
"""

from cython.view cimport array
from cython.parallel import prange

from bmtools.exact.helpers cimport state_t, state_t_code, neg_energy
from bmtools.exact.moments cimport calc_norm_const

cdef extern from 'math.h':
    double log(double x) nogil

def log_likelihood(state_t[:, :] data, double[:,:] weights,
                   double[:] biases, int num_threads=2):
    """
    Calculate log-likelihoods of data points.

    Calculate the log-likeihoods of a set of binary data vectors with
    respect to a Boltzmann machine invariant distribution with specified
    weight and bias parameters. Number of hidden units implicitly defined
    by difference in size of weight matrix compared to length of data
    vectors with hidden units assumed to be corresponds to entries at
    beginning of weight matrix / bias vector then visible units.

    Note log-likelihoods can be calculated from normalisation constants
    already calculated during gradient updates so that should be
    preferred method for tracking progression of log-likelihood during
    training rather than using this function.

    Parameters
    ----------
    data : state_t[:]
        Array of binary data vectors. Should have shape (n_data, n_visible)
        where n_data is the number of data points and n_visible is the
        number of visible units in system (which must be less than or equal
        to the total number of units as implicitly defined by shape of
        weight matrix and bias vector). Whether data vector binary elements
        are unsigned binary values (0/1) or signed binary values (-1/+1)
        should be determined by convention used when setting parameters.
    weights : double[:, :]
        Matrix of weight parameters. Should be symmetric and zero-diagonal.
    biases : double[:]
        Vector of bias parameters.
    num_threads : int (default=2)
        Number of parallel threads to use.

    Returns
    -------
    log_lik : double[:]
        A memory view on an array of doubles with each entry the log
        likelihood for the corresponding entry in the data array.
    """
    if num_threads <= 0:
        raise ValueError('Number of threads must be > 0')
    cdef int t
    cdef int n_unit = weights.shape[0]
    cdef int n_state = 2**n_unit
    cdef int n_data = data.shape[0]
    cdef int n_visible = data.shape[1]
    cdef int n_hidden = n_unit - n_visible
    cdef state_t[:, :] states = array(
            shape=(num_threads, n_unit), itemsize=sizeof(state_t),
            format=state_t_code)
    cdef double[:] log_lik = array(
            shape=(n_data,), itemsize=sizeof(double), format='d')
    cdef double[:] norm_consts = array(
            shape=(num_threads,), itemsize=sizeof(double), format='d')
    cdef double[:, :] b_h_given_v = array(
            shape=(num_threads, n_hidden), itemsize=sizeof(double), format='d')
    cdef int[:, :] intervals = array(
            shape=(2, num_threads+1), itemsize=sizeof(int), format='i')
    # partition state space and data points between threads
    for t in range(num_threads):
        intervals[0, t] = <int>(t * <float>(n_state) / num_threads)
        intervals[1, t] = <int>(t * <float>(n_data) / num_threads)
    intervals[0, num_threads] = n_state
    intervals[1, num_threads] = n_data
    # get partitioned views of parameters for easier referencing
    cdef double[:, :] W_hh = weights[:n_hidden, :n_hidden]
    cdef double[:, :] W_hv = weights[:n_hidden, n_hidden:]
    cdef double[:, :] W_vv = weights[n_hidden:, n_hidden:]
    cdef double[:] b_h = biases[:n_hidden]
    cdef double[:] b_v = biases[n_hidden:]
    # set parallel threads calculating overall components of overall
    # normalisation constant and data dependent log likelihood terms
    for t in prange(num_threads, nogil=True, schedule='static', chunksize=1,
                    num_threads=num_threads):
        norm_consts[t] = calc_norm_const(
            weights, biases, states[t], intervals[0, t], intervals[0, t+1])
        accum_data_dependent_log_lik_terms(
            W_hh, W_hv, W_vv, b_h, b_v, b_h_given_v[t], data,
            states[t, :n_hidden], log_lik, intervals[1, t],
            intervals[1, t+1])
    # accumulate overall normalisation constant terms calculated by
    # different threads and take log
    cdef double log_norm_const = 0.
    for t in range(num_threads):
        log_norm_const += norm_consts[t]
    log_norm_const = log(log_norm_const)
    # adjust log-likelihood terms to account for overall normalisation
    # constant
    for i in range(n_data):
        log_lik[i] -= log_norm_const
    return log_lik


cdef void accum_data_dependent_log_lik_terms(
        double[:, :] W_hh, double[:, :] W_hv, double[:, :] W_vv,
        double[:] b_h, double[:] b_v, double[:] b_h_given_v,
        state_t[:, :] data, state_t[:] state, double[:] log_lik,
        int start_data_index, int end_data_index) nogil:
    """
    Accumulate the data-dependent log likelihood terms for a specified
    range of data vectors.
    """
    cdef int i
    for i in range(start_data_index, end_data_index):
        calc_b_h_given_v(W_hv, data[i], b_h, b_h_given_v)
        log_lik[i] = log(calc_norm_const(W_hh, b_h_given_v, state))
        log_lik[i] += neg_energy(data[i], W_vv, b_v)


cdef double calc_b_h_given_v(double[:, :] W_hv, state_t[:] v, double[:] b_h,
                             double[:] b_h_given_v) nogil:
    """
    Calculate the effective bias vector for the hidden units when visible
    units are clamped (i.e. the bias vector parameter for the conditional
    distribution on the hidden units given known configuration of the
    visible units).
    """
    cdef int i, j
    for i in range(W_hv.shape[0]):
        b_h_given_v[i] = b_h[i]
        for j in range(W_hv.shape[1]):
            b_h_given_v[i] += W_hv[i, j] * v[j]
