# -*- coding: utf-8 -*-
"""Boltzmann machine moment calculation.

Functions for calculating first and second moments of Boltzmann machine
invariant distribution exactly given weight and bias parameters. Intended
for use only with small toy systems due to exponential scaling of
summations over state space with number of units. As well as a single
threaded sequential implementation a parallel implementation which can
be distributed over multiple compute units using OpenMP in a shared memory
architecture is provided.
"""

from cython.view cimport array
from cython.parallel import prange

from bmtools.exact.helpers cimport (
    state_t, state_t_code, next_state, neg_energy,
    partition_state_space, check_state_space_size, index_to_state)

cdef extern from 'math.h':
    double exp(double x) nogil
    double log(double x) nogil

def calculate_moments_sequential(double[:, :] weights, double[:] biases,
                                 bint force=False):
    """Calculate Boltzmann machine distribution moments.

    Calculates normalisation constant (~zeroth moment), first and second
    moments of Boltzmann machine invariant distribution specified by the
    provided weight and bias parameters. Moment calculations are done
    using a single thread iterating across all possible state configurations
    sequentially.

    Parameters
    ----------
    weights : double[:, :]
        Matrix of weight parameters. Should be symmetric and zero-diagonal.
    biases : double[:]
        Vector of bias parameters.
    force : bool (bint)
        Flag to override forced exit when number of units is more than 20
        due to large size of state space.

    Returns
    -------
    norm_const : double
        Sum of unnormalised probability terms across all states.
    first_mom : double[:]
        Expectation of state vector with respect to distribution.
    second_mom : double[:, :]
        Expectation of outer product of state vectors with respect to
        distribution.
    """
    cdef int state_index, i, j
    cdef int num_units = weights.shape[0]
    cdef double prob = 0.
    cdef double norm_const = 0.
    cdef int num_states = 2**num_units
    check_state_space_size(num_units, force)
    cdef state_t[:] state = array(
            shape=(num_units,), itemsize=sizeof(state_t), format=state_t_code)
    cdef double[:] first_mom = array(
            shape=(num_units,), itemsize=sizeof(double), format='d')
    cdef double[:, :] second_mom = array(
            shape=(num_units, num_units), itemsize=sizeof(double), format='d')
    accum_moments_for_state_range(weights, biases, state, &norm_const,
                                  first_mom, second_mom, 0, num_states)
    normalise_first_moment(first_mom, norm_const)
    normalise_and_reflect_second_moment(second_mom, norm_const)
    return norm_const, first_mom, second_mom


def calculate_moments_parallel(double[:, :] weights, double[:] biases,
                               bint force=False, int num_threads=2,
                               double[:] norm_consts=None,
                               double[:, :] first_moms=None,
                               double[:, :, :] second_moms=None):
    """Calculate Boltzmann machine distribution moments.

    Calculates normalisation constant (~zeroth moment), first and second
    moments of Boltzmann machine invariant distribution specified by the
    provided weight and bias parameters. Moment calculations are done in
    parallel using a specified number of thread each iterating across an
    equal partition of the state space.

    Parameters
    ----------
    weights : double[:, :]
        Matrix of weight parameters. Should be symmetric and zero-diagonal.
    biases : double[:]
        Vector of bias parameters.
    force : bool (bint)
        Flag to override forced exit when number of units is more than 20
        due to large size of state space.
    num_threads : int (default=2)
        Number of parallel threads to use.
    norm_consts : double[:], optional
        Allocated array to use in parallel normalisation constant
        calculation. Should have shape (num_threads,). Final value accumulated
        over all threads will be written to first entry. If not
        specified (or either of first_moms or second_moms not specidied) new
        array allocated and returned.
    first_moms : double[:, :], optional
        Allocated array to use in parallel first moment calculation. Should
        have shape (num_threads, num_units). Final value accumulated
        over all threads will be written to first element. If not specified
        (or either of norm_consts or second_moms not specified) new array
        allocated and first entry returned.
    second_moms : double[:, :, :], optional
        Allocated array to use in parallel second moment calculation. Should
        have shape (num_threads, num_units, num_units). Final values
        accumulated over all threads will be written to first entry. If not
        specified (or either of norm_consts or first_moms not specified) new
        array allocated and first entry returned.

    Returns
    -------
    **If no arrays specified for in-place calculation, otherwise no return**
    norm_const : double
        Sum of unnormalised probability terms across all states.
    first_mom : double[:]
        Expectation of state vector with respect to distribution.
    second_mom : double[:, :]
        Expectation of outer product of state vectors with respect to
        distribution.
    """
    cdef int t
    cdef int num_units = weights.shape[0]
    cdef double prob = 0.
    cdef int num_states = 2**num_units
    if num_threads <= 0:
        raise ValueError('Number of threads must be > 0')
    check_state_space_size(num_units, force)
    cdef state_t[:, :] states = array(
            shape=(num_threads, num_units), itemsize=sizeof(state_t),
            format=state_t_code)
    # check if any arrays for in place updates not specified and if so
    # initialise
    cdef bint all_in_place = True
    if norm_consts is None or first_moms is None or second_moms is None:
        all_in_place = False
        norm_consts = array(shape=(num_threads,),
                            itemsize=sizeof(double), format='d')
    if first_moms is None:
        all_in_place = False
        first_moms = array(shape=(num_threads, num_units),
                           itemsize=sizeof(double), format='d')
    if second_moms is None:
        all_in_place = False
        second_moms = array(shape=(num_threads, num_units, num_units),
                            itemsize=sizeof(double), format='d')
    # partition state space in to equal sized sections to allocate to
    # different parallel threads
    cdef int[:] intervals = partition_state_space(num_states, num_threads)
    # parallel loop over partitions of state space, with each thread
    # accumulating moments for its assigned states into thread-specific
    # arrays
    for t in prange(num_threads, nogil=True, schedule='static', chunksize=1,
                    num_threads=num_threads):
        norm_consts[t] = 0.
        accum_moments_for_state_range(
            weights, biases, states[t], &norm_consts[t], first_moms[t],
            second_moms[t], intervals[t], intervals[t+1])
    # accumulate normalisation constant terms calculated by each individual
    # thread to get overall value
    for t in range(1, num_threads):
        norm_consts[0] += norm_consts[t]
    # if multiple threads used accumulate moment values calculated by each
    # thread before normalising (and for second moments filling in
    # diagonal and upper triangle values using symmetricity)
    if num_threads > 1:
        combine_and_normalise_first_moments(first_moms, norm_consts[0])
        combine_normalise_and_reflect_second_moments(second_moms,
                                                     norm_consts[0])
    else:
        normalise_first_moment(first_moms[0], norm_consts[0])
        normalise_and_reflect_second_moment(second_moms[0], norm_consts[0])
    # only return values if not all arrays update in place
    if not all_in_place:
        return norm_consts[0], first_moms[0], second_moms[0]


def calculate_probs_parallel(
        double[:, :] weights, double[:] biases, bint force=False,
        int num_threads=2,):
    """Calculate BM distribution probabilities using parallel implementation.

    Calculates the probabilities of all signed binary states in according to a
    Boltzmann machine invariant distribution specified by the provided weight
    and bias parameters. Calculation is done by exahustive iteration over the
    2**num_units state space so should only be attempted for moderate
    dimensionalities. Calculations are done in parallel using a specified number
    of thread each iterating across an equal partition of the state space.

    Parameters
    ----------
    weights : double[:, :]
        Matrix of weight parameters. Should be symmetric and zero-diagonal.
    biases : double[:]
        Vector of bias parameters.
    force : bool (bint)
        Flag to override forced exit when number of units is more than 20
        due to large size of state space.
    num_threads : int
        Number of parallel threads to use.

    Returns
    -------
    probs : double[:]
        Array of normalised probabilities for all states.
    norm_const : double
        Sum of unnormalised probability terms across all states.
    """
    cdef int t
    cdef int num_units = weights.shape[0]
    cdef int num_states = 2 ** num_units
    if num_threads <= 0:
        raise ValueError('Number of threads must be > 0')
    check_state_space_size(num_units, force)
    cdef state_t[:, :] states = array(
        shape=(num_threads, num_units), itemsize=sizeof(state_t), format=state_t_code)
    cdef double[:] norm_consts = array(shape=(num_threads,),
        itemsize=sizeof(double), format='d')
    cdef double[:] probs = array(
        shape=(num_states,), itemsize=sizeof(double), format='d'
    )
    # partition state space in to equal sized sections to allocate to
    # different parallel threads
    cdef int[:] intervals = array(
            shape=(num_threads+1,), itemsize=sizeof(int), format='i')
    for t in range(num_threads):
        intervals[t] = <int>(t * <float>(num_states) / num_threads)
    intervals[num_threads] = num_states
    # parallel loop over partitions of state space, with each thread
    # calculating probabilities for its assigned states into thread-specific
    # slice of probs array
    for t in prange(num_threads, nogil=True, schedule='static', chunksize=1,
                    num_threads=num_threads):
        norm_consts[t] = 0.
        calc_unnormed_probs_for_state_range(
            weights, biases, states[t], &norm_consts[t],
            probs[intervals[t]:intervals[t+1]],
            intervals[t], intervals[t+1])
    # accumulate normalisation constant terms calculated by each individual
    # thread to get overall value
    for t in range(1, num_threads):
        norm_consts[0] += norm_consts[t]
    # normalise probabilities by dividing through by normalisation constant
    # in parallel over multiple threads
    for t in prange(num_threads, nogil=True, schedule='static', chunksize=1,
                    num_threads=num_threads):
        normalise_probabilities(probs[intervals[t]:intervals[t+1]],
                                norm_consts[0])
    return probs, norm_consts[0]


cdef void calc_unnormed_probs_for_state_range(
        double[:, :] weights, double[:] biases, state_t[:] state,
        double* norm_const, double[:] probs,
        int start_state_index, int end_state_index) nogil:
    """
    Calculates the unnormalised probabilities for a portion of the state space
    corresponding to a contiguous range of state integer indices.
    """
    cdef int i
    index_to_state(start_state_index, state)
    for i in range(end_state_index - start_state_index):
        probs[i] = exp(neg_energy(state, weights, biases))
        norm_const[0] += probs[i]
        next_state(state, start_state_index + i + 1)


cdef void normalise_probabilities(double[:] probs, double norm_const) nogil:
    """Divides an array of probabilities by a normalisation constant."""
    cdef int i
    for i in range(probs.shape[0]):
        probs[i] /= norm_const


cdef void accum_moments_for_state_range(
        double[:, :] weights, double[:] biases, state_t[:] state,
        double* norm_const, double[:] first_mom, double[:, :] second_mom,
        int start_state_index, int end_state_index) nogil:
    """
    Accumulate the moment values for a portion of the state space
    corresponding to a contiguous range of state integer indices.
    """
    cdef double prob = 0.
    cdef int state_index, i, j
    index_to_state(start_state_index, state)
    for i in range(weights.shape[0]):
        first_mom[i] = 0.
        for j in range(i):
            second_mom[i, j] = 0.
    for state_index in range(start_state_index, end_state_index):
        prob = exp(neg_energy(state, weights, biases))
        norm_const[0] += prob
        for i in range(state.shape[0]):
            first_mom[i] += state[i] * prob
            for j in range(i):
                second_mom[i, j] += state[i] * state[j] * prob
        next_state(state, state_index + 1)


cdef double calc_norm_const(double[:,:] weights, double[:] biases,
                            state_t[:] state, int start_state_index=0,
                            int end_state_index=-1) nogil:
    """
    Calculate the normalisation constant (i.e. sum of unnormalised
    probability terms) for distribution with specified parameters, only
    including probabilities of states with indices in specified range.
    """
    cdef double norm_const = 0.
    cdef int state_index
    index_to_state(start_state_index, state)
    if end_state_index == -1:
        end_state_index = 2**weights.shape[0]
    for state_index in range(start_state_index, end_state_index):
        norm_const += exp(neg_energy(state, weights, biases))
        next_state(state, state_index+1)
    return norm_const


cdef void normalise_first_moment(
        double[:] first_mom, double norm_const) nogil:
    """
    Divide all accumulated first moment values by normalisation constant.
    """
    cdef int i
    for i in range(first_mom.shape[0]):
        first_mom[i] /= norm_const


cdef void combine_and_normalise_first_moments(
        double[:, :] first_moms, double norm_const) nogil:
    """
    Accumulate first moment values calculated by different threads into
    elements corresponding to first thread (first_moms[0, :]) and then
    normalised accumulated values by normalisation constant.
    """
    cdef int i, j
    for i in range(1, first_moms.shape[0]):
        for j in range(first_moms.shape[1]):
            first_moms[0, j] += first_moms[i, j]
            if i == first_moms.shape[0] - 1:
                first_moms[0, j] /= norm_const


cdef void normalise_and_reflect_second_moment(double[:, :] second_mom,
                                              double norm_const) nogil:
    """
    Divide all accumulated second moment values in lower triangle of matrix
    by normalisation constant, set diagonal terms to constant value of 1 and
    set upper triangular terms from corresponding elements in lower triangle
    using fact second moment must be symmetric.
    """
    cdef int i, j
    for i in range(second_mom.shape[0]):
        second_mom[i, i] = 1.
        for j in range(i):
            second_mom[i, j] /= norm_const
            second_mom[j, i] = second_mom[i, j]


cdef void combine_normalise_and_reflect_second_moments(
        double[:, :, :] second_moms, double norm_const) nogil:
    """
    Accumulate second moment values calculated by different threads into
    elements corresponding to first thread (second_moms[0, :]) and
    then normalise values by normalisation constant, fill in diagonal with
    constant value of 1 and set upper triangular terms from corresponding
    elements in lower triangle using fact second moment must be symmetric.
    """
    cdef int i, j, k
    for i in range(1, second_moms.shape[0]):
        for j in range(second_moms.shape[1]):
            second_moms[0, j, j] = 1.
            for k in range(j):
                second_moms[0, j, k] += second_moms[i, j, k]
                if i == second_moms.shape[0] - 1:
                    second_moms[0, j, k] /= norm_const
                    second_moms[0, k, j] = second_moms[0, j, k]
