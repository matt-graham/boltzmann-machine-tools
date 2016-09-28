"""
===================================
Bolzmann machine moment calculation
===================================

Functions for calculating first and second moments of Boltzmann machine
invariant distribution exactly given weight and bias parameters. Intended
for use only with small toy systems due to exponential scaling of
summations over state space with number of units. As well as a single
threaded sequential implementation a parallel implementation which can
be distributed over multiple compute units using OpenMP in a shared memory
architecture is provided.
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

from cython.view cimport array
from cython.parallel import prange

cdef extern from 'math.h':
    double exp(double x) nogil
    double log(double x) nogil

ctypedef signed char state_t
cdef char* state_t_code = 'c'


def calculate_moments_sequential(double[:, :] weights, double[:] biases,
                                 bint force=False):
    """
    Calculate Boltzmann machine distribution moments using sequential
    implementation.

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
    cdef int n_unit = weights.shape[0]
    cdef double prob = 0.
    cdef double norm_const = 0.
    cdef int n_state = 2**n_unit
    if n_unit > 20 and not force:
        raise ValueError('Size of state space ({0}) very large. Set '
                         'force=True if you want to force calculation.'
                         .format(n_state))
    cdef state_t[:] state = array(
            shape=(n_unit,), itemsize=sizeof(state_t), format=state_t_code)
    cdef double[:] first_mom = array(
            shape=(n_unit,), itemsize=sizeof(double), format='d')
    cdef double[:, :] second_mom = array(
            shape=(n_unit, n_unit), itemsize=sizeof(double), format='d')
    accum_moments_for_state_range(weights, biases, state, &norm_const,
                                  first_mom, second_mom, 0, n_state)
    normalise_first_moment(first_mom, norm_const)
    normalise_and_reflect_second_moment(second_mom, norm_const)
    return norm_const, first_mom, second_mom


def calculate_moments_parallel(double[:, :] weights, double[:] biases,
                               bint force=False, int n_thread=2,
                               double[:] norm_consts=None,
                               double[:, :] first_moms=None,
                               double[:, :, :] second_moms=None):
    """
    Calculate Boltzmann machine distribution moments using parallel
    implementation.

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
    n_thread : int (default=2)
        Number of parallel threads to use.
    norm_consts : double[:], optional
        Allocated array to use in parallel normalisation constant
        calculation. Should have shape (n_thread,). Final value accumulated
        over all threads will be written to first entry. If not
        specified (or either of first_moms or second_moms not specidied) new
        array allocated and returned.
    first_moms : double[:, :], optional
        Allocated array to use in parallel first moment calculation. Should
        have shape (n_thread, n_unit). Final value accumulated
        over all threads will be written to first element. If not specified
        (or either of norm_consts or second_moms not specified) new array
        allocated and first entry returned.
    second_moms : double[:, :, :], optional
        Allocated array to use in parallel second moment calculation. Should
        have shape (n_thread, n_unit, n_unit). Final values accumulated over
        all threads will be written to first entry. If not specified (or
        either of norm_consts or first_moms not specified) new array
        allocated and first entry returned.

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
    cdef int n_unit = weights.shape[0]
    cdef double prob = 0.
    cdef int n_state = 2**n_unit
    if n_thread <= 0:
        raise ValueError('Number of threads must be > 0')
    if n_unit > 20 and not force:
        raise ValueError('Size of state space ({0}) very large. Set '
                         'force=True if you want to force calculation.'
                         .format(n_state))
    cdef state_t[:, :] states = array(
            shape=(n_thread, n_unit), itemsize=sizeof(state_t),
            format=state_t_code)
    # check if any arrays for in place updates not specified and if so
    # initialise
    cdef bint all_in_place = True
    if norm_consts is None or first_moms is None or second_moms is None:
        all_in_place = False
        norm_consts = array(shape=(n_thread,),
                            itemsize=sizeof(double), format='d')
    if first_moms is None:
        all_in_place = False
        first_moms = array(shape=(n_thread, n_unit),
                           itemsize=sizeof(double), format='d')
    if second_moms is None:
        all_in_place = False
        second_moms = array(shape=(n_thread, n_unit, n_unit),
                            itemsize=sizeof(double), format='d')
    # partition state space in to equal sized sections to allocate to
    # different parallel threads
    cdef int[:] intervals = array(
            shape=(n_thread+1,), itemsize=sizeof(int), format='i')
    for t in range(n_thread):
        intervals[t] = <int>(t * <float>(n_state) / n_thread)
    intervals[n_thread] = n_state
    # parallel loop over partitions of state space, with each thread
    # accumulating moments for its assigned states into thread-specific
    # arrays
    for t in prange(n_thread, nogil=True, schedule='static', chunksize=1,
                    num_threads=n_thread):
        norm_consts[t] = 0.
        accum_moments_for_state_range(
            weights, biases, states[t], &norm_consts[t], first_moms[t],
            second_moms[t], intervals[t], intervals[t+1])
    # accumulate normalisation constant terms calculated by each individual
    # thread to get overall value
    for t in range(1, n_thread):
        norm_consts[0] += norm_consts[t]
    # if multiple threads used accumulate moment values calculated by each
    # thread before normalising (and for second moments filling in
    # diagonal and upper triangle values using symmetricity)
    if n_thread > 1:
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
        int n_thread=2,):
    """Calculate BM distribution probabilities using parallel implementation.

    Calculates the probabilities of all signed binary states in according to a
    Boltzmann machine invariant distribution specified by the provided weight
    and bias parameters. Calculation is done by exahustive iteration over the
    2**n_unit state space so should only be attempted for moderate
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
    n_thread : int
        Number of parallel threads to use.

    Returns
    -------
    probs : double[:]
        Array of normalised probabilities for all states.
    norm_const : double
        Sum of unnormalised probability terms across all states.
    """
    cdef int t
    cdef int n_unit = weights.shape[0]
    cdef int n_state = 2 ** n_unit
    if n_thread <= 0:
        raise ValueError('Number of threads must be > 0')
    if n_unit > 20 and not force:
        raise ValueError('Size of state space ({0}) very large. Set '
                         'force=True if you want to force calculation.'
                         .format(n_state))
    cdef state_t[:, :] states = array(
        shape=(n_thread, n_unit), itemsize=sizeof(state_t), format=state_t_code)
    cdef double[:] norm_consts = array(shape=(n_thread,),
        itemsize=sizeof(double), format='d')
    cdef double[:] probs = array(
        shape=(n_state,), itemsize=sizeof(double), format='d'
    )
    # partition state space in to equal sized sections to allocate to
    # different parallel threads
    cdef int[:] intervals = array(
            shape=(n_thread+1,), itemsize=sizeof(int), format='i')
    for t in range(n_thread):
        intervals[t] = <int>(t * <float>(n_state) / n_thread)
    intervals[n_thread] = n_state
    # parallel loop over partitions of state space, with each thread
    # calculating probabilities for its assigned states into thread-specific
    # slice of probs array
    for t in prange(n_thread, nogil=True, schedule='static', chunksize=1,
                    num_threads=n_thread):
        norm_consts[t] = 0.
        calc_unnormed_probs_for_state_range(
            weights, biases, states[t], &norm_consts[t],
            probs[intervals[t]:intervals[t+1]],
            intervals[t], intervals[t+1])
    # accumulate normalisation constant terms calculated by each individual
    # thread to get overall value
    for t in range(1, n_thread):
        norm_consts[0] += norm_consts[t]
    # normalise probabilities by dividing through by normalisation constant
    # in parallel over multiple threads
    for t in prange(n_thread, nogil=True, schedule='static', chunksize=1,
                    num_threads=n_thread):
        normalise_probabilities(probs[intervals[t]:intervals[t+1]],
                                norm_consts[0])
    return probs, norm_consts[0]


def update_state_from_int_enum(int int_enum, state_t[:] state):
    """Sets a state array to correspond to a given integer state enumeration.

    Params
    ------
    int_enum : int
        Integer enumeration of a binary state space. The encoding is big-endian,
        that the most significant bit of the integer ID determines the first
        element of the binary state.
    state : state_t[:]
        Binary state array to populate. Will hold signed-binary state
        corresponding to `int_enum` after function returns.
    """
    cdef int i
    for i in range(state.shape[0]):
        state[i] = (int_enum % 2) * 2 - 1
        int_enum = int_enum // 2


def log_likelihood(state_t[:, :] data, double[:,:] weights,
                   double[:] biases, int n_thread=2):
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
    n_thread : int (default=2)
        Number of parallel threads to use.

    Returns
    -------
    log_lik : double[:]
        A memory view on an array of doubles with each entry the log
        likelihood for the corresponding entry in the data array.
    """
    if n_thread <= 0:
        raise ValueError('Number of threads must be > 0')
    cdef int t
    cdef int n_unit = weights.shape[0]
    cdef int n_state = 2**n_unit
    cdef int n_data = data.shape[0]
    cdef int n_visible = data.shape[1]
    cdef int n_hidden = n_unit - n_visible
    cdef state_t[:, :] states = array(
            shape=(n_thread, n_unit), itemsize=sizeof(state_t),
            format=state_t_code)
    cdef double[:] log_lik = array(
            shape=(n_data,), itemsize=sizeof(double), format='d')
    cdef double[:] norm_consts = array(
            shape=(n_thread,), itemsize=sizeof(double), format='d')
    cdef double[:, :] b_h_given_v = array(
            shape=(n_thread, n_hidden), itemsize=sizeof(double), format='d')
    cdef int[:, :] intervals = array(
            shape=(2, n_thread+1), itemsize=sizeof(int), format='i')
    # partition state space and data points between threads
    for t in range(n_thread):
        intervals[0, t] = <int>(t * <float>(n_state) / n_thread)
        intervals[1, t] = <int>(t * <float>(n_data) / n_thread)
    intervals[0, n_thread] = n_state
    intervals[1, n_thread] = n_data
    # get partitioned views of parameters for easier referencing
    cdef double[:, :] W_hh = weights[:n_hidden, :n_hidden]
    cdef double[:, :] W_hv = weights[:n_hidden, n_hidden:]
    cdef double[:, :] W_vv = weights[n_hidden:, n_hidden:]
    cdef double[:] b_h = biases[:n_hidden]
    cdef double[:] b_v = biases[n_hidden:]
    # set parallel threads calculating overall components of overall
    # normalisation constant and data dependent log likelihood terms
    for t in prange(n_thread, nogil=True, schedule='static', chunksize=1,
                    num_threads=n_thread):
        norm_consts[t] = calc_norm_const(
            weights, biases, states[t], intervals[0, t], intervals[0, t+1])
        accum_data_dependent_log_lik_terms(
            W_hh, W_hv, W_vv, b_h, b_v, b_h_given_v[t], data,
            states[t, :n_hidden], log_lik, intervals[1, t],
            intervals[1, t+1])
    # accumulate overall normalisation constant terms calculated by
    # different threads and take log
    cdef double log_norm_const = 0.
    for t in range(n_thread):
        log_norm_const += norm_consts[t]
    log_norm_const = log(log_norm_const)
    # adjust log-likelihood terms to account for overall normalisation
    # constant
    for i in range(n_data):
        log_lik[i] -= log_norm_const
    return log_lik


cdef void calc_unnormed_probs_for_state_range(
        double[:, :] weights, double[:] biases, state_t[:] state,
        double* norm_const, double[:] probs,
        int start_state_index, int end_state_index) nogil:
    """
    Calculates the unnormalised probabilities for a portion of the state space
    corresponding to a contiguous range of state integer indices.
    """
    cdef int k = start_state_index
    cdef int i
    for i in range(weights.shape[0]):
        if k & 1 == 1:
            state[i] = 1
        else:
            state[i] = -1
        k >>= 1
    for i in range(end_state_index - start_state_index):
        probs[i] = exp(energy(state, weights, biases))
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
    cdef int k = start_state_index
    cdef int state_index, i, j
    for i in range(weights.shape[0]):
        if k & 1 == 1:
            state[i] = 1
        else:
            state[i] = -1
        k >>= 1
        first_mom[i] = 0.
        for j in range(i):
            second_mom[i, j] = 0.
    for state_index in range(start_state_index, end_state_index):
        prob = exp(energy(state, weights, biases))
        norm_const[0] += prob
        for i in range(state.shape[0]):
            first_mom[i] += state[i] * prob
            for j in range(i):
                second_mom[i, j] += state[i] * state[j] * prob
        next_state(state, state_index+1)


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
        log_lik[i] += energy(data[i], W_vv, b_v)


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


cdef double calc_norm_const(double[:,:] weights, double[:] biases,
                            state_t[:] state, int start_state_index=0,
                            int end_state_index=-1) nogil:
    """
    Calculate the normalisation constant (i.e. sum of unnormalised
    probability terms) for distribution with specified parameters, only
    including probabilities of states with indices in specified range.
    """
    cdef double norm_const = 0.
    cdef int k = start_state_index
    cdef int state_index, i
    if end_state_index == -1:
        end_state_index = 2**weights.shape[0]
    for i in range(weights.shape[0]):
        if k & 1 == 1:
            state[i] = 1
        else:
            state[i] = -1
        k >>= 1
    for state_index in range(start_state_index, end_state_index):
        norm_const += exp(energy(state, weights, biases))
        next_state(state, state_index+1)
    return norm_const


cdef double energy(state_t[:] state, double[:,:] weights,
                   double[:] biases) nogil:
    """
    Calculate the energy (logarithm of unnormalised probability) for a
    specified state given weight and bias parameters.
    """
    cdef double energy = 0.
    cdef int i
    for i in range(state.shape[0]):
        energy += state[i] * biases[i]
        for j in range(i):
            energy += state[i] * weights[i,j] * state[j]
    return energy


cdef void next_state(state_t[:] state, int next_state_index) nogil:
    """
    Update state vector to proceed to next state in state space (for a
    fixed ordering of states determined by mapping them to integer indices).
    Should be used for iterating through state space without having to
    explcitly construct all possible states at once which would have a very
    high memory cost for large state spaces.
    """
    cdef int unit_index = 1
    state[0] *= -1
    while next_state_index % 2 == 0 and unit_index < state.shape[0]:
        state[unit_index] *= -1
        next_state_index /= 2
        unit_index += 1


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
