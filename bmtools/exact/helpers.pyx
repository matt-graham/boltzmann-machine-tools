# -*- coding: utf-8 -*-
"""Boltzmann machine helpers.

Cython helper functions for working with Boltzmann machine distributions.
"""

from cython.view cimport array
state_t_code = 'c'


cdef double neg_energy(
        state_t[:] state, double[:,:] weights, double[:] biases) nogil:
    """
    Calculate the negative energy (logarithm of unnormalised probability) for a
    specified state given weight and bias parameters.
    """
    cdef double neg_energy = 0.
    cdef int i, j
    for i in range(state.shape[0]):
        neg_energy += state[i] * biases[i]
        for j in range(i):
            neg_energy += state[i] * weights[i, j] * state[j]
    return neg_energy


cdef int[:] partition_state_space(int num_states, int num_threads):
    """Partition state space in to equal sized intervals.

    Used to allocate state space range to different parallel threads.

    Args:
        num_state (int): Total number of states (2 ** num_units).
        num_threads (int): Number of parallel threads being used.

    Returns:
        Memory view of integer array of state space index intervals.
    """
    cdef int t
    cdef int[:] intervals = array(
            shape=(num_threads + 1,), itemsize=sizeof(int), format='i')
    for t in range(num_threads):
        intervals[t] = <int>(t * <float>(num_states) / num_threads)
    intervals[num_threads] = num_states
    return intervals


cpdef void check_state_space_size(int num_units, bint force):
    """Raises ValueError if size of state space is large unless force set."""
    if num_units > 20 and not force:
        raise ValueError('Size of state space ({0}) very large. Set '
                         'force=True if you want to force calculation.'
                         .format(2**num_units))


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


cpdef void index_to_state(int index, state_t[:] state) nogil:
    """Update signed binary state vector to correspond to a state space index.

    The encoding is big-endian, that is the most significant bit of the
    integer index determines the first element of the binary state.

    Args:
        index (int): Integer state space enumeration index.
        state (state_t[:]): Binary state vector to update.
    """
    cdef int i
    for i in range(state.shape[0]):
        if index & 1 == 1:
            state[i] = 1
        else:
            state[i] = -1
        index >>= 1
