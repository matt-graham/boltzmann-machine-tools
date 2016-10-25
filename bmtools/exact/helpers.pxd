# -*- coding: utf-8 -*-
"""Boltzmann machine helpers.

Cython helper functions for working with Boltzmann machine distributions.
"""

ctypedef signed char state_t
cdef char* state_t_code

cdef double neg_energy(
        state_t[:] state, double[:,:] weights, double[:] biases) nogil

cpdef void check_state_space_size(int num_unit, bint force)

cdef int[:] partition_state_space(int num_states, int num_threads)

cpdef void index_to_state(int index, state_t[:] state) nogil

cdef void next_state(state_t[:] state, int next_state_index) nogil
