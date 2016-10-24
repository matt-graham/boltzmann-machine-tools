# -*- coding: utf-8 -*-
"""Sequential scan Gibbs sampler for Boltzmann machine.

Simple implementation of a standard single-unit update Gibbs sampler with
fixed sequential scan order.
"""

cimport randomkit_wrapper as rk
from bmtools.exact.helpers cimport state_t


cdef class SequentialGibbsSampler:

    cdef double[:, :] weights
    cdef double[:] biases
    cdef int[:] update_order
    cdef int n_unit
    cdef rk.RandomKit rng

    cdef void sample_unit(SequentialGibbsSampler self, state_t[:] state,
                          int j)
