"""
===================================================
Sequential scan Gibbs sampler for Boltzmann machine
===================================================

Simple implementation of a standard single-unit update Gibbs sampler with
fixed sequential scan order.
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

cimport randomkit_wrapper as rk
include "shared_defs.pxd"

cdef class SequentialGibbsSampler:

    cdef double[:, :] weights
    cdef double[:] biases
    cdef int[:] update_order
    cdef int n_unit
    cdef rk.RandomKit rng

    cdef void sample_unit(SequentialGibbsSampler self, state_t[:] state,
                          int j)
