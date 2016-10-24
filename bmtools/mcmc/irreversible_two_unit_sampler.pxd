# -*- coding: utf-8 -*-
"""Irreversible two-unit Boltzmann machine sampler.

Boltzmann machine sampler with irreversible two-unit update dynamics.

Update dynamics leave Boltzmann machine distribution invariant and consist
of sequential composition of irreversible dynamics which resample states
of pairs of units at a time.
"""

cimport randomkit_wrapper as rk
from bmtools.exact.helpers cimport state_t


cdef inline void argsort_4(double seq[4], int order[4]):
    """ Explicit five comparison ascending argsort of four values """
    if seq[0] <= seq[1]:
        order[0], order[1] = 0, 1
    else:
        order[0], order[1] = 1, 0
    if seq[2] <= seq[3]:
        order[2], order[3] = 2, 3
    else:
        order[2], order[3] = 3, 2
    if seq[order[0]] > seq[order[2]]:
        order[0], order[2] = order[2], order[0]
    if seq[order[1]] > seq[order[3]]:
        order[1], order[3] = order[3], order[1]
    if seq[order[1]]> seq[order[2]]:
        order[1], order[2] = order[2], order[1]


cdef class IrreversibleTwoUnitSampler:

    cdef double[:, :] weights
    cdef double[:] biases
    cdef int[:, :] update_pairs
    cdef int n_unit
    cdef rk.RandomKit rng

    cdef void sample_pair(IrreversibleTwoUnitSampler self, state_t[:] state,
                          int k, int l)

    cdef void calc_pair_probs(IrreversibleTwoUnitSampler self,
                              state_t[:] state, int k, int l,
                              double pair_probs[4])
