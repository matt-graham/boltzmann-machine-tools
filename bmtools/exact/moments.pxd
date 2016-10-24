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

from bmtools.exact.helpers cimport state_t


cdef void calc_unnormed_probs_for_state_range(
        double[:, :] weights, double[:] biases, state_t[:] state,
        double* norm_const, double[:] probs,
        int start_state_index, int end_state_index) nogil

cdef void normalise_probabilities(double[:] probs, double norm_const) nogil


cdef void accum_moments_for_state_range(
        double[:, :] weights, double[:] biases, state_t[:] state,
        double* norm_const, double[:] first_mom, double[:, :] second_mom,
        int start_state_index, int end_state_index) nogil


cdef double calc_norm_const(double[:,:] weights, double[:] biases,
                            state_t[:] state, int start_state_index=?,
                            int end_state_index=?) nogil


cdef void normalise_first_moment(
        double[:] first_mom, double norm_const) nogil


cdef void combine_and_normalise_first_moments(
        double[:, :] first_moms, double norm_const) nogil


cdef void normalise_and_reflect_second_moment(double[:, :] second_mom,
                                              double norm_const) nogil


cdef void combine_normalise_and_reflect_second_moments(
        double[:, :, :] second_moms, double norm_const) nogil
