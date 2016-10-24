# -*- coding: utf-8 -*-
"""Sequential scan Gibbs sampler for Boltzmann machine.

Simple implementation of a standard single-unit update Gibbs sampler with
fixed sequential scan order.
"""

cimport randomkit_wrapper as rk
from cython.view cimport array
from bmtools.exact.helpers cimport state_t, state_t_code

cdef extern from 'math.h':
    double exp(double x) nogil


cdef class SequentialGibbsSampler:
    """Boltzmann machine using standard sequential Gibbs updates."""

    def __init__(SequentialGibbsSampler self, double[:, :] weights,
                 double[:] biases, int[:] update_order, unsigned long seed):
        self.n_unit = weights.shape[0]
        self.weights = weights
        self.biases = biases
        self.update_order = update_order
        self.rng = rk.RandomKit(seed)

    def get_samples(SequentialGibbsSampler self, state_t[:, :] samples,
                    state_t[:] state=None):
        """Get samples by performing sequential sweeps through units."""
        cdef int i, s, n_sample
        n_sample = samples.shape[0]
        if state is None:
            state = array(shape=(self.n_unit,), itemsize=sizeof(state_t),
                          format=state_t_code)
            for i in range(self.n_unit):
                state[i] = 2*(self.rng.uniform() < 0.5) - 1
        samples[0, :] = state
        for s in range(1, n_sample):
            for i in range(self.update_order.shape[0]):
                self.sample_unit(state, self.update_order[i])
            samples[s, :] = state

    cdef void sample_unit(SequentialGibbsSampler self, state_t[:] state,
                          int j):
        """Sample state of single unit conditional on rest of units."""
        cdef int i
        cdef double u = self.biases[j]
        for i in range(self.n_unit):
            u += self.weights[j, i] * state[i]
        cdef double p = 1. / (1. + exp(-2*u))
        state[j] = 2*(self.rng.uniform() < p) - 1
