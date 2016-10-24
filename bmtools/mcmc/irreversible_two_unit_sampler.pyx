# -*- coding: utf-8 -*-
"""Irreversible two-unit Boltzmann machine sampler.

Boltzmann machine sampler with irreversible two-unit update dynamics.

Update dynamics leave Boltzmann machine distribution invariant and consist
of sequential composition of irreversible dynamics which resample states
of pairs of units at a time.
"""

cimport randomkit_wrapper as rk
from bmtools.exact.helpers cimport state_t, state_t_code
from cython.view cimport array

cdef extern from 'math.h':
    double exp(double x) nogil


cdef class IrreversibleTwoUnitSampler:
    """
    Boltzmann machine sampler with irreversible two-unit update dynamics

    Update dynamics leave Boltzmann machine distribution invariant and consist
    of sequential composition of irreversible dynamics which resample states
    of pairs of units at a time
    """

    def __init__(IrreversibleTwoUnitSampler self, double[:, :] weights,
                 double[:] biases, int[:, :] update_pairs, unsigned long seed):
        self.n_unit = weights.shape[0]
        self.weights = weights
        self.biases = biases
        self.update_pairs = update_pairs
        self.rng = rk.RandomKit(seed)

    def get_samples(IrreversibleTwoUnitSampler self, int n_sample,
                    state_t[:] init_state=None):
        """
        Get samples by performing specificied number of sweeps through pairs of
        units
        """
        cdef int i, k, l, s, p
        cdef state_t[:] state
        cdef state_t[:, :] samples = array(
            shape=(n_sample,self.n_unit), itemsize=sizeof(state_t),
            format=state_t_code)
        if init_state is None:
            state = array(shape=(self.n_unit,), itemsize=sizeof(state_t),
                          format=state_t_code)
            for i in range(self.n_unit):
                state[i] = 2*(self.rng.uniform() < 0.5) - 1
        else:
            state = init_state
        for s in range(n_sample):
            for p in range(self.update_pairs.shape[0]):
                k, l = self.update_pairs[p, 0], self.update_pairs[p, 1]
                self.sample_pair(state, k, l)
            samples[s,:] = state
        return samples

    cdef void sample_pair(IrreversibleTwoUnitSampler self,
                          state_t[:] state, int k, int l):
        """
        Resample the binary states of a pair of units using an irreversible
        'cyclonic' update dynamic
        """
        # calculate integer corresponding to current unit pair config
        cdef int curr_pair_state_index = (state[k]+1) + (state[l]+1)/2
        # get unormalised probabilities of four configs of binary pair
        cdef double probs[4]
        self.calc_pair_probs(state, k, l, probs)
        # explicit four element sort quicker than built-ins
        cdef int order[4]
        argsort_4(probs, order)
        # new state depends on current state due to irreversibility
        cdef double q
        cdef int next_pair_state_index
        if curr_pair_state_index == order[3]:
            q = self.rng.uniform() * probs[order[3]]
            if q <= probs[order[0]]:
                next_pair_state_index = order[0]
            elif q <= probs[order[2]]:
                next_pair_state_index = order[2]
            else:
                next_pair_state_index = order[3]
        elif curr_pair_state_index == order[2]:
            q = self.rng.uniform() * probs[order[2]]
            if q <= probs[order[1]]:
                next_pair_state_index = order[1]
            else:
                next_pair_state_index = order[3]
        # if current state one of two with lowest cond prob make
        # deterministic move to one of higher cond prob states
        elif curr_pair_state_index == order[1]:
            next_pair_state_index = order[3]
        else:
            next_pair_state_index = order[2]
        # update configuration of unit pair
        state[k] = 2*(next_pair_state_index/2)-1
        state[l] = 2*(next_pair_state_index%2)-1

    cdef void calc_pair_probs(IrreversibleTwoUnitSampler self,
                              state_t[:] state, int k, int l,
                              double pair_probs[4]):
        """
        Calculates *unnormalised* conditional probabilities of different
        binary configurations of unit pair to be updated given current
        configuration of rest of units.
        """
        # set configuration of unit pair to zero thus masking them from
        # dot product with weight matrix - hacky however unit pair will be
        # reassigned in sample_pair anyway
        state[k] = 0
        state[l] = 0
        # calculate weighted sum of current unit states excluding pair to be
        # updated
        cdef double x_k = self.biases[k]
        cdef double x_l = self.biases[l]
        cdef int i
        for i in range(self.n_unit):
            x_k += self.weights[k,i] * state[i]
            x_l += self.weights[l,i] * state[i]
        # get weight corresponding to unit pair being updated
        cdef double W_kl = self.weights[k, l]
        # set probabilities of four binary pair possible configurations
        pair_probs[0] = exp(-x_k - x_l + W_kl)
        pair_probs[1] = exp(-x_k + x_l - W_kl)
        pair_probs[2] = exp(x_k - x_l - W_kl)
        pair_probs[3] = exp(x_k + x_l + W_kl)
