# -*- coding: utf-8 -*-
"""Boltzmann machine variational inference.

Functions for fitting an approximate distribution to a target Boltzmann
machine distribution by minimising a variational objective corresponding to
the Kullback-Leibler (KL) divergence between approximate and target
distributions. The expectations are calculated exactly (rather than for
example using a Monte Carlo estimator).
"""

from cython.view cimport array
from cython.parallel import prange

from bmtools.exact.helpers cimport (
    state_t, state_t_code, next_state, neg_energy,
    partition_state_space, check_state_space_size, index_to_state)

cdef extern from 'math.h':
    double exp(double x) nogil
    double log(double x) nogil
    double tanh(double x) nogil


def kl_divergence(
        double[:, :] weights_1, double[:] biases_1,
        double[:, :] weights_2, double[:] biases_2,
        bint force=False, int num_threads=4):
    """Calculates the KL divergence between 2 Boltzmann machine distributions.

    If the two distributions are defined as

    ```
    def prob_1(s):
        return exp(0.5 * s.dot(weights_1).dot(s) + s.dot(biases_1)) / Z_1
    def prob_2(s):
        return exp(0.5 * s.dot(weights_2).dot(s) + s.dot(biases_2)) / Z_2
    ```

    with `s[i] = +/- 1`, then the KL divergence is calculated as

    ```
    kl_div = 0
    for s in state_space:
        kl_div += prob_1(s) * log(prob_1(s) / prob_2(s))
    ```

    Args:
        weights_1 (double[:, :]): Weight parameters of first distribution.
        biases_1 (double[:]): Bias parameters of first distribution.
        weights_2 (double[:, :]): Weight parameters of second distritbution.
        biases_2 (double[:]): Bias parameters of second distribution.
        force (bool): Whether to force calculation for large state spaces.
        num_threads (int): Number of parallel threads to use in calculation.

    Returns:
        kl_div (double): KL divergence between two distributions.
        log_norm_const_1 (double): Logarithm of normalising constant for
           first distribution (i.e. `log(Z_1)` in expression above).
        log_norm_const_2 (double): Logarithm of normalising constant for
           second distribution (i.e. `log(Z_2)` in expression above).
    """
    cdef int num_units = weights_1.shape[0]
    check_state_space_size(num_units, force)
    cdef int num_states = 2**num_units
    cdef int[:] intervals = partition_state_space(num_states, num_threads)
    cdef state_t[:, :] states = array(
        shape=(num_threads, num_units), itemsize=sizeof(state_t),
        format=state_t_code)
    cdef double[:] kl_div_terms = array(
        shape=(num_threads,), itemsize=sizeof(double), format='d')
    cdef double[:] norm_const_1_terms = array(
        shape=(num_threads,), itemsize=sizeof(double), format='d')
    cdef double[:] norm_const_2_terms = array(
        shape=(num_threads,), itemsize=sizeof(double), format='d')
    cdef int t
    for t in prange(num_threads, nogil=True, schedule='static', chunksize=1,
                    num_threads=num_threads):
        kl_div_terms[t] = 0.
        norm_const_1_terms[t] = 0.
        norm_const_2_terms[t] = 0.
        accum_kl_terms_for_state_range(
            weights_1, biases_1, weights_2, biases_2,
            states[t], &kl_div_terms[t],
            &norm_const_1_terms[t], &norm_const_2_terms[t],
            intervals[t], intervals[t+1])
    for t in range(1, num_threads):
        norm_const_1_terms[0] += norm_const_1_terms[t]
        norm_const_2_terms[0] += norm_const_2_terms[t]
        kl_div_terms[0] += kl_div_terms[t]
    kl_div_terms[0] /= norm_const_1_terms[0]
    cdef double log_norm_const_1 = log(norm_const_1_terms[0])
    cdef double log_norm_const_2 = log(norm_const_2_terms[0])
    kl_div_terms[0] += log_norm_const_2 - log_norm_const_1
    return kl_div_terms[0], log_norm_const_1, log_norm_const_2


def kl_divergence_and_gradients(
        double[:, :] weights_1, double[:] biases_1,
        double[:, :] weights_2, double[:] biases_2,
        bint force=False, int num_threads=4):
    """
    Calculates the KL divergence between two Boltzmann machine distributions
    and its gradients with respect to the paramters of the first distribution.

    Args:
        weights_1 (double[:, :]): Weight parameters of first distribution.
        biases_1 (double[:]): Bias parameters of first distribution.
        weights_2 (double[:, :]): Weight parameters of second distritbution.
        biases_2 (double[:]): Bias parameters of second distribution.
        force (bool): Whether to force calculation for large state spaces.
        num_threads (int): Number of parallel threads to use in calculation.

    Returns:
        kl_div (double): KL divergence between two distributions.
        log_norm_const_1 (double): Logarithm of normalising constant for
           first distribution (i.e. `log(Z_1)` in expression above).
        log_norm_const_2 (double): Logarithm of normalising constant for
           second distribution (i.e. `log(Z_2)` in expression above).
        grads_wrt_biases_1 (double[:]): Gradients of KL divergence with
            respect to bias parameters of first distribution.
        grads_wrt_weights_1 (double[:, :]): Gradients of KL divergence with
            respect to weight parameters of second distribution.
        first_mom_1 (double[:]): First moments of first distribution.
        second_mom_1 (double[:, :]): Second moments of first distribution.
    """
    cdef int num_units = weights_1.shape[0]
    check_state_space_size(num_units, force)
    cdef int num_states = 2**num_units
    cdef int[:] intervals = partition_state_space(num_states, num_threads)
    cdef state_t[:, :] states = array(
        shape=(num_threads, num_units), itemsize=sizeof(state_t),
        format=state_t_code)
    cdef double[:] kl_div_terms = array(
        shape=(num_threads,), itemsize=sizeof(double), format='d')
    cdef double[:] norm_const_1_terms = array(
        shape=(num_threads,), itemsize=sizeof(double), format='d')
    cdef double[:] norm_const_2_terms = array(
        shape=(num_threads,), itemsize=sizeof(double), format='d')
    cdef double[:, :] first_mom_1_terms = array(
        shape=(num_threads, num_units), itemsize=sizeof(double),
        format='d')
    cdef double[:, :, :] second_mom_1_terms = array(
        shape=(num_threads, num_units, num_units), itemsize=sizeof(double),
        format='d')
    cdef double[:, :] grads_wrt_biases_1 = array(
        shape=(num_threads, num_units), itemsize=sizeof(double),
        format='d')
    cdef double[:, :, :] grads_wrt_weights_1 = array(
        shape=(num_threads, num_units, num_units), itemsize=sizeof(double),
        format='d')
    cdef int t, i, j
    for t in prange(num_threads, nogil=True, schedule='static', chunksize=1,
                    num_threads=num_threads):
        kl_div_terms[t] = 0.
        norm_const_1_terms[t] = 0.
        norm_const_2_terms[t] = 0.
        accum_kl_and_grad_terms_for_state_range(
            weights_1, biases_1, weights_2, biases_2,
            states[t], &kl_div_terms[t],
            &norm_const_1_terms[t], &norm_const_2_terms[t],
            first_mom_1_terms[t], second_mom_1_terms[t],
            grads_wrt_biases_1[t], grads_wrt_weights_1[t],
            intervals[t], intervals[t+1])
    for t in range(1, num_threads):
        norm_const_1_terms[0] += norm_const_1_terms[t]
        norm_const_2_terms[0] += norm_const_2_terms[t]
        kl_div_terms[0] += kl_div_terms[t]
        for i in range(num_units):
            first_mom_1_terms[0][i] += first_mom_1_terms[t][i]
            grads_wrt_biases_1[0][i] += grads_wrt_biases_1[t][i]
            for j in range(i):
                second_mom_1_terms[0][i, j] += second_mom_1_terms[t][i, j]
                grads_wrt_weights_1[0][i, j] += grads_wrt_weights_1[t][i, j]
    kl_div_terms[0] /= norm_const_1_terms[0]
    for i in range(num_units):
        first_mom_1_terms[0][i] /= norm_const_1_terms[0]
        grads_wrt_biases_1[0][i] /= norm_const_1_terms[0]
        grads_wrt_biases_1[0][i] -= (
            kl_div_terms[0] * first_mom_1_terms[0][i])
        second_mom_1_terms[0][i, i] = 1.
        grads_wrt_weights_1[0][i, i] = 0.
        for j in range(i):
            second_mom_1_terms[0][i, j] /= norm_const_1_terms[0]
            second_mom_1_terms[0][j, i] = second_mom_1_terms[0][i, j]
            grads_wrt_weights_1[0][i, j] /= norm_const_1_terms[0]
            grads_wrt_weights_1[0][i, j] -= (
                kl_div_terms[0] * second_mom_1_terms[0][i, j]
            )
            grads_wrt_weights_1[0][j, i] = grads_wrt_weights_1[0][i, j]
    cdef double log_norm_const_1 = log(norm_const_1_terms[0])
    cdef double log_norm_const_2 = log(norm_const_2_terms[0])
    kl_div_terms[0] += log_norm_const_2 - log_norm_const_1
    return (
        kl_div_terms[0], log_norm_const_1, log_norm_const_2,
        grads_wrt_biases_1[0], grads_wrt_weights_1[0],
        first_mom_1_terms[0], second_mom_1_terms[0]
    )


cdef void accum_kl_terms_for_state_range(
        double[:, :] weights_1, double[:] biases_1,
        double[:, :] weights_2, double[:] biases_2,
        state_t[:] state, double* kl_div_term,
        double* norm_const_1_term, double* norm_const_2_term,
        int start_state_index, int end_state_index) nogil:
    """Accumulates KL divergence terms for a subset of states."""
    cdef int index_offset
    cdef double neg_energy_1, neg_energy_2, unrm_prob_1, unrm_prob_2
    index_to_state(start_state_index, state)
    for index_offset in range(end_state_index - start_state_index):
        neg_energy_1 = neg_energy(state, weights_1, biases_1)
        neg_energy_2 = neg_energy(state, weights_2, biases_2)
        unrm_prob_1 = exp(neg_energy_1)
        unrm_prob_2 = exp(neg_energy_2)
        norm_const_1_term[0] += unrm_prob_1
        norm_const_2_term[0] += unrm_prob_2
        kl_div_term[0] += unrm_prob_1 * (neg_energy_1 - neg_energy_2)
        next_state(state, start_state_index + index_offset + 1)


cdef void accum_kl_and_grad_terms_for_state_range(
        double[:, :] weights_1, double[:] biases_1,
        double[:, :] weights_2, double[:] biases_2,
        state_t[:] state, double* kl_div_term,
        double* norm_const_1_term, double* norm_const_2_term,
        double[:] first_mom_1, double[:, :] second_mom_1,
        double[:] grads_wrt_biases_1, double[:, :] grads_wrt_weights_1,
        int start_state_index, int end_state_index) nogil:
    """Accumulates KL divergence and gradient terms for a subset of states."""
    cdef int i, j, index_offset
    cdef double neg_energy_1, neg_energy_2, unrm_prob_1, unrm_prob_2
    index_to_state(start_state_index, state)
    # Initialise terms to zero
    for i in range(state.shape[0]):
        first_mom_1[i] = 0.
        grads_wrt_biases_1[i] = 0.
        for j in range(i):
            second_mom_1[i, j] = 0.
            grads_wrt_weights_1[i, j] = 0.
    # Accumulate expectation terms
    for index_offset in range(end_state_index - start_state_index):
        neg_energy_1 = neg_energy(state, weights_1, biases_1)
        neg_energy_2 = neg_energy(state, weights_2, biases_2)
        unrm_prob_1 = exp(neg_energy_1)
        unrm_prob_2 = exp(neg_energy_2)
        norm_const_1_term[0] += unrm_prob_1
        norm_const_2_term[0] += unrm_prob_2
        kl_div_term[0] += unrm_prob_1 * (neg_energy_1 - neg_energy_2)
        for i in range(state.shape[0]):
            first_mom_1[i] += unrm_prob_1 * state[i]
            grads_wrt_biases_1[i] += (
                unrm_prob_1 * (neg_energy_1 - neg_energy_2) * state[i])
            for j in range(i):
                second_mom_1[i, j] += unrm_prob_1 * state[i] * state[j]
                grads_wrt_weights_1[i, j] += (
                    unrm_prob_1 * (neg_energy_1 - neg_energy_2) *
                    state[i] * state[j]
                )
        next_state(state, start_state_index + index_offset + 1)


cdef double logistic_sigmoid(double x) nogil:
    """Calculates the logistic sigmoid function."""
    return 1. / (1. + exp(-x))


def var_obj_and_grads_mean_field(
        double[:] var_biases, double[:, :] weights, double[:] biases):
    """Calculate mean-field variational objective and gradients.

    Calculates the variational free-energy for a mean-field / full-factorised
    variational approximation for a Boltzmann machine distribution and its
    gradients with respect to the variational bias parameters. The form of the
    approximating distribution is

    ```
    def var_approx_prob(s):
        return prod(logistic_sigmoid(2 * s * var_biases))
    ```

    The variational free-energy corresponds to the KL divergence from the
    target distribution to the approximating distribution minus the logarithm
    of the normalising constant of the target distribution. As the KL
    divergence is bounded from below by zero, the negative variational
    objective is therefore a lower bound on the log normalisation constant of the target distribution with equality if and only if the target and
    approximating distribution are equal across all states.

    Args:
        var_biases (double[:]): Parameters of approximate distribution.
        weights (double[:, :]): Weight parameters of target distribution.
        biases (double[:]): Bias parameters of target distribution.

    Returns:
        var_obj (double): Value of variational objective.
        grads_wrt_var_biases (double[:]): Gradient of objective with respect
            to variational bias parameters.
        var_first_mom (double[:]): First moments of variational distribution.
    """
    cdef int num_units = var_biases.shape[0]
    cdef int i, j
    cdef double[:] var_first_mom = array(
        shape=(num_units,), itemsize=sizeof(double), format='d')
    cdef double[:] grads_wrt_var_biases = array(
        shape=(num_units,), itemsize=sizeof(double), format='d')
    cdef double var_obj = 0.
    cdef double prob_p_i, prob_m_i
    for i in range(num_units):
        var_first_mom[i] = tanh(var_biases[i])
    for i in range(num_units):
        grads_wrt_var_biases[i] = 0.
        for j in range(num_units):
            grads_wrt_var_biases[i] -= weights[i, j] * var_first_mom[j]
        prob_p_i = logistic_sigmoid(2 * var_biases[i])
        prob_m_i = 1. - prob_p_i
        var_obj += (
            0.5 * var_first_mom[i] * grads_wrt_var_biases[i] -
            var_first_mom[i] * biases[i]
            + prob_p_i * log(prob_p_i) + prob_m_i * log(prob_m_i)
        )
        grads_wrt_var_biases[i] += var_biases[i] - biases[i]
        grads_wrt_var_biases[i] *= 4. * prob_p_i * prob_m_i
    return var_obj, grads_wrt_var_biases, var_first_mom
