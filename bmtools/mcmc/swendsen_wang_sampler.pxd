"""
=======================================
Swendsen-Wang Boltzmann machine sampler
=======================================

Wrapper class implementing Swendsen-Wang algorithm (spin-cluster method) for
MCMC sampling from a Boltzmann machine distribution (~Ising spin model).

References
----------

> Swendsen, R. H., and Wang, J.-S. (1987), 
> Nonuniversal critical dynamics in Monte Carlo simulations, 
> Phys. Rev. Lett., 58(2):86–88.

"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

cimport samplers.randomkit_wrapper as rk

ctypedef signed char state_t

cdef extern from 'math.h':
    double exp(double x)

cdef class SwendsenWangSampler:

    cdef:
        double[:, :] bond_probs, weights
        double[:] biases, cluster_biases, flip_probs
        state_t[:, :] bonds_state
        state_t[:] units_state
        int[:] clusters
        int n_unit
        rk.RandomKit rng

    cdef void allocate_arrays(SwendsenWangSampler self)

    cdef void sample_bonds_given_units(SwendsenWangSampler self)

    cdef void sample_units_given_bonds(SwendsenWangSampler self)

    cpdef get_samples(SwendsenWangSampler self, state_t[:, :] samples,
                      state_t[:] init_state)

    cdef int find_cluster_root(SwendsenWangSampler self, int i)

    cdef void construct_clusters_and_update_biases(SwendsenWangSampler self)

    cdef void calc_flip_probs_and_sample(SwendsenWangSampler self)

    cdef void sample_flip_states_given_bonds(SwendsenWangSampler self)
