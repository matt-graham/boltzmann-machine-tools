# -*- coding: utf-8 -*-
"""Swendsen-Wang Boltzmann machine sampler.

Wrapper class implementing Swendsen-Wang algorithm (spin-cluster method) for
MCMC sampling from a Boltzmann machine distribution (~Ising spin model).

References
----------
> Swendsen, R. H., and Wang, J.-S. (1987),
> Nonuniversal critical dynamics in Monte Carlo simulations,
> Phys. Rev. Lett., 58(2):86–88.
"""

from cython.view cimport array
from bmtools.exact.helpers cimport state_t, state_t_code

cdef extern from 'math.h':
    double exp(double x)

cdef class SwendsenWangSampler:

    def __init__(SwendsenWangSampler self, double[:, :] weights,
                 double[:] biases, long seed):
        cdef int i, j
        self.n_unit = weights.shape[0]
        self.weights = weights
        self.biases = biases
        self.allocate_arrays()
        for i in range(self.n_unit):
            for j in range(i):
                self.bond_probs[i, j] = 1 - exp(-abs(weights[i, j]) * 2.)
        self.rng = rk.RandomKit(seed)

    cdef void allocate_arrays(SwendsenWangSampler self):
        """ Allocate memory for various internal state memory views. """
        cdef tuple shape_1 = (self.n_unit,)
        cdef tuple shape_2 = (self.n_unit, self.n_unit)
        cdef Py_ssize_t d_size = sizeof(double)
        cdef Py_ssize_t s_size = sizeof(state_t)
        cdef Py_ssize_t i_size = sizeof(int)
        self.bond_probs = array(shape=shape_2, itemsize=d_size, format='d')
        self.flip_probs = array(shape=shape_1, itemsize=d_size, format='d')
        self.cluster_biases = array(shape=shape_1, itemsize=d_size, format='d')
        self.bonds_state = array(
            shape=shape_2, itemsize=s_size, format=state_t_code)
        self.units_state = array(
            shape=shape_1, itemsize=s_size, format=state_t_code)
        self.clusters = array(shape=shape_1, itemsize=i_size, format='i')

    cdef void sample_bonds_given_units(SwendsenWangSampler self):
        cdef int i, j
        cdef state_t bond_mask
        for i in range(self.n_unit):
            for j in range(i):
                bond_mask = (self.units_state[i] * self.units_state[j] *
                             self.weights[i, j] > 0.)
                self.bonds_state[i, j] = (
                    self.rng.uniform() < self.bond_probs[i, j]) * bond_mask

    cdef void sample_units_given_bonds(SwendsenWangSampler self):
        cdef int i
        for i in range(self.n_unit):
            self.cluster_biases[i] = 2 * self.biases[i] * self.units_state[i]
        self.sample_flip_states_given_bonds()
        for i in range(self.n_unit):
            if self.flip_probs[i] > 0.:
                self.units_state[i] *= -1

    cpdef get_samples(SwendsenWangSampler self, state_t[:, :] samples,
                      state_t[:] init_state):
        cdef int i, s
        cdef int n_sample = samples.shape[0]
        if init_state is None:
            for i in range(self.n_unit):
                self.units_state[i] = 2 * (self.rng.uniform() > 0.5) - 1
        else:
            self.units_state[:] = init_state
        samples[0, :] = self.units_state
        for s in range(1, n_sample):
            self.sample_bonds_given_units()
            self.sample_units_given_bonds()
            samples[s, :] = self.units_state

    cdef int find_cluster_root(SwendsenWangSampler self, int i):
        """
        Returns index of root of unit cluster given a unit index.

        Also performs path compression i.e. updates index pointer at non-root
        entries to directly point at root rather than requiring multiple hops.

        Parameters
        ----------
        i : int
            Index of unit to find cluster root index for.
        """
        if self.clusters[i] < 0:
            return i
        else:
            # if non-root entry make sure points directly at root rather than
            # requiring multiple hops for quicker future access
            self.clusters[i] = self.find_cluster_root(self.clusters[i])
            return self.clusters[i]

    cdef void construct_clusters_and_update_biases(SwendsenWangSampler self):
        """
        Identify unit clusters from bond states and update cluster biases.

        Cluster biases updated so that entries for cluster root units
        correspond to sum of biases for all units in that cluster.
        """
        cdef int i, j, root_i, root_j
        # clusters initialised with all units to being in singleton clusters
        # i.e. all entries set to -1
        for i in range(self.n_unit):
            self.clusters[i] = -1
        # loop over lower diagonal of bond states matrix
        for i in range(1, self.n_unit):
            for j in range(i):
                # skip if bond inactive
                if self.bonds_state[i, j] == 0:
                    continue
                # get cluster root indices for two units
                root_i = self.find_cluster_root(i)
                root_j = self.find_cluster_root(j)
                # if not already in same cluster (i.e. same root) merge
                if root_i != root_j:
                    # establish which of the two clusters is larger
                    root_small, root_large =  (
                        (root_i, root_j)
                        if self.clusters[root_i] > self.clusters[root_j]
                        else (root_j, root_i)
                    )
                    # add bias of smaller cluster to larger
                    self.cluster_biases[root_large] += (
                        self.cluster_biases[root_small])
                    # add size of smaller cluster to larger
                    self.clusters[root_large] += self.clusters[root_small]
                    # redirect pointer of smaller cluster to larger
                    self.clusters[root_small] = root_large

    cdef void calc_flip_probs_and_sample(SwendsenWangSampler self):
        cdef int i, root
        for i in range(self.n_unit):
            self.flip_probs[i] = -2.
        # loop across all units and set flip probability and sample cluster state
        for i in range(self.n_unit):
            root = self.find_cluster_root(i)
            # check if probability of root calculated yet
            if self.flip_probs[root] == -2.:
                self.flip_probs[root] = (
                    1. / (1. + exp(-self.cluster_biases[root])))
                # sample flip sign and multiply by probability
                self.flip_probs[root] *= 2 * (self.rng.uniform() <
                                              self.flip_probs[root]) - 1
            # propagate (signed) probability to all units in cluster
            self.flip_probs[i] = self.flip_probs[root]

    cdef void sample_flip_states_given_bonds(SwendsenWangSampler self):
        """
        Compute unit flip state samples and probabilities given bond states.
        """
        self.construct_clusters_and_update_biases()
        self.calc_flip_probs_and_sample()
