"""
==============================
Swendsen-Wang helper functions
==============================

Cython implementations of helpers for performing clustering and flip
probability calculation on a Boltzmann machine / Ising model augmented with
bond variables for use in a Swendsen-Wang sampling scheme.

Code is largely a direct port of MATLAB/C code by Iain Murray at
    http://homepages.inf.ed.ac.uk/imurray2/code/swendsen_wang/
This adapted code for performing the clustering from:
     A fast Monte Carlo algorithm for site or bond percolation
     M. E. J. Newman and R. M. Ziff
     arXiv:cond-mat/0101295 v2 8th April 2001

I have changed naming to suit python conventions and also to reflect my
attempts to understand what code is doing (also explained in comments) - no
guarantees here that this is a particularly sensible or even wholly accurate
description of the original intention of the algorithm.

This implementation makes no assumptions about the connectivity of units (i.e.
it assumes full connectivity) or strength of connections - more efficient
implementations could be made to exploit known structure to the connectivity.
See also comments in Iain's code on this issue.
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

cimport randomkit_wrapper as rk
from cython cimport view
include "shared_defs.pxd"

cdef int find_cluster_root(int[:] clusters, int i):
    """
    Returns index of root of unit cluster given a unit index.

    Also performs path compression i.e. updates index pointer at non-root
    entries to directly point at root rather than requiring multiple hops.

    Parameters
    ----------
    clusters : int[:]
        One-dimensional array of indices of cluster root indices and sizes.
    i : int
        Index of unit to find cluster root index for.
    """
    if clusters[i] < 0:
        return i
    else:
        # if non-root entry make sure points directly at root rather than
        # requiring multiple hops for quicker future access
        clusters[i] = find_cluster_root(clusters, clusters[i])
        return clusters[i]

# use cpdef rather than cdef here to allow external calling by test suite
cpdef int[:] construct_clusters_and_update_biases(
        unsigned char[:, :] bond_states, double[:] biases):
    """
    Identify unit clusters from current bond states and update biases array.

    Biases updated so that entries for cluster root units correspond to sum of
    biases for all units in that cluster.

    Returns clusters array, a vector of length n_unit with two entry types
      * if clusters[i] < 0 then i is the root unit of a cluster and
        -1*clusters[i] is the size (number of members) of the corresponding
        cluster
      * else if clusters[i] >= 0 then is is a non-root unit of a cluster and
        clusters[i] is the index of the corresponding cluster root unit.

    Parameters
    ----------
    bond_states : unsigned char[:, :]
        Two-dimensional array of binary (0/1) bond states, with array being of
        shape (n_unit, n_unit) and the (i,j)th entry corresponding to the
        bond between units i and j being active. Only the lower diagonal terms
        (i.e. j < i where j is the column index and i the row index) are used.
    biases : double[:]
        One-dimensional array of bias parameters values for each unit.
    """
    cdef int i, j, root_i, root_j
    cdef int n_unit = bond_states.shape[0]
    # clusters initialised with all units to being in singleton clusters
    # i.e. all entries set to -1
    cdef int[:] clusters = view.array(
        shape=(n_unit,), itemsize=sizeof(int), format='i')
    for i in range(n_unit):
        clusters[i] = -1
    # loop over lower diagonal of bond states matrix
    for i in range(1, n_unit):
        for j in range(i):
            # skip if bond inactive
            if bond_states[i,j] == 0:
                continue
            # get cluster root indices for two units
            root_i = find_cluster_root(clusters, i)
            root_j = find_cluster_root(clusters, j)
            # if not already in same cluster (i.e. same root) merge
            if root_i != root_j:
                # establish which of the two clusters is larger
                root_small, root_large =  (
                    (root_i, root_j)
                    if clusters[root_i] > clusters[root_j]
                    else (root_j, root_i)
                )
                # add bias of smaller cluster to larger
                biases[root_large] += biases[root_small]
                # add size of smaller cluster to larger
                clusters[root_large] += clusters[root_small]
                # redirect pointer of smaller cluster to larger
                clusters[root_small] = root_large
    return clusters

cpdef double[:] calc_flip_probs_and_sample(
        int[:] clusters, double[:] biases, rk.RandomKit rng):
    """
    Returns probabilities of clustered units flips with signs flip samples.

    The flip probability of each cluster is calculated from the provided
    accumulated bias values, a flip sign (-1/1) is then sampled given this
    calculated flip probability, multiplied by the probability and this
    signed value then stored in an entry of the return array for each unit
    in a cluster.

    Therefore for an element of flip_probs[i] of the return value,
        abs(flip_probs[i]) is the probability of unit i flipping
        sign(flip_probs[i]) is the corresponding flip state sample for unit i

    Parameters
    ----------
    clusters : int[:]
        Array of length n_unit which encodes cluster structure (see above).
    biases: double[:]
        Array of accumulated bias values for clusters (see above).
    rng: RandomKit
        Instance of a RandomKit wrapper for pseudorandom number generation.
    """
    cdef int i, root
    cdef int n_unit = clusters.shape[0]
    # initialise flip probs array to -2 values as outside valid range of
    # signed probability values
    cdef double[:] flip_probs = view.array(
        shape=(n_unit,), itemsize=sizeof(double), format='d')
    for i in range(n_unit):
        flip_probs[i] = -2.
    # loop across all units and set flip probability and sample cluster state
    for i in range(n_unit):
        root = find_cluster_root(clusters, i)
        # check if probability of root calculated yet
        if flip_probs[root] == -2.:
            flip_probs[root] = 1. / (1. + exp(-biases[root]))
            # sample flip sign and multiply by probability
            flip_probs[root] *= 2 * (rng.uniform() < flip_probs[root]) - 1
        # propagate (signed) probability to all units in cluster
        flip_probs[i] = flip_probs[root]
    return flip_probs

cpdef double[:] sample_flip_states_given_bonds(
        unsigned char[:, :] bond_states,  double[:] biases, rk.RandomKit rng):
    """
    Returns flip state samples and probabilities given bond states and biases.

    This is intended to be the main externally used function, wrapping the
    internal separate cluster construction and flip probability / state
    calculation, with some basic parameter checking.

    Return value is a MemoryView object which can be used in place of a numpy
    array in many functions and easily accessed as one using numpy.array().

    Parameters
    ----------
    bond_states : unsigned char[:, :] (np.uint8[:, :])
        Two-dimensional array of binary (0/1) bond states, with array being of
        shape (n_unit, n_unit) and the (i,j)th entry corresponding to the
        bond between units i and j being active. Only the lower diagonal terms
        (i.e. j < i where j is the column index and i the row index) are used.
    biases : double[:]
        One-dimensional array of flip bias parameters values for each unit.
        These values will be altered in place so a copy should be kept if
        needed elsewhere.
    rng: RandomKit
        Instance of a RandomKit wrapper for pseudorandom number generation.
    """
    assert bond_states.shape[0] == bond_states.shape[1], (
        'bond_states array must be a square matrix'
    )
    assert bond_states.shape[0] == biases.shape[0], (
        'biases must be 1D array with length matching bond_states'
    )
    cdef int[:] clusters = construct_clusters_and_update_biases(
        bond_states, biases)
    cdef double[:] flip_probs = calc_flip_probs_and_sample(
        clusters, biases, rng)
    return flip_probs
