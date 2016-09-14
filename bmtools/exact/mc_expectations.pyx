"""
=========================================
Bolzmann machine Monte-Carlo expectations
=========================================

Tool for calculating the root-mean-square error between pre-calculated true
first and second moments of a Boltzmann machine and Monte Carlo estimates of
these values from samples.
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

cdef extern from 'math.h':
    double sqrt(double x)

cpdef calculate_incremental_expectations_errors(
        double[:, :] samples, double[:] expc_s_true,
        double[:, :] expc_ss_true, double[:] expc_s, double[:, :] expc_ss,
        double[:] dist_s, double[:] dist_ss):
    cdef int s, i, j, n_sample, n_unit
    cdef double d
    n_sample = samples.shape[0]
    n_unit = samples.shape[1]
    for s in range(n_sample):
        if s == 0:
            for i in range(n_unit):
                expc_s[i] = 0.
                expc_ss[i, i] = 1.
                for j in range(i):
                    expc_ss[i, j] = 0.
        dist_s[s] = 0.
        dist_ss[s] = 0.
        for i in range(n_unit):
            expc_s[i] += (samples[s, i] - expc_s[i]) / (s + 1.)
            for j in range(i):
                expc_ss[i, j] += (samples[s, i] * samples[s, j]
                                  - expc_ss[i, j]) / (s + 1.)
                d = expc_ss[i, j] - expc_ss_true[i, j]
                dist_ss[s] += 2 * d * d
            d = expc_s[i] - expc_s_true[i]
            dist_s[s] += d * d
        dist_s[s] = sqrt(dist_s[s])
        dist_ss[s] = sqrt(dist_ss[s])
    for i in range(1, n_unit):
        for j in range(i):
            expc_ss[j, i] = expc_ss[i, j]
