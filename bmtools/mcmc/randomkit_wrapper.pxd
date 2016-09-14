"""
========================
Random kit wrapper class
========================

Implementation of a basic Cython wrapper class around the 'Random kit' library
by Jean-Sebastien Roy. Intended for use in other Cython modules as a more
robust replacement for C stdlib rand().
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

cdef extern from 'randomkit.h':
    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss
    void rk_seed(unsigned long seed, rk_state *state)
    double rk_gauss(rk_state *state)
    double rk_double(rk_state *state)
    long rk_interval(unsigned long maximum, rk_state *state)

cdef class RandomKit:
    cdef rk_state *state
    cdef unsigned long integer(RandomKit self, unsigned long maximum)
    cdef double uniform(RandomKit self)
    cdef double gaussian(RandomKit self)
