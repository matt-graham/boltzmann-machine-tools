"""
========================
Random kit wrapper class
========================

Implementation of a basic Cython wrapper class around the 'Random kit' library
by Jean-Sebastien Roy. Intended for use in other Cython modules as a more
robust replacement for C stdlib rand().
"""

__authors__ = "Matt Graham"
__copyright__ = "Copyright 2015, Matt Graham"
__license__ = "MIT"

cimport randomkit_wrapper
from libc cimport stdlib

cdef class RandomKit:

    """
    Basic wrapper around 'Random kit' for pseudorandom number generation.

    Intended for use in other Cython modules as a more robust replacement for
    C stdlib rand().
    """

    def __cinit__(RandomKit self):
        self.state = <rk_state*> stdlib.malloc(sizeof(rk_state))
        if (self.state == NULL):
            raise MemoryError

    def __init__(RandomKit self, unsigned long seed):
        rk_seed(seed, self.state)

    def __dealloc__(RandomKit self):
        if self.state:
            stdlib.free(self.state)

    cdef unsigned long integer(RandomKit self, unsigned long maximum):
        """
        Returns a random integer in range 0 and maximum inclusive.

        Parameters
        ----------
            maximum : unsigned long
                Maximum of integer range to sample from.
        """
        return rk_interval(maximum, self.state)

    cdef double uniform(RandomKit self):
        """
        Returns a sample from a uniform distribution over [0,1].
        """
        return rk_double(self.state)

    cdef double gaussian(RandomKit self):
        """
        Returns a sample from a zero-mean unit-variance Gaussian distribution.
        """
        return rk_gauss(self.state)
