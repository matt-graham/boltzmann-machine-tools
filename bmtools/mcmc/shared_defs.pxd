"""
Shared definitions file for Boltzmann machine samplers
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

# Use exponential function from C math library as faster than native
# implementation
cdef extern from 'math.h':
    double exp(double x) nogil
    double log(double x) nogil

# Standard type for Boltzmann machine unit states which are signed binary
# values (-1/+1). Also specify a type code for use in initialising
# cython arrays with this type.
ctypedef signed char state_t
cdef char* state_t_code = 'c'
#ctypedef int state_t
#cdef char state_t_code = 'i'
