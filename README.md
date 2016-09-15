# Boltzmann machine tools

Python tools for analysing Boltzmann machine / Ising spin models. Includes functions for calculating exact moments of small Boltzmann machine models, exact maximum-likelihood gradient based training and various MCMC sampler implementations.

## Installation

Requires [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/) and [CVXPY](http://www.cvxpy.org/en/latest/).

If [Cython](http://cython.org/) is installed, the Cython extension modules can be build from the Cython `.pyx` source files using

```
python setup.py build_ext -use-cython
```

If Cython is not installed, the generated C source of the modules is also included and running `setup.py` without the `-use-cython` option will use the C source instead.

The following additional optional arguments can also be used with `setup.py`:

```
  -debug            Export GDB debug information when compiling.
  -use-gcc-opts     Use GCC compiler optimisations for quicker but less safe
                    mathematical operations.
  -use-cython-opts  Add extra Cython compile directives for quicker but less
                    safe array access. Requires -use-cython to be set to have
                    an effect.
  -use-openmp       Compiles with OpenMP support.
```
