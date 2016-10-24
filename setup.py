# -*- coding: utf-8 -*-
"""Set up script for Boltzmann machine tools package.

Builds Cython modules either from .pyx files or .c files.
"""

from setuptools import setup, Extension
import argparse
import sys

parser = argparse.ArgumentParser(
    description='Boltzmann machine tools setup')
parser.add_argument('-debug', action='store_true', default=False,
                    help='Export GDB debug information when compiling.')
parser.add_argument('-use-cython', action='store_true', default=False,
                    help='Use Cython to compile from .pyx files.')
parser.add_argument('-use-gcc-opts', action='store_true', default=False,
                    help='Use GCC compiler optimisations for quicker but '
                         'less safe mathematical operations.')
parser.add_argument('-use-cython-opts', action='store_true', default=False,
                    help='Add extra Cython compile directives for quicker '
                         'but less safe array access. Requires -use-cython '
                         'to be set to have an effect.')
parser.add_argument('-use-openmp', action='store_true', default=False,
                    help='Compiles with OpenMP support.')
# hack to get both argparser help and distutils help displaying
if '-h' in sys.argv:
    sys.argv.remove('-h')
    help_flag = '-h'
elif '--help' in sys.argv:
    sys.argv.remove('--help')
    help_flag = '--help'
else:
    help_flag = None

args, unknown = parser.parse_known_args()

# remove custom arguments from sys.argv to avoid conflicts with distutils
for action in parser._actions:
    for opt_str in action.option_strings:
        try:
            sys.argv.remove(opt_str)
        except ValueError:
            pass
# if a help flag was found print parser help string then readd so distutils
# help also displayed
if help_flag:
    parser.print_help()
    sys.argv.append(help_flag)

ext = '.pyx' if args.use_cython else '.c'

extra_compile_args = ['-O3', '-ffast-math'] if args.use_gcc_opts else []
extra_link_args = []

if args.use_openmp:
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')


if args.use_cython and args.use_cython_opts:
    compiler_directives = {
        'boundscheck': False,  # don't check for invalid indexing
        'wraparound': False,  # assume no negative indexing
        'cdivision': True,  # don't check for zero division
        'initializedcheck': False,  # don't check memory view init
        'embedsignature': True  # include call signature in docstrings
    }
else:
    compiler_directives = {}

ext_modules = [
    Extension('bmtools.mcmc.randomkit_wrapper',
              ['bmtools/mcmc/randomkit_wrapper' + ext,
               'bmtools/mcmc/randomkit.c'],
              extra_compile_args=extra_compile_args),
    Extension('bmtools.mcmc.sequential_gibbs_sampler',
              ['bmtools/mcmc/sequential_gibbs_sampler' + ext],
              extra_compile_args=extra_compile_args),
    Extension('bmtools.mcmc.irreversible_two_unit_sampler',
              ['bmtools/mcmc/irreversible_two_unit_sampler' + ext],
              extra_compile_args=extra_compile_args),
    Extension('bmtools.mcmc.swendsen_wang_helpers',
              ['bmtools/mcmc/swendsen_wang_helpers' + ext],
              extra_compile_args=extra_compile_args),
    Extension('bmtools.mcmc.swendsen_wang_sampler',
              ['bmtools/mcmc/swendsen_wang_sampler' + ext],
              extra_compile_args=extra_compile_args),
    Extension('bmtools.mcmc.mc_expectations',
              ['bmtools/mcmc/mc_expectations' + ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('bmtools.exact.helpers',
              ['bmtools/exact/helpers' + ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('bmtools.exact.moments',
              ['bmtools/exact/moments' + ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('bmtools.exact.likelihood',
              ['bmtools/exact/likelihood' + ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args),
    Extension('bmtools.exact.variational',
              ['bmtools/exact/variational' + ext],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args)
]

if args.use_cython:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules,
                            compiler_directives=compiler_directives,
                            gdb_debug=args.debug)

packages = [
    'bmtools',
    'bmtools.exact',
    'bmtools.mcmc',
    'bmtools.relaxations'
]

setup(
    name='bmtools',
    author='Matt Graham',
    description='Boltzmann machine tools',
    url="https://github.com/matt-graham/boltzmann-machine-tools",
    packages=packages,
    ext_modules=ext_modules,
)
