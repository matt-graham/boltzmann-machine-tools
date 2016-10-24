# -*- coding: utf-8 -*-
"""Bolzmann machine representations.

Wrapper classes for representing Boltzmann machine with both signed and
unsigned representations and calculating associated moments.
"""

import numpy as np
import bmtools.exact.moments as mom


class SignedBoltzmannMachine(object):
    """Boltzmann machine using signed binary state {-1, +1}^D."""

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def calculate_moments(self, force=False, num_threads=4):
        Z, expc_s, expc_ss = mom.calculate_moments_parallel(
            self.weights, self.biases, force, num_threads
        )
        expc_s = np.array(expc_s)
        expc_ss = np.array(expc_ss)
        return np.log(Z), expc_s, expc_ss

    def to_unsigned(self):
        return UnsignedBoltzmannMachine(signed_rep=self)


class UnsignedBoltzmannMachine(object):
    """Boltzmann machine using unsigned binary state {0, 1}^D."""

    def __init__(self, weights=None, biases=None, signed_rep=None):
        if signed_rep is None:
            self.signed_rep = SignedBoltzmannMachine(
                weights / 4., (biases + 0.5 * weights.sum(0)) / 2.)
            self.weights = weights
            self.biases = biases
        elif weights is not None or biases is not None:
            raise ValueError(
                'Either signed_rep or parameters should be given not both.')
        else:
            self.signed_rep = signed_rep
            self.weights = 4 * signed_rep.weights
            self.biases = 2 * (signed_rep.biases - signed_rep.weights.sum(0))

    def calculate_moments(self, force=False, num_threads=4):
        log_Z_, expc_s, expc_ss = self.signed_rep.calculate_moments(
            force, num_threads)
        log_Z = (
            log_Z_ + self.signed_rep.biases.sum() -
            0.5 * self.signed_rep.weights.sum()
        )
        expc_u = (1. + expc_s) / 2.
        expc_uu = (expc_ss + expc_s[:, None] + expc_s[None, :] + 1) / 4.
        return log_Z, expc_u, expc_uu

    def to_signed(self):
        return self.signed_rep
