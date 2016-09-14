# -*- coding: utf-8 -*-
"""
Unit tests for Boltzmann machine moment calculation code
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

import itertools as it
import numpy as np
import bmtools.exact.moments as mom


def calculate_bm_moments_test_1():
    W = np.array([[0., -4.5, -4.5], [-4.5, 0., 4.5], [-4.5, 4.5, 0.]])
    b = np.array([0.125, 0.125, -0.125])
    ps = calc_3_unit_probs(W, b)
    # calculate moments with native python code
    norm_const, expc_s, expc_ss = calc_first_and_second_moments(ps, 3)
    # calculate moments using sequential Cython code
    norm_const_sq, expc_s_sq, expc_ss_sq = (
        mom.calculate_moments_sequential(W, b))
    # calculate moments using parallel Cython code, 1 thread
    norm_const_p1, expc_s_p1, expc_ss_p1 = (
        mom.calculate_moments_parallel(W, b, n_thread=1))
    # calculate moments using parallel Cython code, 2 threads
    norm_const_p2, expc_s_p2, expc_ss_p2 = (
        mom.calculate_moments_parallel(W, b, n_thread=2))
    # calculate moments using parallel Cython code, 4 threads
    norm_const_p4, expc_s_p4, expc_ss_p4 = (
        mom.calculate_moments_parallel(W, b, n_thread=4))
    # calculate moments using parallel Cython code, in place updates
    norm_const_ip = np.empty(1)
    expc_s_ip = np.empty((1, 3))
    expc_ss_ip = np.empty((1, 3, 3))
    mom.calculate_moments_parallel(W, b, n_thread=1,
                                   norm_consts=norm_const_ip,
                                   first_moms=expc_s_ip,
                                   second_moms=expc_ss_ip)
    assert np.allclose(norm_const, norm_const_sq), (
        'Normalisation constant calculated using sequential code incorrect')
    assert np.allclose(expc_s, expc_s_sq), (
        'First moment calculated using sequential code incorrect')
    assert np.allclose(expc_ss, expc_ss_sq), (
        'Second moment calculated using sequential code incorrect')
    assert np.allclose(norm_const, norm_const_p1), (
        'Normalisation calculated using parallel code (1 thread) incorrect')
    assert np.allclose(expc_s, expc_s_p1), (
        'First moment calculated using parallel code (1 thread) incorrect')
    assert np.allclose(expc_ss, expc_ss_p1), (
        'Second moment calculated using parallel code (1 thread) incorrect')
    assert np.allclose(norm_const, norm_const_p2), (
        'Normalisation calculated using parallel code (2 thread) incorrect')
    assert np.allclose(expc_s, expc_s_p2), (
        'First moment calculated using parallel code (2 thread) incorrect')
    assert np.allclose(expc_ss, expc_ss_p2), (
        'Second moment calculated using parallel code (2 thread) incorrect')
    assert np.allclose(norm_const, norm_const_p4), (
        'Normalisation calculated using parallel code (4 thread) incorrect')
    assert np.allclose(expc_s, expc_s_p4), (
        'First moment calculated using parallel code (4 thread) incorrect')
    assert np.allclose(expc_ss, expc_ss_p4), (
        'Second moment calculated using parallel code (4 thread) incorrect')
    assert np.allclose(norm_const, norm_const_ip), (
        'Normalisation calculated using parallel code (in-place) incorrect')
    assert np.allclose(expc_s, expc_s_ip), (
        'First moment calculated using parallel code (in-place) incorrect')
    assert np.allclose(expc_ss, expc_ss_ip), (
        'Second moment calculated using parallel code (in-place) incorrect')


def calculate_bm_moments_test_2():
    W = np.array([[0., -4.5, -4.5], [-4.5, 0., 4.5], [-4.5, 4.5, 0.]])
    b = np.array([0.125, 0.125, -0.125])
    ps = calc_3_unit_probs(W, b)
    data = np.array([[1, 1], [-1, -1]], dtype=np.int8)
    log_liks = np.zeros(2)
    for i, s in enumerate(it.product(*[[-1, 1]]*3)):
        s = np.array(s)
        for j, v in enumerate(data):
            if all(s[1:] == v):
                log_liks[j] += ps[i]
    log_liks = np.log(log_liks)
    log_liks -= np.log(ps.sum())
    log_liks_p1 = mom.log_likelihood(data, W, b, n_thread=1)
    log_liks_p2 = mom.log_likelihood(data, W, b, n_thread=2)
    log_liks_p4 = mom.log_likelihood(data, W, b, n_thread=4)
    assert np.allclose(log_liks, log_liks_p1), (
        'Log likelihood calculation (1 thread) incorrect')
    assert np.allclose(log_liks, log_liks_p2), (
        'Log likelihood calculation (2 threads) incorrect')
    assert np.allclose(log_liks, log_liks_p4), (
        'Log likelihood calculation (4 threads) incorrect')


def calc_first_and_second_moments(ps, n_unit):
    expc_s = np.zeros(n_unit)
    expc_ss = np.zeros((n_unit, n_unit))
    for i, s in enumerate(it.product(*[[-1, 1]]*n_unit)):
        s = np.array(s)
        expc_s += s * ps[i]
        expc_ss += s[:, None] * s[None, :] * ps[i]
    norm_const = ps.sum()
    expc_s /= norm_const
    expc_ss /= norm_const
    return norm_const, expc_s, expc_ss


def calc_3_unit_probs(W, b):
   return np.array([
        np.exp(+W[0, 1] + W[0, 2] + W[1, 2] - b[0] - b[1] - b[2]),
        np.exp(+W[0, 1] - W[0, 2] - W[1, 2] - b[0] - b[1] + b[2]),
        np.exp(-W[0, 1] + W[0, 2] - W[1, 2] - b[0] + b[1] - b[2]),
        np.exp(-W[0, 1] - W[0, 2] + W[1, 2] - b[0] + b[1] + b[2]),
        np.exp(-W[0, 1] - W[0, 2] + W[1, 2] + b[0] - b[1] - b[2]),
        np.exp(-W[0, 1] + W[0, 2] - W[1, 2] + b[0] - b[1] + b[2]),
        np.exp(+W[0, 1] - W[0, 2] - W[1, 2] + b[0] + b[1] - b[2]),
        np.exp(+W[0, 1] + W[0, 2] + W[1, 2] + b[0] + b[1] + b[2])
    ])


if __name__ == '__main__':
    calculate_bm_moments_test_1()
    calculate_bm_moments_test_2()
